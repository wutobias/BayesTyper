#!/usr/bin/env pythonf

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================
import numpy as np
from .vectors import BaseVector
import math
import copy
import ray

from .constants import _TIMEOUT, _VERBOSE, _EPSILON
from .ray_tools import retrieve_failed_workers

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================


class LikelihoodVectorized(object):

    def __init__(
        self,
        pvec_list,
        targetcomputer,
        N_sys_per_batch = 4,
        three_point = False,
        ):

        self._three_point = three_point

        self._N_sys_per_batch = N_sys_per_batch

        self.pvec_list = pvec_list
        self.targetcomputer = targetcomputer

        self._N_pvec = len(pvec_list)

        self._initialize_systems()

    def _initialize_systems(self):

        self.N_parms   = 0
        self.pvec_idxs = list()
        self.jacobian  = list()

        start = 0
        stop  = 0
        self.pvec_list_fwd = list()
        self.openmm_system_fwd_dict = dict()
        if self._three_point:
            self.openmm_system_bkw_dict = dict()
            self.pvec_list_bkw = list()
        self.openmm_system_dict = dict()
        self.parm_idx_sysname_dict = dict()
        self.sysname_parm_idx_dict = dict()
        parm_idx = 0
        for i in range(self._N_pvec):
            pvec    = self.pvec_list[i]
            for sys in pvec.parameter_manager.system_list:
                if not sys.name in self.openmm_system_dict:
                    self.openmm_system_dict[sys.name] = [sys.openmm_system]
                else:
                    if self.openmm_system_dict[sys.name][0] != sys.openmm_system:
                        raise ValueError(
                            f"Parameter vectors for systems {sys.name} are not connected to same system object."
                            )

            remove_sys_idx_dict = dict()
            all_systems_set = set(pvec.parameter_manager.system_idx_list)
            _parm_idx = 0
            for f in range(pvec.force_group_count):
                ranks = pvec.allocations.index([f])[0]
                valid_system_list = list()
                for r in ranks:
                    valids = np.where(
                        pvec.parameter_manager.force_ranks == r
                    )
                    valid_system_list.extend(
                        pvec.parameter_manager.system_idx_list[valids].tolist()
                    )
                to_be_deleted = all_systems_set.difference(set(valid_system_list))
                for _ in range(pvec.parameters_per_force_group):
                    remove_sys_idx_dict[_parm_idx] = list(to_be_deleted)[::-1]
                    _parm_idx += 1

            start   = stop
            stop    = start+pvec.size
            self.pvec_idxs.append([start, stop])            
            self.N_parms += pvec.size
            self.jacobian.extend(
                pvec.jacobian.tolist()
                )

            self.pvec_list_fwd.append(list())
            if self._three_point:
                self.pvec_list_bkw.append(list())
            for _parm_idx in range(pvec.size):
                pvec_cp_fwd = pvec.copy(
                    include_systems=True, 
                    rebuild_to_old_systems=False
                    )
                pvec_cp_fwd.remove_system(
                    remove_sys_idx_dict[_parm_idx]
                    )

                self.pvec_list_fwd[-1].append(pvec_cp_fwd)
                self.openmm_system_fwd_dict[parm_idx] = list()
                if self._three_point:
                    pvec_cp_bkw = pvec_cp_fwd.copy(
                        include_systems=True, 
                        rebuild_to_old_systems=False
                        )
                    ### Note we don't have to remove systems
                    ### since we are copying from `pvec_cp_fwd`
                    self.pvec_list_bkw[-1].append(pvec_cp_bkw)
                    self.openmm_system_bkw_dict[parm_idx] = list()

                N_sys = len(pvec_cp_fwd.parameter_manager.system_list)
                fwd_dict  = dict()
                bkw_dict  = dict()
                for sys_idx in range(N_sys):
                    sys_fwd = pvec_cp_fwd.parameter_manager.system_list[sys_idx]
                    fwd_dict[sys_fwd.name] = [sys_fwd.openmm_system]
                    if self._three_point:
                        sys_bkw = pvec_cp_bkw.parameter_manager.system_list[sys_idx]
                        assert sys_bkw.name == sys_fwd.name
                        bkw_dict[sys_bkw.name] = [sys_bkw.openmm_system]

                    if len(fwd_dict) == self._N_sys_per_batch:
                        self.openmm_system_fwd_dict[parm_idx].append(fwd_dict)
                        fwd_dict = dict()
                        if self._three_point:
                            self.openmm_system_bkw_dict[parm_idx].append(bkw_dict)
                            bkw_dict = dict()

                    if sys_fwd.name in self.parm_idx_sysname_dict:
                        self.parm_idx_sysname_dict[sys_fwd.name].append(parm_idx)
                    else:
                        self.parm_idx_sysname_dict[sys_fwd.name] = [parm_idx]

                    if parm_idx in self.sysname_parm_idx_dict:
                        self.sysname_parm_idx_dict[parm_idx].append(sys_fwd.name)
                    else:
                        self.sysname_parm_idx_dict[parm_idx] = [sys_fwd.name]

                if len(fwd_dict) > 0:
                    self.openmm_system_fwd_dict[parm_idx].append(fwd_dict)
                    if self._three_point:
                        self.openmm_system_bkw_dict[parm_idx].append(bkw_dict)

                parm_idx += 1

        self.jacobian = np.array(self.jacobian)

    @property
    def pvec(self):

        pvec_list = list()
        for i in range(self._N_pvec):
            pvec_list.extend(
            self.pvec_list[i].copy()[:].tolist()
            )
        return np.array(pvec_list)

    def apply_changes(
        self, 
        vec, 
        grad=True, 
        parm_idx_list=None,
        grad_diff=_EPSILON):

        if vec.size != self.N_parms:
            raise ValueError(
                f"Length of vec is {vec.size} but must be {self.N_parms}"
                )

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        counts = 0
        for i in range(self._N_pvec):
            pvec       = self.pvec_list[i]
            start_stop = self.pvec_idxs[i]
            pvec[:]    = vec[start_stop[0]:start_stop[1]]
            pvec.apply_changes()
            if grad:
                for j in range(pvec.size):
                    if j in parm_idx_list:
                        pvec_fwd = self.pvec_list_fwd[i][j]
                        pvec_fwd[:] = vec[start_stop[0]:start_stop[1]]
                        pvec_fwd[j] += grad_diff
                        pvec_fwd.apply_changes()
                        if self._three_point:
                            pvec_bkw = self.pvec_list_bkw[i][j]
                            pvec_bkw[:] = vec[start_stop[0]:start_stop[1]]
                            pvec_bkw[j] -= grad_diff
                            pvec_bkw.apply_changes()
                    counts += 1

    def _results_dict_to_likelihood(self, results_dict_all):

        logP_likelihood = 0.
        for sys_name in results_dict_all:
            for results_dict in results_dict_all[sys_name]:
                for target_idx in results_dict["rss"]:
                    rss = results_dict["rss"][target_idx]
                    log_norm_factor = results_dict["log_norm_factor"][target_idx]
                    logP_likelihood += -log_norm_factor - 0.5 * np.log(2.*np.pi) - 0.5 * rss

        return logP_likelihood

    def __call__(
        self,
        vec,
        parm_idx_list=None):

        self.apply_changes(
            vec, grad=False,
            parm_idx_list=parm_idx_list)

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        sysname_run_list = list()
        worker_id_dict = dict()
        for sysname in self.openmm_system_dict:
            if sysname in sysname_run_list:
                continue
            _parm_idx_list = self.parm_idx_sysname_dict[sysname]
            for parm_idx in parm_idx_list:
                if parm_idx in _parm_idx_list:
                    worker_id = self.targetcomputer.__call__(
                        {sysname:self.openmm_system_dict[sysname]}, False)
                    worker_id_dict[worker_id] = sysname
                    sysname_run_list.append(sysname)
                    break
            
        worker_id_list = list(worker_id_dict.keys())
        logP_likelihood = 0.
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(
                worker_id_list, timeout=_TIMEOUT)
            failed = len(worker_id) == 0
            if not failed:
                try:
                    _logP_likelihood = ray.get(
                        worker_id[0], timeout=_TIMEOUT)
                except:
                    failed = True
                if not failed:
                    logP_likelihood += _logP_likelihood
                    del worker_id_dict[worker_id[0]]
            if failed:
                if len(worker_id) > 0:
                    if worker_id[0] not in worker_id_list:
                        worker_id_list.append(worker_id[0])
                resubmit_list = retrieve_failed_workers(worker_id_list)
                for worker_id in resubmit_list:
                    ray.cancel(worker_id, force=True)
                    key = worker_id_dict[worker_id]
                    del worker_id_dict[worker_id]
                    value = self.openmm_system_dict[key]
                    worker_id = self.targetcomputer.__call__({key:value}, False)
                    worker_id_dict[worker_id] = key
                worker_id_list = list(worker_id_dict.keys())

        return logP_likelihood

    def grad(
        self, 
        vec, 
        parm_idx_list=None,
        grad_diff=_EPSILON, 
        use_jac=False,
        ):

        import numpy as np

        if not self._three_point:
            grad_diff *= 2.

        self.apply_changes(
            vec, grad=True, 
            parm_idx_list=parm_idx_list, 
            grad_diff=grad_diff)

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        worker_id_dict = dict()
        for parm_idx in self.openmm_system_fwd_dict:
            if not parm_idx in parm_idx_list:
                continue
            for sys_idx, openmm_system_dict in enumerate(self.openmm_system_fwd_dict[parm_idx]):
                worker_id = self.targetcomputer.__call__(openmm_system_dict, False)
                worker_id_dict[worker_id] = [parm_idx], 1., sys_idx
            if self._three_point:
                for sys_idx, openmm_system_dict in enumerate(self.openmm_system_bkw_dict[parm_idx]):
                    worker_id = self.targetcomputer.__call__(openmm_system_dict, False)
                    worker_id_dict[worker_id] = [parm_idx], -1., sys_idx
        if not self._three_point:
            sysname_run_list = list()
            for sysname in self.openmm_system_dict:
                if sysname in sysname_run_list:
                    continue
                _parm_idx_list = self.parm_idx_sysname_dict[sysname]
                for parm_idx in parm_idx_list:
                    if parm_idx in _parm_idx_list:
                        worker_id = self.targetcomputer.__call__(
                            {sysname:self.openmm_system_dict[sysname]}, False)
                        worker_id_dict[worker_id] = _parm_idx_list, -1., sysname
                        sysname_run_list.append(sysname)
                        break

        grad = np.zeros(self.N_parms, dtype=float)
        worker_id_list = list(worker_id_dict.keys())
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(
                worker_id_list, timeout=_TIMEOUT)
            failed = len(worker_id) == 0
            if not failed:
                try:
                    _logP_likelihood = ray.get(
                        worker_id[0], timeout=_TIMEOUT)
                except:
                    failed = True
                if not failed:
                    _parm_idx_list, sign, sys_idx = worker_id_dict[worker_id[0]]
                    for parm_idx in parm_idx_list:
                        if parm_idx in _parm_idx_list:
                            grad[parm_idx] += sign * _logP_likelihood
                    del worker_id_dict[worker_id[0]]
            if failed:
                if len(worker_id) > 0:
                    if worker_id[0] not in worker_id_list:
                        worker_id_list.append(worker_id[0])
                resubmit_list = retrieve_failed_workers(worker_id_list)
                for worker_id in resubmit_list:
                    ray.cancel(worker_id, force=True)
                    parm_idx, sign, sys_idx = worker_id_dict[worker_id]
                    del worker_id_dict[worker_id]
                    if isinstance(sys_idx, int):
                        if sign > 0.:
                            openmm_system_dict = self.openmm_system_fwd_dict[parm_idx[0]][sys_idx]
                        else:
                            openmm_system_dict = self.openmm_system_bkw_dict[parm_idx[0]][sys_idx]
                        worker_id = self.targetcomputer.__call__(openmm_system_dict, False)
                    else:
                        sysname = sys_idx
                        worker_id = self.targetcomputer.__call__({sysname:self.openmm_system_dict[sysname]}, False)
                        _parm_idx_list = self.parm_idx_sysname_dict[sysname]
                        worker_id_dict[worker_id] = _parm_idx_list, -1., sysname

                    worker_id_dict[worker_id] = parm_idx, sign, sys_idx
                worker_id_list = list(worker_id_dict.keys())

        if use_jac:
            grad -= np.log(self.jacobian)

        if self._three_point:
            grad *= 1./(grad_diff * 2.)
        else:
            grad *= 1./grad_diff

        return grad


class ObjectiveFunction(object):

    def __init__(self, N_tgt):

        self.N_tgt    = N_tgt
        self.sig2_vec = BaseVector(dtype=np.float64)
        self.sig2_vec.append(np.ones(N_tgt, dtype=float))

    def compute(self, diffs, squared=False):

        sig2           = self.sig2_vec.vector_values

        if sig2.size != diffs.size:
            raise ValueError(
                f"Length of `diffs` must be {sig2.size}, but is {diffs.size}."
                )
        
        if squared:
            weighted_diffs = diffs/sig2
        else:
            weighted_diffs = diffs**2/sig2
        rss            = math.fsum(weighted_diffs)
        return rss

    def __call__(self, diffs):

        return self.compute(diffs)


class LogGaussianLikelihood(ObjectiveFunction):

    def __init__(self, N_tgt):

        super().__init__(N_tgt)

        self.is_log = True

    def compute(self, diffs, squared=False):

        pre  = -0.5 * self.N_tgt * np.log(2.*np.pi)
        sig2 = self.sig2_vec.vector_values

        if sig2.size != diffs.size:
            raise ValueError(
                f"Length of `diffs` must be {sig2.size}, but is {diffs.size}."
                )

        if np.any(sig2 <= 0.):
            L = -np.inf
        else:
            pre = pre - 0.5 * math.fsum(np.log(sig2))
            if squared:
                exp = -0.5 * math.fsum(diffs / sig2)
            else:
                exp = -0.5 * math.fsum(diffs**2 / sig2)
            L   = pre + exp

        return L


class GaussianLikelihood(LogGaussianLikelihood):

    def __init__(self, N_tgt):

        super().__init__(N_tgt)

        self.is_log = False

    def compute(self, diffs, squared=False):

        return np.exp(super().compute(diffs, squared))