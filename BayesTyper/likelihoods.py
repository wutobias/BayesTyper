#!/usr/bin/env python

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

        self.pvec_list = pvec_list
        self.targetcomputer = targetcomputer

        self.N_pvec = len(pvec_list)

        self.N_parms   = 0
        self.pvec_idxs = list()
        self.jacobian  = list()

        start = 0
        stop  = 0
        self.pvec_list_fwd = list()
        self.openmm_system_fwd_dict = dict()
        if three_point:
            self.openmm_system_bkw_dict = dict()
            self.pvec_list_bkw = list()
        self.openmm_system_dict = dict()
        self.parm_idx_sysname_dict = dict()
        parm_idx = 0
        for i in range(self.N_pvec):
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
            if three_point:
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
                if three_point:
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
                    if three_point:
                        sys_bkw = pvec_cp_bkw.parameter_manager.system_list[sys_idx]
                        assert sys_bkw.name == sys_fwd.name
                        bkw_dict[sys_bkw.name] = [sys_bkw.openmm_system]

                    if len(fwd_dict) == N_sys_per_batch:
                        self.openmm_system_fwd_dict[parm_idx].append(fwd_dict)
                        fwd_dict = dict()
                        if three_point:
                            self.openmm_system_bkw_dict[parm_idx].append(bkw_dict)
                            bkw_dict = dict()

                    if not three_point:
                        if sys_fwd.name in self.parm_idx_sysname_dict:
                            self.parm_idx_sysname_dict[sys_fwd.name].append(parm_idx)
                        else:
                            self.parm_idx_sysname_dict[sys_fwd.name] = [parm_idx]

                if len(fwd_dict) > 0:
                    self.openmm_system_fwd_dict[parm_idx].append(fwd_dict)
                    if three_point:
                        self.openmm_system_bkw_dict[parm_idx].append(bkw_dict)

                parm_idx += 1

        self.jacobian = np.array(self.jacobian)

    @property
    def pvec(self):

        pvec_list = list()
        for i in range(self.N_pvec):
            pvec_list.extend(
            self.pvec_list[i].copy()[:].tolist()
            )
        return np.array(pvec_list)

    def apply_changes(self, vec, grad=True, grad_diff=1.e-2):

        if vec.size != self.N_parms:
            raise ValueError(
                f"Length of vec is {vec.size} but must be {self.N_parms}"
                )

        for i in range(self.N_pvec):
            pvec       = self.pvec_list[i]
            start_stop = self.pvec_idxs[i]
            pvec[:]    = vec[start_stop[0]:start_stop[1]]
            pvec.apply_changes()
            if grad:
                for j in range(pvec.size):
                    pvec_fwd = self.pvec_list_fwd[i][j]
                    pvec_fwd[:] = vec[start_stop[0]:start_stop[1]]
                    pvec_fwd[j] += grad_diff
                    pvec_fwd.apply_changes()
                    if self._three_point:
                        pvec_bkw = self.pvec_list_bkw[i][j]
                        pvec_bkw[:] = vec[start_stop[0]:start_stop[1]]
                        pvec_bkw[j] -= grad_diff
                        pvec_bkw.apply_changes()

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
        vec):

        self.apply_changes(vec, grad=False)

        worker_id_list = [
            self.targetcomputer.__call__({key:value}, False) for key, value in self.openmm_system_dict.items()
            ]
        logP_likelihood = 0.
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(worker_id_list)
            _logP_likelihood = ray.get(worker_id[0])
            logP_likelihood += _logP_likelihood

        return logP_likelihood

    def grad(
        self, 
        vec, 
        parm_idx_list=None,
        grad_diff=1.e-2, 
        use_jac=True,
        ):

        if not self._three_point:
            grad_diff *= 2.

        self.apply_changes(vec, grad=True, grad_diff=grad_diff)

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        worker_id_dict = dict()
        for parm_idx in self.openmm_system_fwd_dict:
            if not parm_idx in parm_idx_list:
                continue
            for openmm_system_dict in self.openmm_system_fwd_dict[parm_idx]:
                worker_id = self.targetcomputer.__call__(openmm_system_dict, False)
                worker_id_dict[worker_id] = parm_idx, 1.
            if self._three_point:
                for openmm_system_dict in self.openmm_system_bkw_dict[parm_idx]:
                    worker_id = self.targetcomputer.__call__(openmm_system_dict, False)
                    worker_id_dict[worker_id] = parm_idx, -1.

        grad = np.zeros(self.N_parms, dtype=float)
        if not self._three_point:
            worker_id_dict_logL = {
                self.targetcomputer.__call__({key:value}, False) : key for key, value in self.openmm_system_dict.items()
                }

            worker_id_list_logL = list(worker_id_dict_logL.keys())
            while worker_id_list_logL:
                worker_id, worker_id_list_logL = ray.wait(worker_id_list_logL)
                worker_id = worker_id[0]
                _logP_likelihood = ray.get(worker_id)
                sysname = worker_id_dict_logL[worker_id]
                parm_idx_list = self.parm_idx_sysname_dict[sysname]
                for parm_idx in parm_idx_list:
                    grad[parm_idx] -= _logP_likelihood

        worker_id_list = list(worker_id_dict.keys())
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(worker_id_list)
            worker_id = worker_id[0]
            _logP_likelihood = ray.get(worker_id)
            parm_idx, sign = worker_id_dict[worker_id]
            grad[parm_idx] += _logP_likelihood * sign

        if use_jac:
            #grad *= 1./self.jacobian
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