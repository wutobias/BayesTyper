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

@ray.remote
def apply_grad(pvec, j, only_apply_to_systems, grad_diff):
    pvec     = pvec.copy(include_systems=True)
    pvec[j] += grad_diff
    pvec.apply_changes(
	only_apply_to_systems)
    system_dict = dict()
    for sys_idx in only_apply_to_systems:
        sys = pvec.parameter_manager.system_list[sys_idx]
        system_dict[sys.name,sys_idx] = sys.openmm_system
    del pvec
    return system_dict


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
        self.jacobian  = list()

        parm_idx_sysname_dict = dict()
        sysidx_parm_idx_dict  = dict()
        parm_idx_map_dict     = dict()

        parm_idx = 0
        for i in range(self._N_pvec):
            pvec = self.pvec_list[i]
            self.jacobian.extend(
                pvec.jacobian.tolist())
            for f in range(pvec.force_group_count):
                ranks = pvec.allocations.index([f])[0]
                for r in ranks:
                    valids = np.where(
                        pvec.parameter_manager.force_ranks == r)
                    system_idx_list = np.unique(
                        pvec.parameter_manager.system_idx_list[valids])
                    for p_idx in range(pvec.parameters_per_force_group):
                        _p_idx = f*pvec.parameters_per_force_group+p_idx
                        _parm_idx = parm_idx + p_idx
                        for sys_idx in system_idx_list:
                            if sys_idx in parm_idx_sysname_dict:
                                if _parm_idx not in parm_idx_sysname_dict[sys_idx]:
                                    parm_idx_sysname_dict[sys_idx].append(_parm_idx)
                            else:
                                parm_idx_sysname_dict[sys_idx] = [_parm_idx]
                            if _parm_idx in sysidx_parm_idx_dict:
                                if sys_idx not in sysidx_parm_idx_dict[_parm_idx]:
                                    sysidx_parm_idx_dict[_parm_idx].append(sys_idx)
                            else:
                                sysidx_parm_idx_dict[_parm_idx] = [sys_idx]
                        parm_idx_map_dict[_parm_idx] = (i,_p_idx)
                self.N_parms += pvec.parameters_per_force_group
                parm_idx     += pvec.parameters_per_force_group
        self.parm_idx_map_dict     = parm_idx_map_dict
        self.sysidx_parm_idx_dict  = dict()
        self.parm_idx_sysname_dict = dict()
        for parm_idx in sysidx_parm_idx_dict:
            self.sysidx_parm_idx_dict[parm_idx] = set(sysidx_parm_idx_dict[parm_idx])
        for sys_idx in parm_idx_sysname_dict:
            self.parm_idx_sysname_dict[sys_idx] = set(parm_idx_sysname_dict[sys_idx])
        self.jacobian = np.array(self.jacobian)

    @property
    def pvec(self):

        pvec_list = list()
        for i in range(self._N_pvec):
            pvec_list.extend(
                self.pvec_list[i][:].tolist()
            )
        return np.array(pvec_list)

    def apply_changes(
        self, 
        vec, 
        grad=False, 
        parm_idx_list=None,
        grad_diff=_EPSILON):

        if vec.size != self.N_parms:
            raise ValueError(
                f"Length of vec is {vec.size} but must be {self.N_parms}"
                )

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        only_apply_to_systems = set()
        for parm_idx in parm_idx_list:
            i,j  = self.parm_idx_map_dict[parm_idx]
            pvec = self.pvec_list[i]
            pvec[j] = vec[parm_idx]
            only_apply_to_systems.update(
                    self.sysidx_parm_idx_dict[parm_idx])
        for i in range(self._N_pvec):
            pvec = self.pvec_list[i]
            pvec.apply_changes(
                only_apply_to_systems=only_apply_to_systems)

        system_dict = dict()
        if not grad or (grad and not self._three_point):
            for i in range(self._N_pvec):
                pvec = self.pvec_list[i]
                for sys_idx in only_apply_to_systems:
                    sys = pvec.parameter_manager.system_list[sys_idx]
                    system_dict[sys.name,(sys_idx,1.,-1)] = sys.openmm_system

        if grad:
            pvec_list = [ray.put(pvec) for pvec in self.pvec_list]
            worker_id_dict = dict()
            for parm_idx in parm_idx_list:
                i,j  = self.parm_idx_map_dict[parm_idx]
                _only_apply_to_systems = self.sysidx_parm_idx_dict[parm_idx]
                worker_id = apply_grad.remote(
                        pvec_list[i], j, _only_apply_to_systems, grad_diff)
                worker_id_dict[worker_id] = 1., parm_idx 
                if self._three_point:
                    worker_id = apply_grad.remote(
                            pvec_list[i], j, _only_apply_to_systems, -grad_diff)
                    worker_id_dict[worker_id] = -1., parm_idx

            worker_id_list = list(worker_id_dict.keys())
            while worker_id_list:
                [worker_id], worker_id_list = ray.wait(worker_id_list)
                _system_dict = ray.get(worker_id)
                sign, parm_idx = worker_id_dict[worker_id]
                for _key in _system_dict:
                    sys_name, sys_idx = _key
                    key = sys_name, (sys_idx, sign, parm_idx)
                    system_dict[key] = _system_dict[_key]

                #pvec = self.pvec_list[i]
                #pvec[j] += grad_diff
                #pvec.apply_changes(
                #    only_apply_to_systems=_only_apply_to_systems)
                #for sys_idx in _only_apply_to_systems:
                #    sys = pvec.parameter_manager.system_list[sys_idx]
                #    system_dict[sys.name,(sys_idx,1.,parm_idx)] = sys.openmm_system
                #if self._three_point:
                #    pvec[j] -= 2.*grad_diff
                #    pvec.apply_changes(
                #        only_apply_to_systems=_only_apply_to_systems)
                #    for sys_idx in _only_apply_to_systems:
                #        sys = pvec.parameter_manager.system_list[sys_idx]
                #        system_dict[sys.name,(sys_idx,-1.,parm_idx)] = sys.openmm_system
                #    pvec[j] += grad_diff
                #else:
                #    pvec[j] -= grad_diff
                #pvec.apply_changes(
                #        only_apply_to_systems=_only_apply_to_systems)

        return system_dict


    def _results_dict_to_likelihood(self, results_dict_all):

        logP_likelihood = 0.
        for sys_name in results_dict_all:
            for results_dict in results_dict_all[sys_name]:
                for target_idx in results_dict["rss"]:
                    rss = results_dict["rss"][target_idx]
                    log_norm_factor = results_dict["log_norm_factor"][target_idx]
                    logP_likelihood += -log_norm_factor - 0.5 * np.log(2.*np.pi) - 0.5 * rss

        return logP_likelihood

    def _call_local(self, vec, parm_idx_list):

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        system_dict = self.apply_changes(
            vec, grad=False, 
            parm_idx_list=parm_idx_list) 
        logP_likelihood, _ = self.targetcomputer(system_dict, False, True)

        return logP_likelihood

    def __call__(
        self,
        vec,
        parm_idx_list=None,
        local=False):

        if local:
            return self._call_local(vec, parm_idx_list)

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        system_dict = self.apply_changes(
            vec, grad=False, 
            parm_idx_list=parm_idx_list)
        worker_id_dict = dict()
        system_batch_dict = dict()
        for key in system_dict:
            system_batch_dict[key] = system_dict[key]
            if len(system_batch_dict) == self._N_sys_per_batch:
                worker_id = self.targetcomputer(system_batch_dict, False, False)
                worker_id_dict[worker_id] = list(system_batch_dict.keys())
                system_batch_dict = dict()
        if len(system_batch_dict) > 0:
            worker_id = self.targetcomputer(system_batch_dict, False, False)
            worker_id_dict[worker_id] = list(system_batch_dict.keys())
            system_batch_dict = dict()

        worker_id_list  = list(worker_id_dict.keys())
        logP_likelihood = 0.
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(
                worker_id_list, timeout=_TIMEOUT)
            failed = len(worker_id) == 0
            if not failed:
                try:
                    _logP_likelihood, _ = ray.get(
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
                    key_list = worker_id_dict[worker_id]
                    del worker_id_dict[worker_id]
                    worker_id = self.targetcomputer(
                        {key:system_dict[key] for key in key_list}, False, False)
                    worker_id_dict[worker_id] = key_list
                worker_id_list = list(worker_id_dict.keys())

        return logP_likelihood

    
    def _grad_local(
        self, 
        vec, 
        parm_idx_list=None,
        grad_diff=_EPSILON, 
        use_jac=False,
        ):

        import numpy as np

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        system_dict = self.apply_changes(
            vec, grad=True, 
            parm_idx_list=parm_idx_list, 
            grad_diff=grad_diff)
        _, results_dict = self.targetcomputer(system_dict, False, True)

        grad = np.zeros(self.N_parms, dtype=float)
        for key in results_dict:
            sys_name, (sys_idx, sign, parm_idx) = key
            if self._three_point:
                grad[parm_idx] += results_dict[key] * sign
            else:
                if parm_idx == -1:
                    _parm_idx = self.parm_idx_sysname_dict[sys_idx]
                    grad[list(_parm_idx)] -= results_dict[key]
                else:
                    grad[parm_idx] += results_dict[key]

        if use_jac:
            grad -= np.log(self.jacobian)

        if self._three_point:
            grad *= 1./(grad_diff * 2.)
        else:
            grad *= 1./grad_diff

        return grad

    def grad(
        self, 
        vec, 
        parm_idx_list=None,
        grad_diff=_EPSILON, 
        use_jac=False,
        local=False
        ):

        if local:
            return self._grad_local(vec, parm_idx_list, grad_diff, use_jac)

        import numpy as np

        if isinstance(parm_idx_list, type(None)):
            parm_idx_list = list(range(self.N_parms))

        system_dict = self.apply_changes(
            vec, grad=True, 
            parm_idx_list=parm_idx_list, 
            grad_diff=grad_diff)
        worker_id_dict = dict()
        system_batch_dict = dict()
        for key in system_dict:
            #sys_name, sign, parm_idx = key
            system_batch_dict[key] = system_dict[key]
            if len(system_batch_dict) == self._N_sys_per_batch:
                worker_id = self.targetcomputer(system_batch_dict, False, False)
                worker_id_dict[worker_id] = list(system_batch_dict.keys())
                system_batch_dict = dict()
        if len(system_batch_dict) > 0:
            worker_id = self.targetcomputer(system_batch_dict, False, False)
            worker_id_dict[worker_id] = list(system_batch_dict.keys())
            system_batch_dict = dict()

        worker_id_list  = list(worker_id_dict.keys())
        grad = np.zeros(self.N_parms, dtype=float)
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(
                worker_id_list, timeout=_TIMEOUT)
            failed = len(worker_id) == 0
            if not failed:
                try:
                    _logP_likelihood, results_dict = ray.get(
                        worker_id[0], timeout=_TIMEOUT)
                except:
                    failed = True
                if not failed:
                    for key in results_dict:
                        sys_name, (sys_idx, sign, parm_idx) = key
                        if self._three_point:
                            grad[parm_idx] += results_dict[key] * sign
                        else:
                            if parm_idx == -1:
                                _parm_idx = self.parm_idx_sysname_dict[sys_idx]
                                grad[list(_parm_idx)] -= results_dict[key]
                            else:
                                grad[parm_idx] += results_dict[key]
                    del worker_id_dict[worker_id[0]]
            if failed:
                if len(worker_id) > 0:
                    if worker_id[0] not in worker_id_list:
                        worker_id_list.append(worker_id[0])
                resubmit_list = retrieve_failed_workers(worker_id_list)
                for worker_id in resubmit_list:
                    ray.cancel(worker_id, force=True)
                    key_list = worker_id_dict[worker_id]
                    del worker_id_dict[worker_id]
                    worker_id = self.targetcomputer(
                        {key:system_dict[key] for key in key_list}, False, False)
                    worker_id_dict[worker_id] = key_list
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
