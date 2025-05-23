import numpy as np
import copy

from .vectors import ForceFieldParameterVector
from .likelihoods import LikelihoodVectorized
from .bitvector_typing import BitSmartsManager

from .constants import (_LENGTH,
                        _ANGLE,
                        _ANGLE_DEG,
                        _ATOMIC_MASS,
                        _ENERGY_PER_MOL,
                        _WAVENUMBER,
                        _UNIT_QUANTITY,
                        _FORCE_CONSTANT_BOND,
                        _FORCE_CONSTANT_ANGLE,
                        _FORCE_CONSTANT_TORSION,
                        _FORCE,
                        _INACTIVE_GROUP_IDX,
                        _TIMEOUT,
                        _VERBOSE,
                        _EPSILON,
                        _EPSILON_GS,
                        _USE_GLOBAL_OPT,
                        _GLOBAL_OPT_METHOD,
                        _OPT_TIMEOUT,
                        _MAX_ON
                        )
from .ray_tools import retrieve_failed_workers

import ray

import gc


@ray.remote
def generate_parameter_manager(sys_list, parm_mngr):

    import copy
    parm_mngr_cp = copy.deepcopy(parm_mngr)
    for sys in sys_list:
        parm_mngr_cp.add_system(sys)
    return parm_mngr_cp


@ray.remote
def _test_logL(
    pvec,
    targetcomputer,
    N=25,
    check_local=False):

    import time

    time_dict = dict()
    pvec_cp = pvec.copy(include_systems=True)
    for N_sys_per_batch in range(len(pvec.parameter_manager.system_list)):
        logL = LikelihoodVectorized(
                [pvec_cp], 
                targetcomputer, 
                N_sys_per_batch=N_sys_per_batch)
        x0   = logL.pvec[:]
        time_dict[N_sys_per_batch, False] = 0.
        for _ in range(N):
            start_time = time.time()
            logL(x0, local=False)
            time_dict[N_sys_per_batch, False] += time.time() - start_time
        time_dict[N_sys_per_batch, False] /= N
    logL = LikelihoodVectorized(
            [pvec_cp],
            targetcomputer,
            N_sys_per_batch=99999)
    x0   = logL.pvec[:]
    if check_local:
        time_dict[99999, True] = 0.
        for _ in range(N):
            start_time = time.time()
            logL(x0, local=True)
            time_dict[99999, True] += time.time() - start_time
        time_dict[99999, True] /= N

    return time_dict


@ray.remote
def likelihood_combine_pvec(
    ref_pvec,
    pvec_list_query,
    bitvec_type_list_query,
    parm_penalty,
    bitvec_list_alloc_dict_list,
    targetcomputer,
    verbose=False):

    N_queries    = len(pvec_list_query)
    N_mngr       = len(ref_pvec)
    N_types_init = sum([pvec.force_group_count for pvec in ref_pvec])
    ref_pvec_cp  = [pvec.copy(include_systems=True) for pvec in ref_pvec]
    system_list  = ref_pvec_cp[0].parameter_manager.system_list
    for mngr_idx in range(N_mngr):
        ### IMPORANT: We must rebuild this with list of systems
        ###           that is linked to all parameter managers.
        ref_pvec_cp[mngr_idx].rebuild_from_systems(
            lazy=True, 
            system_list=system_list)
        ref_pvec_cp[mngr_idx].apply_changes()

    logL = LikelihoodVectorized(
            ref_pvec_cp, targetcomputer, three_point=False, N_sys_per_batch=8)

    from .bitvector_typing import bitvec_hierarchy_to_allocations
    import itertools

    results_dict = dict()
    for selection in itertools.product(range(N_queries), repeat=N_mngr):
        failed = False
        N_parms_new = 0
        N_types_new = 0
        for mngr_idx in range(N_mngr):
            candidate_idx = selection[mngr_idx]
            N_types_new  += len(bitvec_type_list_query[candidate_idx][mngr_idx])
        if not (N_types_new > N_types_init):
            continue
        for mngr_idx in range(N_mngr):
            candidate_idx = selection[mngr_idx]
            ### Must always start from pvec_ref_cp[mngr_idx] since this
            ### is the only one that is related to the actual systems
            ### that we want to test
            allocations = [-1 for _ in ref_pvec_cp[mngr_idx].allocations]
            bitvec_hierarchy_to_allocations(
                bitvec_list_alloc_dict_list[mngr_idx],
                bitvec_type_list_query[candidate_idx][mngr_idx],
                allocations)
            if allocations.count(-1) == 0:
                max_alloc  = len(pvec_list_query[candidate_idx][mngr_idx])
                max_alloc  = int(max_alloc/ref_pvec_cp[mngr_idx].parameters_per_force_group)
                max_alloc -= 1
                if max(allocations) > max_alloc:
                    failed = True
                else:
                    ref_pvec_cp[mngr_idx].allocations[:] = allocations[:]
                    ref_pvec_cp[mngr_idx].reset(
                            pvec_list_query[candidate_idx][mngr_idx],
                            ref_pvec_cp[mngr_idx].allocations)
            else:
                failed = True
            N_parms_new += ref_pvec_cp[mngr_idx].size
            if failed:
                if verbose:
                    print(
                        f"Selection {selection} failed.")
                break
        if not failed:
            logL._initialize_systems()
            x0         = logL.pvec[:]
            likelihood = logL(x0)
            aic        = 2. * N_parms_new * parm_penalty - 2. * likelihood
            results_dict[selection] = aic

    return results_dict


def batch_likelihood_typing(
    pvec,
    targetcomputer,
    typing_list,
    ):

    import copy
    import numpy as np

    N_queries = len(typing_list)

    init_alloc = pvec.allocations[:]
    worker_id_dict = dict()
    cache_dict = dict()
    cached_idxs = list()
    pvec_list = [pvec]
    logL = LikelihoodVectorized(
            pvec_list, targetcomputer, three_point=False, N_sys_per_batch=4)
    x0   = logL.pvec[:]
    logL_list = np.zeros(N_queries, dtype=float)
    for idx in range(N_queries):
        typing = list(typing_list[idx])
        typing_tuple = tuple(typing)
        if typing_tuple in cache_dict:
            cached_idxs.append(idx)
        else:
            cache_dict[typing_tuple] = idx
            pvec_list[0].allocations[:] = typing[:]
            pvec_list[0].apply_changes()
            logL._initialize_systems()
            logL_list[idx] = logL(x0)
    
    for idx in cached_idxs:
        typing = list(typing_list[idx])
        typing_tuple = tuple(typing)
        idx_cache = cache_dict[typing_tuple]
        logL_list[idx] = logL_list[idx_cache]

    return logL_list


def ungroup_forces(parameter_manager, perturbation=None):

    __doc__ = """
    This method takes a ParameterManager object and puts each force rank
    into its own force group. Caution: The ParameterManager object will
    be altered *in place*.

    If perturbation is a float, new value will perturbed from a zero-mean
    Gaussian with width `perturbation`.
    """

    import numpy as np

    ### Put every rank into their own force group
    ### The inital force ranks will remain, but will be empty
    ### until we remove them in the subsequent for loop (see below).
    N_forces_ranks  = np.max(parameter_manager.force_ranks) + 1
    N_force_objects = parameter_manager.force_group_count
    for force_rank in range(N_forces_ranks):
        force_object_idxs = np.where(parameter_manager.force_ranks == force_rank)[0]
        force_group_idx   = parameter_manager.force_group_idx_list[force_object_idxs[0]]
        input_parms_dict  = dict()
        for name in parameter_manager.parm_names:
            value = parameter_manager.get_parameter(force_group_idx, name)
            if not isinstance(perturbation, type(None)):
                if isinstance(value, _UNIT_QUANTITY):
                    value += abs(np.random.normal(0., perturbation)) * value.unit
                else:
                    value += abs(np.random.normal(0., perturbation))
            input_parms_dict[name] = value
        parameter_manager.add_force_group(input_parms_dict)
        parameter_manager.relocate_by_rank(
            force_rank,
            parameter_manager.force_group_count-1
            )
    
    ### Remove all empty force groups.
    ### Note that after removal of force group i, all force groups
    ### with j>i will be re-assigned to j-1
    for _ in range(N_force_objects):
        parameter_manager.remove_force_group(0)


@ray.remote(num_cpus=1)
def get_gradient_scores(
    ff_parameter_vector,
    targetcomputer,
    type_i, # type to be split
    selection_i = None,
    k_values_ij = None,
    grad_diff = _EPSILON_GS,
    N_trials = 5,
    N_sys_per_likelihood_batch = 4,
    local_targets=False,
    ):

    ff_parameter_vector_cp = ff_parameter_vector.copy(
        include_systems = True)

    N_types    = ff_parameter_vector_cp.force_group_count

    N_parms    = ff_parameter_vector_cp.parameters_per_force_group
    N_typables = ff_parameter_vector_cp.allocations.size

    type_j = type_i + 1

    ff_parameter_vector_cp.duplicate(type_i, type_j)

    first_parm_i = type_i * N_parms
    last_parm_i  = first_parm_i + N_parms

    first_parm_j = type_j * N_parms
    last_parm_j  = first_parm_j + N_parms

    parm_idx_list = list()
    parm_idx_list.extend(range(first_parm_i, last_parm_i))
    parm_idx_list.extend(range(first_parm_j, last_parm_j))

    N_comb = k_values_ij.shape[0]

    likelihood_func = LikelihoodVectorized(
        [ff_parameter_vector_cp],
        targetcomputer,
        N_sys_per_batch = N_sys_per_likelihood_batch
        )

    initial_vec = copy.deepcopy(ff_parameter_vector_cp[first_parm_i:last_parm_i])

    grad_score_dict      = {i:list() for i in range(N_trials)}
    grad_norm_dict       = {i:list() for i in range(N_trials)}
    allocation_list_dict = list()
    selection_list_dict  = list()
    type_list_dict       = list()

    for comb_idx in range(N_comb):

        ff_parameter_vector_cp.allocations[:] = k_values_ij[comb_idx]
        ff_parameter_vector_cp.apply_changes()

        likelihood_func._initialize_systems()

        allocation_list_dict.append(tuple(k_values_ij[comb_idx].tolist()))
        selection_list_dict.append(tuple(selection_i))
        type_list_dict.append(tuple([type_i, type_j]))

        for trial_idx in range(N_trials):

            ff_parameter_vector_cp[first_parm_i:last_parm_i]  = initial_vec[:]
            ff_parameter_vector_cp[first_parm_i:last_parm_i] += np.random.normal(0, 0.01, initial_vec.size)
            ff_parameter_vector_cp[first_parm_j:last_parm_j]  = ff_parameter_vector_cp[first_parm_i:last_parm_i]

            grad = likelihood_func.grad(
                ff_parameter_vector_cp[:],
                parm_idx_list=parm_idx_list,
                grad_diff=grad_diff,
                use_jac=False,
                local=local_targets,
                )

            grad_i = grad[first_parm_i:last_parm_i]
            grad_j = grad[first_parm_j:last_parm_j]

            norm_i = np.linalg.norm(grad_i)
            norm_j = np.linalg.norm(grad_j)

            if np.isnan(norm_i) or np.isinf(norm_i):
                continue
            if np.isnan(norm_j) or np.isinf(norm_j):
                continue

            if norm_i > 0.:
                grad_i /= norm_i
            if norm_j > 0.:
                grad_j /= norm_j

            grad_ij_dot = np.dot(
                grad_i,
                grad_j,
                )
            grad_ij_diff = np.linalg.norm(
                grad_i - grad_j,
                )

            if N_parms == 1:
                grad_score_dict[trial_idx].append(grad_ij_diff)
            else:
                grad_score_dict[trial_idx].append(grad_ij_dot)
            grad_norm_dict[trial_idx].append([norm_i, norm_j])

    return grad_score_dict, grad_norm_dict, allocation_list_dict, selection_list_dict, type_list_dict


@ray.remote(num_cpus=1, max_retries=0)
def minimize_FF(
    system_list,
    targetcomputer,
    pvec_list,
    frozen_types_list,
    bitvec_type_list,
    bounds_list,
    parm_penalty,
    grad_diff = _EPSILON,
    local_targets = False,
    N_sys_per_likelihood_batch = 4,
    bounds_penalty = 10.,
    use_global_opt = _USE_GLOBAL_OPT,
    verbose = False,
    TIMEOUT = _OPT_TIMEOUT, #timeout for optimizier in seconds, default is 1hr
    get_timing = False):

    verbose = False

    from openmm import unit
    from . import priors
    import numpy as np
    import copy
    import time

    pvec_list_cp   = copy.deepcopy(pvec_list)
    system_list_cp = copy.deepcopy(system_list)

    pvec_min_list = list()
    N_parms_all   = 0
    pvec_idx_list = range(len(pvec_list_cp))
    for pvec in pvec_list_cp:
        N_parms_all += pvec.size

    if isinstance(bounds_penalty, float):
        _bounds_penalty = [bounds_penalty for _ in pvec_list_cp]
        bounds_penalty = _bounds_penalty

    prior_idx_list_all  = list()
    prior_idx_list_notk = list()
    prior_idx_list_k    = list()
    parm_idx_list_all   = list()
    parm_idx_list_notk  = list()
    parm_idx_list_k     = list()
    bounds_penalty_list = list()
    N_parms = 0

    for pvec_idx in pvec_idx_list:

        pvec = pvec_list_cp[pvec_idx]
        pvec_min_list.append(pvec)
        ### Look for dead parameters
        hist = pvec.force_group_histogram
        for type_i in range(pvec.force_group_count):
            b = bitvec_type_list[pvec_idx][type_i]
            for idx in range(pvec.parameters_per_force_group):
                freeze_parm = False
                ### We want to check if b is a child of a frozen type.
                ### Including the b itself.
                for _b in frozen_types_list[pvec_idx]:
                    ### b is subset of _b
                    if _b == (b & _b):
                        if idx in frozen_types_list[pvec_idx][_b]:
                            freeze_parm = True
                            break
                if freeze_parm:
                    if verbose:
                        ### Actual idx
                        _idx = type_i * pvec.parameters_per_force_group + idx
                        print(
                            f"Freezing MNGR {pvec_idx} parameter {_idx} {pvec.vector_k[_idx]}")
                    continue
                if hist[type_i] == 0:
                    if verbose:
                        ### Actual idx
                        _idx = type_i * pvec.parameters_per_force_group + idx
                        print(
                            f"Excluding MNGR {pvec_idx} parameter {_idx} {pvec.vector_k[_idx]}")
                    continue
                parm_idx = type_i * pvec.parameters_per_force_group + idx + N_parms
                name     = pvec.parameter_name_list[idx]
                if name == 'k':
                    parm_idx_list_k.append(parm_idx)
                if name != 'k':
                    parm_idx_list_notk.append(parm_idx)
                parm_idx_list_all.append(parm_idx)

        ### `lazy=True` just replaces the systems stored in
        ### in the parameter manager with the ones in `system_list_cp`.
        ### It does not completely rebuild the parameter_manager, but
        ### performs some basic sanity checks.
        pvec.rebuild_from_systems(
            lazy = True,
            system_list = system_list_cp
            )
        pvec.apply_changes()

        bounds = bounds_list[pvec_idx]
        if isinstance(bounds, priors.BaseBounds):
            bounds.apply_pvec(pvec)
            hist = pvec.force_group_histogram
            for type_i in range(pvec.force_group_count):
                for idx in range(pvec.parameters_per_force_group):
                    parm_idx = type_i * pvec.parameters_per_force_group + idx + N_parms
                    name     = pvec.parameter_name_list[idx]
                    bounds_penalty_list.append(bounds_penalty[pvec_idx])
                    b = bitvec_type_list[pvec_idx][type_i]
                    freeze_parm = False
                    ### We want to check if b_old and any child-type
                    ### of b_old must be frozen
                    for _b in frozen_types_list[pvec_idx]:
                        ### b is subset of _b
                        if _b == (b & _b):
                            ### Note only _b is stored in frozen_types_list,
                            if idx in frozen_types_list[pvec_idx][_b]:
                                freeze_parm = True
                                break
                    if freeze_parm:
                        continue
                    if hist[type_i] == 0:
                        continue
                    if name == 'k':
                        prior_idx_list_k.append(parm_idx)
                    if name != 'k':
                        prior_idx_list_notk.append(parm_idx)
                    prior_idx_list_all.append(parm_idx)

        N_parms += pvec.size

    prior_idx_list_all  = np.array(prior_idx_list_all, dtype=int)
    prior_idx_list_notk = np.array(prior_idx_list_notk, dtype=int)
    prior_idx_list_k    = np.array(prior_idx_list_k, dtype=int)
    parm_idx_list_all   = np.array(parm_idx_list_all, dtype=int)
    parm_idx_list_notk  = np.array(parm_idx_list_notk, dtype=int)
    parm_idx_list_k     = np.array(parm_idx_list_k, dtype=int)

    bounds_penalty_list  = np.array(bounds_penalty_list, dtype=float)

    time_start = time.time()

    def penalty(x):

        penalty_val = 0.
        if prior_idx_list.size > 0:
            _x = x0_ref.copy()
            _x[parm_idx_list] = x.copy()
            penalty_val += np.sum(bounds_penalty_list[prior_idx_list] * _x[prior_idx_list]**2)
        return penalty_val

    def fun(x):

        _x = x0_ref.copy()
        _x[parm_idx_list] = x.copy()

        likelihood = likelihood_func(_x, 
            parm_idx_list=parm_idx_list, local=local_targets)
        AIC_score  = 2. * N_parms_all * parm_penalty - 2. * likelihood
        
        return AIC_score

    def grad_penalty(x):

        grad = np.zeros_like(x0_ref)
        if prior_idx_list.size > 0:
            _x = x0_ref.copy()
            _x[parm_idx_list] = x.copy()
            grad[prior_idx_list] += bounds_penalty_list[prior_idx_list] * 2. * _x[prior_idx_list]

        return grad[parm_idx_list]

    def grad(x):

        _x = x0_ref.copy()
        _x[parm_idx_list] = x.copy()

        _grad = likelihood_func.grad(_x, 
            grad_diff=grad_diff, parm_idx_list=parm_idx_list,
            use_jac=False, local=local_targets)
        ### Multiply by -2. due to AIC definition
        _grad *= -2.

        return _grad[parm_idx_list]

    def stop_after_timeout(xk, convergence):
        elapsed = time.time() - time_start
        if elapsed > TIMEOUT:
            print(f"Stopping Optimization after {elapsed:.2f} seconds")
            return True
        return False

    likelihood_func = LikelihoodVectorized(
        pvec_min_list,
        targetcomputer,
        N_sys_per_batch = N_sys_per_likelihood_batch)

    search_space_list = list()
    for idx, pvec in enumerate(pvec_min_list):
        bounds = bounds_list[idx]
        if isinstance(bounds, priors.BaseBounds):
            lower_list, upper_list = bounds.get_bounds(pvec)
            _l, _u = list(), list()
            for i in range(pvec.force_group_count):
                for j in range(pvec.parameters_per_force_group):
                    _l.append(lower_list[i][j])
                    _u.append(upper_list[i][j])
            lower_trans = pvec.get_transform(_l * pvec.vector_units).tolist()
            upper_trans = pvec.get_transform(_u * pvec.vector_units).tolist()
            for _lu in zip(lower_trans, upper_trans):
                _lu = list(_lu)
                if abs(_lu[0]-_lu[1]) < 1.e-2:
                    _lu[0] -= 4
                    _lu[1] += 4
                search_space_list.append(_lu)
        else:
            N = pvec.force_group_count * pvec.parameters_per_force_group
            for _ in range(N):
                search_space_list.append([-4,4])
    search_space_list = np.array(search_space_list)

    if verbose:
        print("SEARCH SPACE SUMMARY")
        print("====================")
        start = 0
        stop  = 0
        for mngr_idx, pvec in enumerate(pvec_list_cp):
            start = stop
            stop  = start + len(pvec.vector_k)
            lower    = pvec.get_real([v[0] for v in search_space_list[start:stop]])
            upper    = pvec.get_real([v[1] for v in search_space_list[start:stop]])
            lower_str = [f"{str(v):10s}" for v in lower]
            upper_str = [f"{str(v):10s}" for v in upper]
            print(f"MANAGER {mngr_idx} LOWER LIMIT")
            print(f"===============================")
            print(",".join(lower_str))
            print(f"MANAGER {mngr_idx} UPPER LIMIT")
            print(f"===============================")
            print(",".join(upper_str))

    from scipy import optimize
    for protocol in ["all"]:
        if protocol == "k":
            parm_idx_list  = parm_idx_list_k
            prior_idx_list = prior_idx_list_k
        elif protocol == "notk":
            parm_idx_list  = parm_idx_list_notk
            prior_idx_list = prior_idx_list_notk
        elif protocol == "all":
            parm_idx_list  = parm_idx_list_all
            prior_idx_list = prior_idx_list_all
        elif protocol == "notk_prior":
            parm_idx_list  = parm_idx_list_all
            prior_idx_list = prior_idx_list_notk
        else:
            raise ValueError(
                f"protocol `{protocol}` not known.")

        likelihood_func._initialize_systems()

        from .constants import _OPT_METHOD
        METHOD = _GLOBAL_OPT_METHOD
        if use_global_opt:
            _fun  = lambda x: fun(x)
            _grad = lambda x: grad(x)

            x0 = copy.deepcopy(likelihood_func.pvec[:])
            x0_ref = copy.deepcopy(likelihood_func.pvec[:])

            ### Make sure that x0 is within bounds of the
            ### search space.
            for idx in range(len(x0)):
                if x0[idx] < search_space_list[idx][0]:
                    x0[idx] = search_space_list[idx][0] + 0.01
                if x0[idx] > search_space_list[idx][1]:
                    x0[idx] = search_space_list[idx][1] - 0.01

            if METHOD == "differential_evolution":
                result = optimize.differential_evolution(
                   _fun, search_space_list[parm_idx_list].tolist(),
                   polish = False, x0 = x0[parm_idx_list],
                   callback = stop_after_timeout,
                   maxiter = 1000,)
            elif METHOD == "basinhopping":
                result = optimize.basinhopping(
                        _fun, x0[parm_idx_list],
                        minimizer_kwargs={
                            "method"  : _OPT_METHOD,
                            "jac"     : _grad,
                            "options" : {
                                "gtol"    : 1e-2}
                            })
            elif METHOD == "direct":
                result = optimize.direct(
                        _fun, search_space_list[parm_idx_list].tolist(), 
                        maxiter=1000)
            else:
                raise NotImplementedError(
                    f"method {METHOD} not implemented.")

            x0 = x0_ref.copy()
            x0[parm_idx_list] = result.x
            likelihood_func.apply_changes(x0)
            result = optimize.minimize(
                    _fun, x0[parm_idx_list], jac = _grad,
                    method = _OPT_METHOD, 
                    bounds = search_space_list[parm_idx_list].tolist())
        
        else:
            likelihood_func._initialize_systems()
            x0 = copy.deepcopy(likelihood_func.pvec[:])[parm_idx_list]
            x0_ref = copy.deepcopy(likelihood_func.pvec[:])
            _fun  = lambda x: fun(x)  + penalty(x)
            _grad = lambda x: grad(x) + grad_penalty(x)
            result = optimize.minimize(
                _fun, x0, jac = _grad, 
                method = _OPT_METHOD)

        x_best = x0_ref.copy()
        x_best[parm_idx_list] = result.x
        likelihood_func.apply_changes(x_best)
    
    #for mngr_idx, pvec in enumerate(pvec_list_cp):
    #    vec_str = [str(v) for v in pvec.vector_k]
    #    print(f"MANAGER {mngr_idx}:")
    #    print(f"==========")
    #    print(pvec.force_group_histogram, ",".join(vec_str))
    
    parm_idx_list = parm_idx_list_all
    prior_idx_list = prior_idx_list_all
    x0 = copy.deepcopy(likelihood_func.pvec[:])[parm_idx_list]
    best_f  = fun(x0)

    _pvec_list_cp = [pvec.vector_k[:].copy() for pvec in pvec_list_cp]
    if get_timing:
        return best_f, _pvec_list_cp, bitvec_type_list, time.time() - time_start
    else:
        return best_f, _pvec_list_cp, bitvec_type_list


@ray.remote(num_cpus=1)
def validate_FF(
    mngr_idx_main,
    pvec_list,
    targetcomputer,
    bitvec_dict, # These are the bitvectors we are querying
    bitvec_alloc_dict_list, # These are the bitvectors for the substructures and their corresponding allocations
    bitvec_type_list_list, # These are the existing bitvector hierarchies
    worker_id_dict = dict(),
    parm_penalty=1.,
    verbose=False,
    ):

    import copy
    import numpy as np
    import ray
    from .bitvector_typing import bitvec_hierarchy_to_allocations

    N_mngr = len(pvec_list)

    pvec_list_cp = [pvec.copy(include_systems=True) for pvec in pvec_list]
    pvec_list_initial = [pvec.copy() for pvec in pvec_list]
    bitvec_type_list_list_cp = copy.deepcopy(bitvec_type_list_list)
    bitvec_type_list_list_initial = copy.deepcopy(bitvec_type_list_list)

    system_list = pvec_list_cp[0].parameter_manager.system_list
    for mngr_idx in range(N_mngr):
        ### IMPORANT: We must rebuild this with list of systems
        ###           that is common to all parameter managers.
        pvec_list_cp[mngr_idx].rebuild_from_systems(
            lazy=True, 
            system_list=system_list
            )
        pvec_list_cp[mngr_idx].apply_changes()

    def _calculate_AIC(
        _pvec_list,
        _mngr_idx,
        _typing_list,
        targetcomputer):

        logL_list = batch_likelihood_typing(
            _pvec_list[_mngr_idx],
            targetcomputer,
            _typing_list,
            )
        N_parms_all = 0.
        for pvec in _pvec_list:
            N_parms_all += pvec.size
        _AIC_list = list()
        for L in logL_list:
            AIC  = 2. * N_parms_all * parm_penalty
            AIC -= 2. * L
            _AIC_list.append(AIC)

        return _AIC_list

    best_pvec_list = [pvec.copy() for pvec in pvec_list_cp]
    best_ast       = None
    best_AIC       = None
    best_bitvec_type_list_list = copy.deepcopy(bitvec_type_list_list_cp)

    found_improvement = False

    allocation_failure_counts = 0
    allocation_all_counts     = 0
    minimization_failure_counts = 0
    minimization_all_counts     = 0
    ### For each system, find the best solution
    worker_id_list = list(worker_id_dict.keys())
    while worker_id_list:

        minimization_all_counts += 1

        [worker_id], worker_id_list = ray.wait(worker_id_list)
        try:
            _, _pvec_list, bitvec_type_all_list = ray.get(worker_id)
            failed = False
        except:
            if _VERBOSE:
                import traceback
                print(traceback.format_exc())
            failed = True
        if failed:
            minimization_failure_counts += 1
            continue

        allocation_all_counts += 1

        ast = worker_id_dict[worker_id]
        _, _, type_ = ast
        type_i, type_j = type_

        allocation_failure = False
        for mngr_idx in range(N_mngr):
            allocations = [0 for _ in pvec_list_cp[mngr_idx].allocations]
            bitvec_hierarchy_to_allocations(
                bitvec_alloc_dict_list[mngr_idx], 
                bitvec_type_all_list[mngr_idx],
                allocations
                )
            if allocations.count(-1) > 0:
                allocation_failure = True
            pvec_list_cp[mngr_idx].allocations[:] = allocations
            pvec_list_cp[mngr_idx].reset(
                _pvec_list[mngr_idx],
                pvec_list_cp[mngr_idx].allocations)

        if allocation_failure:
            for mngr_idx in range(N_mngr):
                pvec_list_cp[mngr_idx].reset(
                    pvec_list_initial[mngr_idx])
            allocation_failure_counts += 1
            continue

        bitvec_list = list()
        alloc_list  = list()
        for b in bitvec_dict[ast]:
            bitvec_type_list_list_cp[mngr_idx_main].insert(
                type_j, b
                )

            allocations = [-1 for _ in pvec_list_cp[mngr_idx_main].allocations]
            bitvec_hierarchy_to_allocations(
                bitvec_alloc_dict_list[mngr_idx_main], 
                bitvec_type_list_list_cp[mngr_idx_main],
                allocations
                )
            N_types = pvec_list_cp[mngr_idx_main].force_group_count
            if (allocations.count(-1) == 0) and (max(allocations) < N_types):
                allocs = tuple(allocations)
                if allocs not in alloc_list:
                    alloc_list.append(
                        allocs
                        )
                    bitvec_list.append(b)
            bitvec_type_list_list_cp[mngr_idx_main].pop(type_j)
        if len(alloc_list) == 0:
            for mngr_idx in range(N_mngr):
                bitvec_type_list_list_cp[mngr_idx] = copy.deepcopy(
                        bitvec_type_list_list_initial[mngr_idx])
            allocation_failure_counts += 1
            continue

        AIC_list = _calculate_AIC(
            pvec_list_cp,
            mngr_idx_main,
            alloc_list,
            targetcomputer
            )
        for idx, b in enumerate(bitvec_list):
            new_AIC = AIC_list[idx]
            if best_ast == None:
                accept = True
            else:
                accept  = new_AIC < best_AIC
            if accept:
                pvec_list_cp[mngr_idx_main].allocations[:] = alloc_list[idx]
                pvec_list_cp[mngr_idx_main].apply_changes()
                best_AIC       = new_AIC
                best_pvec_list = [pvec.copy() for pvec in pvec_list_cp]
                best_ast       = ast

                best_bitvec_type_list_list = copy.deepcopy(
                        bitvec_type_list_list_cp)
                best_bitvec_type_list_list[mngr_idx_main].insert(
                    type_j, b)

                found_improvement = True

        for mngr_idx in range(N_mngr):
            pvec_list_cp[mngr_idx].reset(
                pvec_list_initial[mngr_idx])
            bitvec_type_list_list_cp[mngr_idx] = copy.deepcopy(
                bitvec_type_list_list_initial[mngr_idx])

    if verbose:
        if allocation_failure_counts:
            print(
                f"{allocation_failure_counts} of {allocation_all_counts} allocation attempts failed.")
        if minimization_failure_counts:
            print(
               f"{minimization_failure_counts} of {minimization_all_counts} minimization attempts failed.")

    del pvec_list_cp

    _best_pvec_list = [pvec.vector_k[:].copy() for pvec in best_pvec_list]

    return found_improvement, _best_pvec_list, best_bitvec_type_list_list, best_AIC, best_ast


class BaseOptimizer(object):

    def __init__(
        self, 
        system_manager_loader,
        name="BaseOptimizer",
        N_sys_per_likelihood_batch = 4,
        verbose=False):

        self._max_neighbor = 3

        self.name = name

        self.bounds_list = list()
        self._bounds_penalty_list_work = list()
        self.bounds_penalty_list = list()

        self.parameter_manager_list = list()
        self.exclude_others = list()
        self.parameter_name_list = list()
        self.scaling_factor_list = list()
        self.best_pvec_list = list()
        self.best_bitvec_type_list = list()

        self.N_parms_traj = list()
        self.pvec_traj    = list()
        self.bitvec_traj  = list()
        self.like_traj    = list()
        self.aic_traj     = list()
        
        self._N_steps   = 0
        self._N_mngr    = 0

        self.system_manager_loader = system_manager_loader
        self.system_list = list()
        self.system_name_list = list()
        self.targetcomputer_id_dict = dict()

        self.grad_diff = _EPSILON
        self.grad_diff_gs = _EPSILON_GS

        ### Note, don't make this too small.
        ### not finding the correct splits
        self.perturbation = 1.e-1

        self.verbose = verbose

        self.parm_mngr_cache_dict = dict()
        self.bsm_cache_dict  = dict()

        self._N_sys_per_likelihood_batch = N_sys_per_likelihood_batch

        self.obs = None

        self.frozen_types_list = list()

        self._ray_cleanup = list()


    def _set_system_list(self, smiles_list):

        import warnings
        from .system import System

        if len(self.system_manager_loader.smiles_list) == 0:
            self.system_manager_loader._generate_rdmol_dict()

        ### First check if we have to flatten the list
        import copy
        isflat = False
        _smiles_list_new = copy.deepcopy(smiles_list)
        while not isflat:
            isflat = True
            _smiles_list = list()
            for smi in _smiles_list_new:
                if isinstance(smi, (list, tuple, set)):
                    _smiles_list.extend(
                        list(smi))
                    isflat = False
                else:
                    _smiles_list.append(smi)
            _smiles_list_new = copy.deepcopy(_smiles_list)
        _smiles_list = list(set(_smiles_list_new))

        self.clear_cache()
        del self.system_list[:]
        del self.system_name_list[:]
        for sys_idx in self.targetcomputer_id_dict:
            self._ray_cleanup.append(
                self.targetcomputer_id_dict[sys_idx])
        self.targetcomputer_id_dict.clear()
        system_manager = self.system_manager_loader.generate_systemmanager(_smiles_list)

        for smi in smiles_list:
            if smi in system_manager._rdmol_list:
                sys_idx = system_manager._rdmol_list.index(smi)
                if isinstance(system_manager._system_list[sys_idx], System):
                    self.system_list.append(
                        system_manager._system_list[sys_idx])
                    self.system_name_list.append(smi)
                else:
                    warnings.warn(
                            f"Could not load system {smi}")
            else:
                warnings.warn(
                        f"Could not load system {smi}")


    def set_targetcomputer(self, system_idx_list, error_factor=1.):

        from .targets import TargetComputer
        from .system import System

        for sys_idx_pair in system_idx_list:
            targetcomputer = TargetComputer.remote(
                [self.system_list[sys_idx] for sys_idx in sys_idx_pair],
                target_type_list=None,
                error_factor=error_factor)
            self.targetcomputer_id_dict[sys_idx_pair] = targetcomputer


    def add_parameters(
        self,
        parameter_manager,
        parameter_name_list = None,
        exclude_others = False,
        scale_list = None,
        bounds = None,
        bounds_penalty = 1.,
        ):

        from . import arrays
        from . import parameters
        from . import bitvector_typing
        from .tools import remove_types
        import numpy as np
        from .system import System

        if parameter_manager.N_systems != 0:
            raise ValueError(
                f"parameter_manager must be empty, but found {parameter_manager.N_systems} systems.")

        self.system_manager_loader.add_parameter_manager(
            parameter_manager)
        ### 100 attempts to build initial type
        found_system = False
        for smi in self.smiles_list:
            self._set_system_list([smi])
            if len(self.system_list) > 0:
                found_system = True
                break
        if not found_system:
            raise ValueError(
                f"Could not build initial system with parameter manager {self._N_mngr}")

        self.parameter_manager_list.append(parameter_manager)
        parm_mngr = self.generate_parameter_manager(self.N_mngr)

        self.parameter_name_list.append(parameter_name_list)
        self.exclude_others.append(exclude_others)
        self.scaling_factor_list.append(scale_list)

        self.bounds_list.append(bounds)
        self._bounds_penalty_list_work.append(1.)
        self.bounds_penalty_list.append(bounds_penalty)

        ### Must increment at the end, not before.
        self._N_mngr += 1

        bsm, bitvec_list_alloc_dict = self.generate_bitsmartsmanager(
            self.N_mngr-1,
            max_neighbor=3)

        self.best_pvec_list.append(None)

        ### This adds [*:1]~[*:2] as initial bitvector
        self.best_bitvec_type_list.append(
            list())

        pvec_list, _ = self.generate_parameter_vectors()
        pvec = pvec_list[-1]
        if pvec.force_group_count == 0:
            value_list = list()
            for key in pvec.parameter_manager.default_parms:
                if key in pvec.parameter_name_list:
                    value_list.append(
                        pvec.parameter_manager.default_parms[key])
            pvec.add_force_group(value_list)

        import copy
        _bond_ring = copy.deepcopy(
            bitvector_typing.BitSmartsManager.bond_ring)
        _bond_aromatic = copy.deepcopy(
            bitvector_typing.BitSmartsManager.bond_aromatic)
        bitvector_typing.BitSmartsManager.bond_ring = []
        bitvector_typing.BitSmartsManager.bond_aromatic = []

        if isinstance(parameter_manager, parameters.BondManager):
            bvc = bitvector_typing.BondBitvectorContainer
            frozen_type_definitions = [
                ["[*:1]~[*:2]",  [None, None]],
                ["[*:1]-[#1:2]", [None, None]]
                ]
        elif isinstance(parameter_manager, parameters.AngleManager):
            bvc = bitvector_typing.AngleBitvectorContainer
            frozen_type_definitions = [
                ["[*:1]~[*:2]~[*:3]", [None, None]],
                ["[*:1]#[*:2]~[*:3]", [None, (180. * _ANGLE_DEG).value_in_unit(_ANGLE)]],
                ["[*:1]=[*:2]=[*:3]", [None, (180. * _ANGLE_DEG).value_in_unit(_ANGLE)]]
                ]
        elif isinstance(parameter_manager, (parameters.ProperTorsionManager, parameters.MultiProperTorsionManager)):
            bvc = bitvector_typing.ProperTorsionBitvectorContainer
            frozen_type_definitions = [
                ["[*:1]~[*:2]~[*:3]~[*:4]", [None] * len(parameter_manager.periodicity_list)],
                ["[*:1]~[*:2]~[*:3]#[*:4]", [0.] * len(parameter_manager.periodicity_list)],
                ["[*:1]~[*:2]#[*:3]~[*:4]", [0.] * len(parameter_manager.periodicity_list)],
                ["[*:1]~[*:2]=[*:3]=[*:4]", [0.] * len(parameter_manager.periodicity_list)],
                ]
        else:
            raise ValueError(
                f"Parameter manager of type {type(parameter_manager)} not known.")

        self.frozen_types_list.append(dict())
        for parm_idx, frozen_parms in enumerate(frozen_type_definitions):
            sma, parms = frozen_parms
            _bsm = bitvector_typing.BitSmartsManager(
                bvc([sma]), 
                max_neighbor=3)
            _bsm.generate(ring_safe=True)
            _, _bitvec_list_alloc_dict = _bsm.prepare_bitvectors(max_neighbor=3)
            alloc_dict, smarts_dict, on_dict, subset_dict, bitvec_dict = _bsm.and_rows(
                max_iter=0, generate_smarts=True, duplicate_removal=False)
            for key in bitvec_dict:
                b = bitvec_dict[key]
                self.frozen_types_list[-1][b] = [idx for idx, p in enumerate(parms) if not isinstance(p, type(None))]
            ### Only add it once to the best_bitvec_type_list.
            ### The frozen type list should contain all permutations.
            self.best_bitvec_type_list[-1].append(b)
            _bsm.bitvector_to_smarts(b) # This should work
            if self.verbose:
                print(
                    f"Adding initial type {_bsm.bitvector_to_smarts(b)}")
            if parm_idx > 0:
                pvec.duplicate(0, -1)
            N = pvec.parameters_per_force_group
            vector_0 = pvec.vector_0[:]
            for i in range(N):
                if not isinstance(parms[i], type(None)):
                    vector_0[parm_idx*N+i] = parms[i]            
            if len(vector_0[:]) > 0:
                pvec.vector_0 = vector_0
        pvec[:] = 0.
        pvec.apply_changes()

        bitvector_typing.BitSmartsManager.bond_ring = _bond_ring
        bitvector_typing.BitSmartsManager.bond_aromatic = _bond_aromatic

        self.best_pvec_list[-1] = pvec.vector_k[:].copy()


    @property
    def N_systems(self):
        return len(self.system_list)


    @property
    def N_mngr(self):
        return self._N_mngr


    def get_number_of_parameters(
        self,
        mngr_idx_list = list(),
        system_idx_list = list()
        ):

        if len(mngr_idx_list) == 0:
            _mngr_idx_list = list(range(self.N_mngr))
        else:
            _mngr_idx_list = mngr_idx_list

        if len(system_idx_list) == 0:
            _system_idx_list = list(range(self.N_systems))
        else:
            _system_idx_list = system_idx_list

        if max(_mngr_idx_list) > (self.N_mngr-1):
            raise ValueError(
                "No element in `mngr_idx_list` can be larger then maximum number of parameter managers."
                )

        if max(_system_idx_list) > (self.N_systems-1):
            raise ValueError(
                "No element in `system_idx_list` can be larger then maximum number of systems."
                )

        N_parms = 0
        for mngr_idx in _mngr_idx_list:
            parm_mngr = self.generate_parameter_manager(
                mngr_idx,
                _system_idx_list,
                )
            N_parms_per_force_group = len(self.parameter_name_list[mngr_idx])
            N_parms += parm_mngr.force_group_count * N_parms_per_force_group

        return N_parms

    
    def generate_parameter_vectors(
        self, 
        system_idx_list = list(),
        as_copy = False,
        copy_include_systems = False,
        adjust_scale = True,
        ):

        import copy

        if len(system_idx_list) == 0:
            _system_idx_list = list(range(self.N_systems))
        else:
            _system_idx_list = system_idx_list

        if max(_system_idx_list) > (self.N_systems-1):
            raise ValueError(
                "No element in `system_idx_list` can be larger then maximum number of systems."
                )

        _system_idx_list = sorted(_system_idx_list)
        pvec_list = list()
        bitvec_type_list = list()
        for mngr_idx in range(self.N_mngr):
            parm_mngr = self.generate_parameter_manager(
                mngr_idx, 
                _system_idx_list
                )
            scaling_factor_list = list()
            if adjust_scale:
                p = self._bounds_penalty_list_work[mngr_idx]
            else:
                p = 1.
            if p < 1.:
                p = 1.
            for s in self.scaling_factor_list[mngr_idx]: 
                scaling_factor_list.append(s/p**2)
            pvec = ForceFieldParameterVector(
                parm_mngr,
                self.parameter_name_list[mngr_idx],
                #self.scaling_factor_list[mngr_idx],
                scaling_factor_list,
                exclude_others = self.exclude_others[mngr_idx]
                )

            _, bitvec_list_alloc_dict = self.generate_bitsmartsmanager(
                mngr_idx, 
                _system_idx_list
                )
            
            bitvec_type_list.append(
                copy.deepcopy(
                    self.best_bitvec_type_list[mngr_idx]
                    )
                )
            from .bitvector_typing import bitvec_hierarchy_to_allocations
            allocations = [None for _ in pvec.allocations]
            bitvec_hierarchy_to_allocations(
                bitvec_list_alloc_dict,
                bitvec_type_list[-1],
                allocations,
                )
            if not isinstance(self.best_pvec_list[mngr_idx], type(None)):
                pvec.allocations[:] = allocations[:]
                pvec.reset(
                    self.best_pvec_list[mngr_idx],
                    pvec.allocations
                    )
                ### No need to call pvec.apply_changes()
                ### here. pvec.reset already does that.

            if as_copy:
                pvec_list.append(
                    pvec.copy(include_systems=copy_include_systems))
            else:
                pvec_list.append(
                    pvec)

        if as_copy and copy_include_systems:
            system_list = pvec_list[0].parameter_manager.system_list
            for i in range(len(pvec_list)):
                pvec_list[i].parameter_manager.system_list = system_list

        return pvec_list, bitvec_type_list


    def calc_log_likelihood(
        self, 
        system_idx_list = list(),
        as_dict = False,
        ):
        
        import numpy as np
        from .system import System

        if len(system_idx_list) == 0:
            _system_idx_list = list(range(self.N_systems))
        else:
            _system_idx_list = system_idx_list

        worker_id_dict = dict()
        for sys_idx in _system_idx_list:
            sys_dict = dict()
            if isinstance(sys_idx, int):
                sys = self.system_list[sys_idx]
                sys_dict[sys.name, sys_idx] = sys.openmm_system
            else:
                for _sys_idx in sys_idx:
                    sys = self.system_list[_sys_idx]
                    sys_dict[sys.name, _sys_idx] = sys.openmm_system
            targetcomputer = self.targetcomputer_id_dict[sys_idx]
            worker_id = ray.get(
                targetcomputer.__call__.remote(sys_dict, False))
            worker_id_dict[worker_id] = sys_idx

        if as_dict:
            logP_likelihood = dict()
        else:
            logP_likelihood = 0.
        worker_id_list = list(worker_id_dict.keys())
        while worker_id_list:
            [worker_id], worker_id_list = ray.wait(worker_id_list)
            try:
                _logP_likelihood, _ = ray.get(worker_id, timeout=_TIMEOUT)
                failed = False
            except:
                if _VERBOSE:
                    import traceback
                    print(traceback.format_exc())
                failed = True
            if not failed:
                sys_idx = worker_id_dict[worker_id]
                if as_dict:
                    logP_likelihood[sys_idx] = _logP_likelihood
                else:
                    logP_likelihood += _logP_likelihood
                del worker_id_dict[worker_id]

        return logP_likelihood


    def update_best(
        self,         
        mngr_idx,
        pvec,
        b_list,
        ):

        import copy

        self.clear_cache([mngr_idx])
        if hasattr(pvec, "vector_k"):
            import warnings
            if pvec.force_group_count != len(b_list):
                warnings.warn(
                        f"Number of force groups {pvec.force_group_count} and length of bitvector list {len(b_list)} must be identical.")
            self.best_pvec_list[mngr_idx] = pvec.vector_k[:].copy()
        else:
            self.best_pvec_list[mngr_idx] = pvec[:].copy()
        self.best_bitvec_type_list[mngr_idx] = copy.deepcopy(b_list)


    def clear_cache(
        self, 
        mngr_idx_list = list()
        ):

        if len(mngr_idx_list) == 0:
            _mngr_idx_list = list(range(self.N_mngr))
        else:
            _mngr_idx_list = mngr_idx_list

        for mngr_idx in _mngr_idx_list:
            to_delete = list()
            for key in self.parm_mngr_cache_dict:
                _mngr_idx, _ = key
                if _mngr_idx == mngr_idx:
                    to_delete.append(key)
            for key in to_delete:
                del self.parm_mngr_cache_dict[key]

            to_delete = list()
            for key in self.bsm_cache_dict:
                _mngr_idx, _, _ = key
                if _mngr_idx == mngr_idx:
                    to_delete.append(key)
            for key in to_delete:
                del self.bsm_cache_dict[key]


    def generate_parameter_manager(
        self,
        mngr_idx,
        system_idx_list=list(),
        ):

        import copy
        import ray
        from .system import System

        _CHUNK_SIZE = 20

        if len(system_idx_list) == 0:
            _system_idx_list = list(range(self.N_systems))
        else:
            _system_idx_list = system_idx_list

        if max(_system_idx_list) > (self.N_systems-1):
            raise ValueError(
                "No element in `system_idx_list` can be larger then maximum number of systems."
                )

        _system_idx_list = sorted(_system_idx_list)
        key = mngr_idx, tuple(_system_idx_list)
        if key in self.parm_mngr_cache_dict:
            parm_mngr = self.parm_mngr_cache_dict[key]
        else:
            worker_id_list = list()
            parm_mngr = copy.deepcopy(
                self.parameter_manager_list[mngr_idx])
            parm_mngr_id = ray.put(parm_mngr)
            s_list   = tuple()
            idx_list = tuple()
            for sys_idx in _system_idx_list:
                sys = self.system_list[sys_idx]
                s_list   += (sys,)
                idx_list += (sys_idx,)
                if len(s_list) == _CHUNK_SIZE:
                    worker_id = generate_parameter_manager.remote(
                        s_list, parm_mngr_id)
                    worker_id_list.append([worker_id, idx_list])
                    s_list   = tuple()
                    idx_list = tuple()
            if len(s_list) > 0:
                worker_id = generate_parameter_manager.remote(
                        s_list, parm_mngr_id)
                worker_id_list.append([worker_id, idx_list])
            sys_counts = 0
            for worker_id, idx_list in worker_id_list:
                _parm_mngr = ray.get(worker_id)
                parm_mngr.add_parameter_manager(_parm_mngr)
                for sys_idx in idx_list:
                    sys = self.system_list[sys_idx]
                    parm_mngr.system_list[sys_counts] = sys
                    sys_counts += 1
            self.parm_mngr_cache_dict[key] = parm_mngr

        return self.parm_mngr_cache_dict[key]


    def generate_bitsmartsmanager(
        self, 
        mngr_idx, 
        system_idx_list=list(),
        parent_smarts=None,
        max_neighbor=3
        ):

        from .bitvector_typing import BitSmartsManager

        if len(system_idx_list) == 0:
            _system_idx_list = list(range(self.N_systems))
        else:
            _system_idx_list = system_idx_list

        if max(_system_idx_list) > (self.N_systems-1):
            raise ValueError(
                "No element in `system_idx_list` can be larger then maximum number of systems."
                )

        if parent_smarts == None:
            parent_smarts = "~".join(
                [f"[*:{i+1:d}]" for i in range(self.parameter_manager_list[mngr_idx].N_atoms)]
                )

        _system_idx_list = sorted(_system_idx_list)
        key = mngr_idx, tuple(_system_idx_list), parent_smarts

        if key not in self.bsm_cache_dict:
            parm_mngr = self.generate_parameter_manager(
                mngr_idx,
                _system_idx_list,
                )
            bsm = BitSmartsManager(
                parm_mngr,
                parent_smarts=parent_smarts,
                max_neighbor=max_neighbor,
                )
            bsm.generate(ring_safe=True)
            _, bitvec_list_alloc_dict = bsm.prepare_bitvectors()

            self.bsm_cache_dict[key] = bsm, bitvec_list_alloc_dict

        return self.bsm_cache_dict[key]


    def save_traj(
        self, 
        parm_penalty=1.
        ):

        pvec_list_cp, bitvec_list = self.generate_parameter_vectors(as_copy=True)
        N_parms_all  = self.get_number_of_parameters()

        self.pvec_traj.append(pvec_list_cp)
        self.bitvec_traj.append(bitvec_list)
        self.N_parms_traj.append(N_parms_all)


    @property
    def rdmol_dict(self):
        if len(self.system_manager_loader.rdmol_dict) == 0:
            self.system_manager_loader._generate_rdmol_dict()
        return self.system_manager_loader.rdmol_dict


    @property
    def smiles_list(self):
        if len(self.system_manager_loader.smiles_list) == 0:
            self.system_manager_loader._generate_rdmol_dict()
        return self.system_manager_loader.smiles_list


    def generate_clustering(self):

        from rdkit.Chem import rdMolDescriptors as rdmd
        from rdkit.Chem import DataStructs
        import numpy as np
        from scipy import cluster

        nBits = 1024
        fp_list = np.zeros((self.N_all_systems, nBits), dtype=np.int8)
        for sys_idx, smi in enumerate(self.rdmol_dict):
            rdmol = self.rdmol_dict[smi]
            fp = rdmd.GetMorganFingerprintAsBitVect(
                    rdmol, 3, nBits=nBits, useChirality=False)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_list[sys_idx,:] = arr

        if self.N_all_systems > 1:
            self.obs = cluster.vq.whiten(fp_list.astype(float))
        else:
            self.obs = fp_list.astype(float)


    @property
    def N_all_systems(self):
        return len(self.rdmol_dict)


    def get_random_system_idx_list(
        self,
        N_sys_per_batch,
        N_batches,
        cluster_systems = False,
        ):

        from scipy import cluster
        import copy
        import numpy as np

        if (int(N_sys_per_batch) >= int(self.N_all_systems)) or (int(self.N_all_systems) == 1):
            N_batches = 1
            N_sys_per_batch = self.N_all_systems
            cluster_systems = False

        if cluster_systems:
            centroid, label = cluster.vq.kmeans2(
                    self.obs, 
                    N_sys_per_batch, 
                    minit='points', # setting to random may lead to matrix decomposition error
                    iter=1000)
            label_re = [list() for _ in range(N_sys_per_batch)]
            dists    = [list() for _ in range(N_sys_per_batch)]
            for i in range(self.N_all_systems):
                k = label[i]
                label_re[k].append(i)
                d = np.linalg.norm(centroid[k] - self.obs[i])
                dists[k].append(d)
            system_idx_list_batch = tuple()
            for _ in range(N_batches):
                sys_list = tuple()
                N_iter_max = N_sys_per_batch * 10
                N_iter     = 0
                k          = 0
                while N_iter < N_iter_max:                    
                    if len(label_re[k]) == 0:
                        i = int(np.random.randint(0, self.N_systems))
                    elif len(label_re[k]) == 1:
                        i = label_re[k][0]
                    else:
                        p = np.array(dists[k])
                        _min = p.min()
                        _max = p.max()
                        diff = _max - _min
                        if diff < 1.e-7:
                            diff = 1.
                        p = (p - _min) / diff
                        p = 1. - p
                        p = p/np.sum(p)
                        if np.any(np.isnan(p)):
                            p = None
                        i = int(np.random.choice(label_re[k], p=p))
                    if i not in sys_list:
                        sys_list += (i,)
                        k += 1
                    N_iter += 1
                    if k == N_sys_per_batch:
                        break

                sys_list = list(sys_list)
                sys_list = tuple(sorted(sys_list))
                if (sys_list not in system_idx_list_batch) and len(sys_list) > 0:
                    system_idx_list_batch += tuple([sys_list])
        else:
            import numpy as np
            system_idx_list = np.arange(
                self.N_all_systems, 
                dtype=int
                )

            system_idx_list_batch = tuple()
            N_iter_max = N_batches * 10
            N_iter     = 0
            while N_iter < N_iter_max:
                sys_list = np.random.choice(
                    system_idx_list, size=N_sys_per_batch, replace=False).tolist()
                sys_list = tuple(sorted(sys_list))
                if (sys_list not in system_idx_list_batch) and len(sys_list) > 0:
                    system_idx_list_batch += tuple([sys_list])
                if len(system_idx_list_batch) == N_batches:
                    break
                N_iter += 1

        ### Gather list of smiles and
        ### generate systems
        smiles_list= list()
        for sys_idx_list in system_idx_list_batch:
            for sys_idx in sys_idx_list:
                smiles_list.append(
                    self.smiles_list[sys_idx])
        smiles_list = list(set(smiles_list))
        ### We need this `self._generator_smiles_list` in order
        ### to restart the optimization run.
        ### Important to first generate the systems and
        ### then parse the system_name_list instead of the initial
        ### smiles_list, since not all entries in smiles_list will
        ### actually be processed.
        self._set_system_list(smiles_list)
        self._generator_smiles_list = copy.deepcopy(
                self.system_name_list)
        ### Re-order and re-index 
        ### system_idx_list_batch

        return self._reorder_system_idxs(system_idx_list_batch)


    def _reorder_system_idxs(self, system_idx_list_batch):

        ### sys_map_dict maps from smiles_list ordering
        ### to system_list ordering
        sys_map_dict = dict()
        for sys_idx, _ in enumerate(self.system_list):
            ### sys_idx in `self.system_list` counting
            ### `self.smiles_list` in system_idx_list_batch
            name = self.system_name_list[sys_idx]
            if name in self.smiles_list:
                _sys_idx = self.smiles_list.index(name)
                sys_map_dict[_sys_idx] = sys_idx

        system_idx_list_batch_new = tuple()
        for sys_idx_list in system_idx_list_batch:
            sys_list = list()
            for sys_idx in sys_idx_list:
                if sys_idx in sys_map_dict:
                    sys_list.append(sys_map_dict[sys_idx])
            sys_list = tuple(sorted(sys_list))
            if len(sys_list) > 0:
                system_idx_list_batch_new += tuple([sys_list])

        return system_idx_list_batch_new


    def split_bitvector(
        self,
        mngr_idx,
        bitvec_type_list,
        system_idx_list,
        pvec_start_list=None,
        N_trials_gradient=5,
        split_all=False,
        max_on=_MAX_ON,
        max_splits=100,
        ):

        from . import arrays
        from .bitvector_typing import bitvec_hierarchy_to_allocations
        import numpy as np
        import copy
        import ray
        from scipy import cluster

        bsm, bitvec_alloc_dict_list = self.generate_bitsmartsmanager(
            mngr_idx,
            system_idx_list
            )
        pvec_list, _ = self.generate_parameter_vectors(
            system_idx_list,
            as_copy=True,
            copy_include_systems=True,
            adjust_scale = False,
            )
        if not isinstance(pvec_start_list, type(None)):
            for _mngr_idx in range(self.N_mngr):
                pvec_list[_mngr_idx].reset(
                    pvec_start_list[_mngr_idx],
                    pvec_list[_mngr_idx].allocations)

        pvec    = pvec_list[mngr_idx]
        N_types = pvec.force_group_count
        if split_all:
            type_query_list = list(range(N_types))
        else:
            type_query_list = [N_types-1]

        worker_id_list = list()
        alloc_bitvec_degeneracy_dict = dict()
        for type_i in type_query_list:
            type_j = type_i + 1
            b_old  = bitvec_type_list[type_i]
            ### We want to check if b_old and any child-type
            ### of b_old must be frozen
            freeze_parm = False
            for _b_old in self.frozen_types_list[mngr_idx]:
                ### _b_old is subset of b_old
                if _b_old == (_b_old & b_old):
                    ### Only `b_old` is stored in self.frozen_types_list
                    N = len(self.frozen_types_list[mngr_idx][_b_old])
                    ### Only if all parameters of this type are
                    ### frozen, we do not proceed here.
                    if N == pvec.parameters_per_force_group:
                        freeze_parm = True
                        break
                if freeze_parm:
                    break
            if freeze_parm:
                continue
            selection_i = pvec.allocations.index([type_i])[0].tolist()

            if len(selection_i) == 0:
                continue
            N_candidates = 0
            _max_on = max_on
            while N_candidates == 0:
                alloc_dict, smarts_dict, on_dict, subset_dict, bitvec_dict = bsm.and_rows(
                    max_iter = 3,
                    allocations = selection_i,
                    generate_smarts = False,
                    max_neighbor = 3,
                    max_on = _max_on,
                    duplicate_removal = False,
                    verbose = self.verbose,
                    )

                on_dict_sorted = sorted(on_dict.items(), key= lambda x: x[1])
                N_candidates = len(on_dict_sorted)
                if N_candidates == 0:
                    if self.verbose:
                        print(
                            f"Did not find any split candidates. Increasing `max_on`")
                    _max_on += 0.01
            if self.verbose:
                print(
                    f"Found {N_candidates} candidate bitvectors for type {type_i} in mngr {mngr_idx} and systems {system_idx_list}."
                    )

            k_values_ij = np.zeros(
                (
                    max_splits, 
                    pvec.allocations.size
                    ),
                dtype=np.int16)

            counts = 0
            for t_idx, _ in on_dict_sorted:
                b_new = bitvec_dict[t_idx]
                if b_old == b_new:
                    continue
                bitvec_type_list.insert(type_j, b_new)
                allocations = [-1 for _ in pvec.allocations]
                bitvec_hierarchy_to_allocations(
                    bitvec_alloc_dict_list,
                    bitvec_type_list,
                    allocations
                    )
                check  = allocations.count(-1) == 0
                check *= allocations.count(type_i) > 0
                check *= allocations.count(type_j) > 0
                for a,t in enumerate(allocations):
                    if a in selection_i:
                        check *= t in [type_i, type_j]
                    else:
                        check *= not (t in [type_i, type_j])
                bitvec_type_list.pop(type_j)

                if check:
                    key = tuple(allocations), tuple(selection_i), tuple([type_i, type_j])
                    if key in alloc_bitvec_degeneracy_dict:
                        alloc_bitvec_degeneracy_dict[key].append(b_new)
                    else:
                        alloc_bitvec_degeneracy_dict[key] = [b_new]
                        k_values_ij[counts,:] = np.array(allocations, dtype=int)
                        counts += 1
                if counts == max_splits:
                    break

            if counts < max_splits:
                max_splits_effective = counts
                k_values_ij = k_values_ij[:counts]
            else:
                max_splits_effective = max_splits

            if self.verbose:
                print(
                    f"MNGR {mngr_idx} / type {type_i}:")
                b_i   = bitvec_type_list[type_i]
                try:
                    smi_i = bsm.bitvector_to_smarts(b_i)
                    for key_idx, key in enumerate(alloc_bitvec_degeneracy_dict):
                        _, _, t = key
                        if (t[0] == type_i) and (t[1] == type_j):
                            for b_j in alloc_bitvec_degeneracy_dict[key]:
                                smi_j = bsm.bitvector_to_smarts(b_j)
                                print(
                                    f"{key_idx}: {smi_i} --> {smi_j}")
                except Exception as e:
                    print(e)

            N_array_splits = 2
            if max_splits_effective > 10:
                N_array_splits = int(max_splits_effective/10)
            pvec_id = ray.put(pvec)
            for split_idxs in np.array_split(np.arange(max_splits_effective), N_array_splits):
                worker_id = get_gradient_scores.remote(
                    ff_parameter_vector = pvec_id,
                    targetcomputer = self.targetcomputer_id_dict[system_idx_list],
                    type_i = type_i,
                    selection_i = selection_i,
                    k_values_ij = k_values_ij[split_idxs],
                    grad_diff = self.grad_diff_gs,
                    N_trials = N_trials_gradient,
                    local_targets = False,
                    N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                    )
                worker_id_list.append(worker_id)

        return worker_id_list, alloc_bitvec_degeneracy_dict
    

class ForceFieldOptimizer(BaseOptimizer):

    def __init__(
        self, 
        system_manager_loader,
        parm_penalty_split = 1.,
        parm_penalty_merge = 1.,
        name="ForceFieldOptimizer",
        N_sys_per_likelihood_batch = 4,
        verbose=False):

        super().__init__(
            system_manager_loader, name, N_sys_per_likelihood_batch, verbose)

        self.parm_penalty_split = parm_penalty_split
        self.parm_penalty_merge = parm_penalty_merge


    def __getstate__(self):

        import copy

        state = self.__dict__.copy()
        del_key_list = [
                "parm_mngr_cache_dict",
                "bsm_cache_dict",
                "system_list",
                "system_name_list",
                "targetcomputer_id_dict",
                "obs",
                "system_manager_loader",
                "_ray_cleanup",
                ]
        for key in del_key_list:
            if key in state:
                del state[key]

        copy_key_list = [
                "minscore_worker_id_dict", 
                "selection_worker_id_dict"
                ]
        for key in copy_key_list:
            if key in state:
                state[key] = copy.deepcopy(state[key])
                for _key in state[key]:
                    state[key][_key] = None

        state["parm_mngr_cache_dict"] = dict()
        state["bsm_cache_dict"] = dict()

        gc.collect()
        
        return state


    def __setstate__(self, state):

        self.__dict__.update(state)
        self.system_list = list()
        self.system_name_list = list()
        self.targetcomputer_id_dict = dict()
        self._ray_cleanup = list()
        self.obs = None
        self.system_manager_loader = None


    def calculate_AIC(
        self,
        mngr_idx_list = list(),
        system_idx_list = list(),
        parm_penalty = 1.,
        as_dict = False
        ):

        if len(mngr_idx_list) == 0:
            _mngr_idx_list = list(range(self.N_mngr))
        else:
            _mngr_idx_list = mngr_idx_list

        if len(system_idx_list) == 0:
            _system_idx_list = list(range(self.N_systems))
        else:
            _system_idx_list = system_idx_list

        if as_dict:
            AIC_score = dict()
            for sys_idx in _system_idx_list:
                N_parms_all = 0.
                if isinstance(sys_idx, int):
                    _sys_idx_list = [sys_idx]
                else:
                    _sys_idx_list = sys_idx
                N_parms_all = self.get_number_of_parameters(
                    _mngr_idx_list,
                    _sys_idx_list
                    )
                AIC_score[sys_idx] = 2. * N_parms_all * parm_penalty
            logP_likelihood = self.calc_log_likelihood(
                _system_idx_list, 
                as_dict=True
                )
            for sys_idx in _system_idx_list:
                AIC_score[sys_idx] -= 2. * logP_likelihood[sys_idx]

        else:
            N_parms_all = self.get_number_of_parameters(
                _mngr_idx_list,
                _system_idx_list
                )
            AIC_score  = 2. * N_parms_all * parm_penalty
            AIC_score -= 2. * self.calc_log_likelihood(
                _system_idx_list,
                as_dict=False
                )

        return AIC_score


    def garbage_collection(
        self, 
        N_systems_validation = 10,
        N_iter_validation = 10,
        cluster_systems = True):

        import copy
        from .bitvector_typing import bitvec_hierarchy_to_allocations

        if self.verbose:
            print(
                "Garbage Collection: Looking for bad performing types.")

        # N_sys_per_batch ---> N_systems_validation
        # N_batches       ---> N_iter_validation
        sys_idx_list_validation = self.get_random_system_idx_list(
                N_systems_validation, N_iter_validation, cluster_systems)
        self.set_targetcomputer(sys_idx_list_validation)

        found_improvement = True
        garbage_cycle_counter = 0
        while found_improvement:
            best_mngr_type_dict = dict()
            AIC_best_dict = self.calculate_AIC(
                system_idx_list=sys_idx_list_validation,
                parm_penalty=self.parm_penalty_merge,
                as_dict=True)

            pvec_list, bitvec_type_list_list = self.generate_parameter_vectors()
            bitvec_list_alloc_dict_list = list()
            for mngr_idx in range(self.N_mngr):            
                _, bitvec_list_alloc_dict = self.generate_bitsmartsmanager(mngr_idx)
                bitvec_list_alloc_dict_list.append(
                    bitvec_list_alloc_dict)
            for mngr_idx in range(self.N_mngr):
                pvec             = pvec_list[mngr_idx]
                bitvec_type_list = bitvec_type_list_list[mngr_idx]
                bitvec_list_alloc_dict = bitvec_list_alloc_dict_list[mngr_idx]
                pvec_old         = pvec.copy()
                N_types          = pvec.force_group_count
                if N_types > 1:
                    for type_i in range(N_types):
                        type_i = N_types - type_i - 1
                        if bitvec_type_list[type_i] in self.frozen_types_list[mngr_idx]:
                            continue
                        best_mngr_type_dict[mngr_idx, type_i] = 0
                        b = bitvec_type_list.pop(type_i)
                        allocations = [-1 for _ in pvec.allocations]
                        bitvec_hierarchy_to_allocations(
                            bitvec_list_alloc_dict,
                            bitvec_type_list,
                            allocations,
                            )
                        if allocations.count(-1) == 0:
                            if type_i == 0:
                                pvec.allocations[:] = 1
                            else:
                                pvec.allocations[:] = 0
                            pvec.apply_changes()
                            pvec.remove(type_i)

                            pvec.allocations[:] = allocations
                            pvec.apply_changes()

                            AIC_query_dict = self.calculate_AIC(
                                system_idx_list=sys_idx_list_validation,
                                parm_penalty=self.parm_penalty_merge,
                                as_dict=True)
                            for sys_idx_pair in sys_idx_list_validation:
                                if AIC_query_dict[sys_idx_pair] < AIC_best_dict[sys_idx_pair]:
                                    best_mngr_type_dict[mngr_idx, type_i] += 1

                            pvec.reset(pvec_old)
                        bitvec_type_list.insert(type_i, b)

            found_improvement = False
            for mngr_idx, type_i in best_mngr_type_dict:
                counts = best_mngr_type_dict[mngr_idx, type_i]
                if counts > (N_iter_validation-1):
                    found_improvement = True

            if found_improvement:
                best_mngr_type = max(best_mngr_type_dict, key=best_mngr_type_dict.get)

            if self.verbose:
                print(
                    f"Garbage cycle {garbage_cycle_counter}:", end=" ")
                if found_improvement:
                    mngr_idx, type_i = best_mngr_type
                    print(
                        f"Removed bad type {type_i} from manager {mngr_idx} with improved AIC in {best_mngr_type_dict[mngr_idx, type_i]} systems.")
                else:
                    print(
                        f"Did not remove any bad type.")

            if found_improvement:
                mngr_idx, type_i = best_mngr_type
                pvec_list, bitvec_type_list_list = self.generate_parameter_vectors()
                _, bitvec_list_alloc_dict = self.generate_bitsmartsmanager(mngr_idx)
                pvec             = pvec_list[mngr_idx]
                bitvec_type_list = bitvec_type_list_list[mngr_idx]
                N_types          = pvec.force_group_count

                if type_i == 0:
                    pvec.allocations[:] = 1
                else:
                    pvec.allocations[:] = 0
                pvec.apply_changes()
                pvec.remove(type_i)
                bitvec_type_list.pop(type_i)

                allocations = [None for _ in pvec.allocations]
                bitvec_hierarchy_to_allocations(
                    bitvec_list_alloc_dict,
                    bitvec_type_list,
                    allocations,
                    )
                pvec.allocations[:] = allocations
                pvec.apply_changes()

                self.update_best(
                    mngr_idx,
                    pvec,
                    bitvec_type_list
                    )

                if self.verbose:
                    print(
                        f"Updated parameters for mngr {mngr_idx}:",
                        pvec.vector_k,
                        )
                    print(
                        f"Updated allocations for mngr {mngr_idx}:",
                        pvec.allocations
                    )
                    print(
                        f"Updated bitvec hashes for mngr {mngr_idx}:",
                        bitvec_type_list
                        )
                    print(
                        f"Updated SMARTS hierarchy for mngr {mngr_idx}:"
                        )

                    bsm, _ = self.generate_bitsmartsmanager(mngr_idx)
                    for idx, b in enumerate(bitvec_type_list):
                        try:
                            sma = bsm.bitvector_to_smarts(b)
                        except:
                            sma = "???"
                        print(
                            f"Type {idx} ({pvec.force_group_histogram[idx]}): {sma}"
                            )

            garbage_cycle_counter += 1


    def get_votes(
        self,
        worker_id_list,
        low_to_high=True,
        abs_grad_score=False,
        norm_cutoff=0.,
        grad_cutoff=0.,
        keep_N_best=10,
        ):

        import numpy as np

        if len(worker_id_list) == 0:
            return list()

        votes_dict = dict()
        alloc_bitvec_dict = dict()

        score_dict  = dict()
        norm_dict   = dict()
        counts_dict = dict()
        
        while worker_id_list:
            [worker_id], worker_id_list = ray.wait(
                worker_id_list)
            try:
                result = ray.get(worker_id, timeout=_TIMEOUT)
                failed = False
            except:
                if _VERBOSE:
                    import traceback
                    print(traceback.format_exc())
                failed = True
            if not failed:
                grad_score_dict, grad_norm_dict, allocation_list, selection_list, type_list = result

                for trial_idx in grad_score_dict:

                    grad_score = grad_score_dict[trial_idx]
                    grad_norm  = grad_norm_dict[trial_idx]

                    for g, n, a, s, t in zip(grad_score, grad_norm, allocation_list, selection_list, type_list):
                        ### Only use this split, when the norm
                        ### is larger than norm_cutoff
                        #print(
                        #    f"Gradient scores {g}",
                        #    f"Gradient norms {n}",
                        #    )

                        ast = a,s,t
                        ### Only consider the absolute value of the gradient score, not the sign
                        if abs_grad_score:
                            _g = [abs(gg) for gg in g]
                            g  = _g
                        if ast in score_dict:
                            score_dict[ast]   += g
                            counts_dict[ast]  += 1
                            norm_dict[ast][0] += abs(n[0])
                            norm_dict[ast][1] += abs(n[1])
                        else:
                            score_dict[ast]  = g
                            counts_dict[ast] = 1
                            norm_dict[ast]   = [abs(n[0]), abs(n[1])]

        to_delete = list()
        for ast in score_dict:
            score_dict[ast]   /= counts_dict[ast]
            norm_dict[ast][0] /= counts_dict[ast]
            norm_dict[ast][1] /= counts_dict[ast]
            #if norm_dict[ast][0] < norm_cutoff and norm_dict[ast][1] < norm_cutoff:
            #    to_delete.append(ast)
            ### For now, let's just sort by gradient score and always keep
            ### a number of solutions.
            #else:
            #    if low_to_high:
            #        if score_dict[ast] > grad_cutoff:
            #            to_delete.append(ast)
            #    else:
            #        if score_dict[ast] < grad_cutoff:
            #            to_delete.append(ast)

        for ast in to_delete:
            del score_dict[ast]
        
        ### Sort dictionaries
        ### For merge: high values favored, i.e. high values first
        score_list = [k for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=False)]
        if not low_to_high:
            ### For split: small values favored, i.e. small values first
            score_list = score_list[::-1]

        ### Only keep the final best `keep_N_best` ast
        if len(score_list) > keep_N_best:
            score_list = score_list[:keep_N_best]

        return score_list


    def get_min_scores(
        self,
        mngr_idx_main,
        bitvec_dict,
        system_idx_list = list(),
        votes_split_list = list(),
        local_targets = False,
        ):

        import copy

        if len(system_idx_list) == 0:
            system_idx_list = list(range(self.N_systems))

        pvec_all, bitvec_type_list = self.generate_parameter_vectors(
            system_idx_list)
        system_list = pvec_all[0].parameter_manager.system_list

        system_list_id = ray.put(system_list)
        pvec_cp = pvec_all[mngr_idx_main].vector_k[:].copy()
        allocations_cp = copy.deepcopy(pvec_all[mngr_idx_main].allocations)

        worker_id_dict = dict()
        ### Query split candidates
        ### ======================
        for counts, ast in enumerate(votes_split_list):
            allocations, selection, type_ = ast
            type_i, type_j = type_
            bitvec_type_list_cp = copy.deepcopy(bitvec_type_list)
            bitvec_type_list_cp[mngr_idx_main].insert(type_j, bitvec_dict[ast][0])
            bitvec_type_list_id = ray.put(bitvec_type_list_cp)

            pvec_all[mngr_idx_main].duplicate(type_i, type_j)
            pvec_all[mngr_idx_main].allocations[:] = allocations[:]
            pvec_all[mngr_idx_main].apply_changes()

            pvec_all_cp = [pvec.copy() for pvec in pvec_all]

            worker_id = minimize_FF.remote(
                    system_list = system_list_id,
                    targetcomputer = self.targetcomputer_id_dict[system_idx_list],
                    pvec_list = pvec_all_cp,
                    frozen_types_list = self.frozen_types_list,
                    bitvec_type_list = bitvec_type_list_id,
                    bounds_list = self.bounds_list,
                    parm_penalty = self.parm_penalty_split,
                    local_targets = local_targets,
                    N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                    #bounds_penalty = self._bounds_penalty_list_work,
                    bounds_penalty = [1. for _ in range(self.N_mngr)],
                    use_global_opt = _USE_GLOBAL_OPT,
                    verbose = self.verbose,
                    )

            worker_id_dict[worker_id] = ast

            pvec_all[mngr_idx_main].reset(
                pvec_cp, allocations_cp)

        return worker_id_dict


    def run(
        self,
        ### Number of iterations in the outer loop.
        iterations=5,
        ### Maximum number of splitting attempts
        max_splitting_attempts=10,
        ### Number of Gibbs Parameter Type relaxation attempts
        max_gibbs_attempts=10,
        ### Number of gradient trials per parameter vector
        N_trials_gradient = 5,
        ### Every `pair_incr` iterations of the outer looop, 
        ### the number of systems per batch `N_sys_per_batch`
        ### is incremented by +1.
        pair_incr_split = 10,
        ### Initial number of systems per batch
        N_sys_per_batch_split = 1,
        ### Number of batches to use per loop
        N_batches = 1,
        ### Optimize ordering in which systems are computed
        optimize_system_ordering = True,
        ### Keep the `keep_N_best` solutions for each split
        keep_N_best = 10,
        ### Do a maximum of `N_max_splits` splits per parameter
        ### manager and batch of systems
        N_max_splits = 100,
        ### Maximum percentage of on-bits to be used as a cutoff
        ### during the `and`-based bitvector search.
        max_on = _MAX_ON,
        ### Offset for numbering the output files
        output_offset = 0,
        ### Should we try to pickup this run from where we stopped last time?
        restart = True,
        ### Benchmark compute time for each batch over `N_trails_opt` replicates.
        N_trials_opt = 20,
        ### dict with system indices for each optimization cycle. If None, will
        ### be determined randomly.
        system_idx_dict_batch = None,
        ### Whether to sample systems from clustering
        cluster_systems = False,
        ### Use this many systems for validation
        N_systems_validation = 10,
        ### Use this many validation iterations
        N_iter_validation = 10,
        ### By this fraction the error (i.e. the likelihood denominator) 
        ### will be decrease upon each splitting iteration.
        error_decrease_factor = 0.2,
        ### We reduce the scaling factor in the prior by this
        ### amount upon each splitting iteration.
        bounds_decrease_factor = 0.7,
        ### Number of solutions to keep from each manager
        ### for final (stage II) validation.
        MAX_VALIDATE = 10):

        self.generate_clustering()

        from .draw_bitvec import draw_bitvector_from_candidate_list
        from .bitvector_typing import bitvec_hierarchy_to_allocations
        from .system import System
        import pickle
         
        ### 100 attempts to build initial type
        found_system = False
        for _ in range(100):
            smi = np.random.choice(self.smiles_list)
            self._set_system_list([smi])
            if len(self.system_list) > 0:
                found_system = True
                break
        if not found_system:
            raise ValueError(
                f"Could not build initial system")
        pvec_list, bitvec_type_list = self.generate_parameter_vectors()
        for mngr_idx in range(self.N_mngr):
            self.update_best(
                mngr_idx,
                pvec_list[mngr_idx],
                bitvec_type_list[mngr_idx]
                )
        self.save_traj(parm_penalty=1.)        

        ### This enables deterministic paramter type moves that
        ### preserve the number of parameter types. We set this to
        ### `False` now since we want this to be done by the Gibbs
        ### Sampler. See parameter `max_gibbs_attempts` and `max_gibbs_post_split_attempts`.
        allow_move = False
        
        ### If we restart from a previous version, make sure
        ### that we do not restart
        if not hasattr(self, "system_idx_list_batch"):
            restart = False
        if not hasattr(self, "minscore_worker_id_dict"):
            restart = False
        if not hasattr(self, "selection_worker_id_dict"):
            restart = False
        if not hasattr(self, "bitvec_dict"):
            restart = False
        if not hasattr(self, "split_iteration_idx"):
            restart = False
        if not hasattr(self, "accepted_counter"):
            restart = False
        if not hasattr(self, "best_ast_dict"):
            restart = False
        if not hasattr(self, "best_pvec_dict"):
            restart = False
        if not hasattr(self, "best_aic_dict"):
            restart = False
        if not hasattr(self, "_optimization_generator_smiles_list"):
            restart = False
        if not hasattr(self, "_validation_generator_smiles_list"):
            restart = False
        if not hasattr(self, "_N_sys_per_likelihood_batch"):
            self._N_sys_per_likelihood_batch = 4

        if not restart:
            self._optimization_generator_smiles_list = list()
            self._validation_generator_smiles_list = list()
            self.system_idx_list_batch = []
            self.minscore_worker_id_dict = dict()
            self.selection_worker_id_dict = dict()
            self.best_ast_dict  = dict()
            self.best_pvec_dict = dict()
            self.best_aic_dict  = dict()
            self.bitvec_dict = dict()
            self.split_iteration_idx = 0
            self.accepted_counter = 0
            self._ray_cleanup = list()
            self.targetcomputer_id_dict.clear()

        if self.verbose and restart:
            print("Attempting to restart from previous run.")

        self.clear_cache()

        ### These will be set automatically, if `optimize_system_ordering`
        ### is switched on.
        N_sys_per_likelihood_batch, use_local_targets = 1, False
        for iteration_idx in range(iterations):

            if iteration_idx > 0:
                if (iteration_idx%pair_incr_split) == 0:
                    N_sys_per_batch_split += 1
                    N_systems_validation  += 1

            raute_fill = ''.ljust(len(str(iteration_idx))-1,"#")
            print("ITERATION", iteration_idx)
            print(f"###########{raute_fill}")

            self._bounds_penalty_list_work = list()
            for val in self.bounds_penalty_list:
                self._bounds_penalty_list_work.append(
                        val * (1. - bounds_decrease_factor)**iteration_idx)

            ### ============== ###
            ### PROCESS SPLITS ###
            ### ============== ###
            while self.split_iteration_idx < max_splitting_attempts:
                print(f"ATTEMPTING SPLIT {iteration_idx}/{self.split_iteration_idx}")
                if not restart:
                    if isinstance(system_idx_dict_batch, dict):
                        self.system_idx_list_batch = system_idx_dict_batch[iteration_idx]
                    else:
                        # BATCH VARIABLES IN TRAINING AND VALIDATION
                        # N_sys_per_batch ---> N_systems_validation
                        # N_batches       ---> N_iter_validation
                        print("Generating systems ...")
                        self.system_idx_list_batch = self.get_random_system_idx_list(
                            N_sys_per_batch_split, N_batches, cluster_systems)
                        self._optimization_generator_smiles_list = copy.deepcopy(
                            self._generator_smiles_list)
                else:
                    self._set_system_list(
                        self._optimization_generator_smiles_list)
                self.set_targetcomputer(
                    self.system_idx_list_batch,
                    error_factor=(1.-error_decrease_factor)**iteration_idx)
                if self.verbose:
                    print(
                        "Initial ordering of systems:",
                        self.system_idx_list_batch)
                if optimize_system_ordering:
                    if self.verbose:
                        print(
                            "Optimizing system priority timings ...")

                    worker_id_dict = dict()
                    time_diff_dict = dict()
                    for sys_idx_pair in self.system_idx_list_batch:
                        time_diff_dict[sys_idx_pair] = 0.
                        pvec_list, _ = self.generate_parameter_vectors(sys_idx_pair)
                        pvec    = pvec_list[0]
                        pvec_id = ray.put(pvec)
                        worker_id = _test_logL.remote(
                            pvec_id, 
                            self.targetcomputer_id_dict[sys_idx_pair],
                            N_trials_opt,
                            check_local=False)
                        worker_id_dict[worker_id] = sys_idx_pair

                    time_dict      = dict()
                    sys_dict       = dict()
                    for worker_id in worker_id_dict:
                        _time_dict = ray.get(worker_id)
                        sys_idx_pair = worker_id_dict[worker_id]
                        for key in _time_dict:
                            if key not in time_dict:
                                time_dict[key] = 0.
                                sys_dict[key]  = dict()
                            time_dict[key]             += _time_dict[key]
                            sys_dict[key][sys_idx_pair] = _time_dict[key]

                    N_sys_per_likelihood_batch, use_local_targets = min(time_dict, key=time_dict.get)
                    self._N_sys_per_likelihood_batch = N_sys_per_likelihood_batch

                    self.system_idx_list_batch = sorted(
                            sys_dict[N_sys_per_likelihood_batch, use_local_targets],
                            key=sys_dict[N_sys_per_likelihood_batch, use_local_targets].get,
                            reverse=True)
                    self.system_idx_list_batch = tuple(self.system_idx_list_batch)

                    del worker_id_dict
                    del time_dict
                    del sys_dict

                    if self.verbose:
                        print(
                            "Optimized ordering of systems:", self.system_idx_list_batch)
                        print(
                            f"N_sys_per_likelihood_batch: {N_sys_per_likelihood_batch} // local_targets {use_local_targets}")

                if not restart:
                    if self.verbose:
                        print(
                            "Optimize parameters.")
                    minimize_initial_worker_id_dict = dict()
                    for sys_idx_pair in self.system_idx_list_batch:
                        pvec_list, bitvec_type_list = self.generate_parameter_vectors(
                            system_idx_list=sys_idx_pair)
                        system_list = pvec_list[0].parameter_manager.system_list
                        worker_id = minimize_FF.remote(
                            system_list = system_list,
                            targetcomputer = self.targetcomputer_id_dict[sys_idx_pair],
                            pvec_list = pvec_list,
                            bitvec_type_list = bitvec_type_list,
                            frozen_types_list = self.frozen_types_list,
                            bounds_list = self.bounds_list,
                            ### This is the only place where we don't want to 
                            ### do local targets
                            local_targets = use_local_targets,
                            parm_penalty = 1.,
                            bounds_penalty = self._bounds_penalty_list_work,
                            #bounds_penalty = [1. for _ in range(self.N_mngr)],
                            N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                            use_global_opt = False,
                            verbose = self.verbose)
                        minimize_initial_worker_id_dict[worker_id] = sys_idx_pair

                    if self.verbose:
                        print(
                            "Generating splits and computing grad scores.")
                    split_worker_id_dict = dict()
                    worker_id_list = list(minimize_initial_worker_id_dict.keys())
                    while worker_id_list:
                        [worker_id], worker_id_list = ray.wait(worker_id_list)
                        try:
                            _, pvec_list_cp, _ = ray.get(worker_id)
                            sys_idx_pair = minimize_initial_worker_id_dict[worker_id]
                            for mngr_idx in range(self.N_mngr):
                                split_worker_id_dict[mngr_idx,sys_idx_pair] = self.split_bitvector(
                                        mngr_idx,
                                        self.best_bitvec_type_list[mngr_idx],
                                        sys_idx_pair,
                                        pvec_start_list=pvec_list_cp,
                                        N_trials_gradient=N_trials_gradient,
                                        max_splits=N_max_splits,
                                        max_on=max_on,
                                        split_all=True)
                            del minimize_initial_worker_id_dict[worker_id]
                            failed = False
                        except:
                            if _VERBOSE:
                                import traceback
                                print(traceback.format_exc())
                            failed = True

                    if self.verbose:
                        print(
                            "Obtaining votes and submitting parameter minimizations.")
                    for key in split_worker_id_dict:
                        mngr_idx, sys_idx_pair = key
                        votes_split_list = self.get_votes(
                            worker_id_list = split_worker_id_dict[mngr_idx,sys_idx_pair][0],
                            low_to_high = True,
                            abs_grad_score = False,
                            norm_cutoff = 1.e-2,
                            keep_N_best = keep_N_best)
                        self.bitvec_dict[mngr_idx,sys_idx_pair] = dict()
                        for ast in votes_split_list:
                            b_list = split_worker_id_dict[mngr_idx,sys_idx_pair][1][ast]
                            self.bitvec_dict[mngr_idx,sys_idx_pair][ast] = b_list
                        self.minscore_worker_id_dict[mngr_idx,sys_idx_pair] = self.get_min_scores(
                            mngr_idx_main=mngr_idx,
                            bitvec_dict = self.bitvec_dict[mngr_idx,sys_idx_pair],
                            system_idx_list=sys_idx_pair,
                            votes_split_list=votes_split_list,
                            local_targets=use_local_targets)
                        if self.verbose:
                            print(
                                f"For mngr {mngr_idx} and systems {sys_idx_pair}:\n"
                                f"Found {len(votes_split_list)} candidate split solutions ...\n")

                    # N_sys_per_batch ---> N_systems_validation
                    # N_batches       ---> N_iter_validation
                    self.sys_idx_list_validation = self.get_random_system_idx_list(
                            N_systems_validation, N_iter_validation, cluster_systems)
                    self._validation_generator_smiles_list = copy.deepcopy(
                        self._generator_smiles_list)
                    self.set_targetcomputer(
                        self.sys_idx_list_validation,
                        error_factor=(1.-error_decrease_factor)**iteration_idx)
                    for sys_idx_validation in self.sys_idx_list_validation:
                        for mngr_idx in range(self.N_mngr):
                            self.best_ast_dict[mngr_idx, sys_idx_validation]  = None
                            self.best_pvec_dict[mngr_idx, sys_idx_validation] = None
                            self.best_aic_dict[mngr_idx, sys_idx_validation]  = None

                    for key in self.bitvec_dict:
                        mngr_idx, sys_idx_pair = key
                        b_list = self.bitvec_dict[mngr_idx, sys_idx_pair]
                        if (mngr_idx, sys_idx_pair) in self.selection_worker_id_dict:
                            del self.selection_worker_id_dict[mngr_idx, sys_idx_pair]
                        for sys_idx_validation in self.sys_idx_list_validation:
                            old_pvec_list, old_bitvec_type_list = self.generate_parameter_vectors(
                                    system_idx_list=sys_idx_validation)
                            bitvec_alloc_dict_list = list()
                            for _mngr_idx in range(self.N_mngr):
                                _, bitvec_alloc_dict = self.generate_bitsmartsmanager(
                                        _mngr_idx, sys_idx_validation)
                                bitvec_alloc_dict_list.append(bitvec_alloc_dict)
                            worker_id = validate_FF.remote(
                                mngr_idx_main = mngr_idx,
                                pvec_list = old_pvec_list,
                                targetcomputer = self.targetcomputer_id_dict[sys_idx_validation],
                                bitvec_dict = b_list,
                                bitvec_alloc_dict_list = bitvec_alloc_dict_list,
                                bitvec_type_list_list = old_bitvec_type_list,
                                worker_id_dict = self.minscore_worker_id_dict[mngr_idx,sys_idx_pair],
                                parm_penalty = self.parm_penalty_split,
                                verbose = self.verbose)
                            if (mngr_idx, sys_idx_pair) not in self.selection_worker_id_dict:
                                self.selection_worker_id_dict[mngr_idx, sys_idx_pair] = dict()
                            self.selection_worker_id_dict[mngr_idx, sys_idx_pair][worker_id] = sys_idx_validation

                        del split_worker_id_dict[mngr_idx,sys_idx_pair]

                    with open(f"{self.name}-MAIN-{iteration_idx+output_offset}-SPLIT-{self.split_iteration_idx}-SELECTION.pickle", "wb") as fopen:
                        pickle.dump(
                            self,
                            fopen)
                    
                if restart:
                    _minscore_worker_id_dict = dict()
                    ### If we want to restart, first make sure that we re-run all
                    ### the left-over minimization runs
                    for mngr_idx, sys_idx_pair in self.minscore_worker_id_dict:
                        pvec_all, bitvec_type_list = self.generate_parameter_vectors(
                            system_idx_list=sys_idx_pair)
                        pvec_cp = pvec_all[mngr_idx].vector_k[:].copy()
                        allocations_cp = copy.deepcopy(pvec_all[mngr_idx].allocations)
                        system_list_id = ray.put(
                            [self.system_list[sys_idx] for sys_idx in sys_idx_pair])

                        _minscore_worker_id_dict[mngr_idx, sys_idx_pair] = dict()
                        for ast in self.bitvec_dict[mngr_idx,sys_idx_pair]:
                            allocations, selection, type_ = ast
                            type_i, type_j = type_
                            bitvec_type_list_cp = copy.deepcopy(bitvec_type_list)
                            bitvec_type_list_cp[mngr_idx].insert(
                                    type_j, self.bitvec_dict[mngr_idx,sys_idx_pair][ast][0])

                            pvec_all[mngr_idx].duplicate(type_i, type_j)
                            pvec_all[mngr_idx].allocations[:] = allocations[:]
                            pvec_all[mngr_idx].apply_changes()

                            pvec_all_cp = [pvec.copy() for pvec in pvec_all]

                            worker_id = minimize_FF.remote(
                                    system_list = system_list_id,
                                    targetcomputer = self.targetcomputer_id_dict[sys_idx_pair],
                                    pvec_list = pvec_all_cp,
                                    frozen_types_list = self.frozen_types_list,
                                    bitvec_type_list = bitvec_type_list_cp,
                                    bounds_list = self.bounds_list,
                                    parm_penalty = self.parm_penalty_split,
                                    local_targets = use_local_targets,
                                    bounds_penalty = self._bounds_penalty_list_work,
                                    #bounds_penalty = [1. for _ in range(self.N_mngr)],
                                    N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                                    use_global_opt = _USE_GLOBAL_OPT,
                                    verbose = self.verbose)
                            _minscore_worker_id_dict[mngr_idx, sys_idx_pair][worker_id] = ast

                            pvec_all[mngr_idx].reset(
                                    pvec_cp, allocations_cp)

                    self.minscore_worker_id_dict = _minscore_worker_id_dict

                    self._set_system_list(
                        self._validation_generator_smiles_list)
                    self.sys_idx_list_validation = self._reorder_system_idxs(
                            self.sys_idx_list_validation)
                    self.set_targetcomputer(
                        self.sys_idx_list_validation,
                        error_factor = (1.-error_decrease_factor)**iteration_idx)
                    for mngr_idx, sys_idx_pair in self.minscore_worker_id_dict:
                        b_list = self.bitvec_dict[mngr_idx, sys_idx_pair]
                        if (mngr_idx, sys_idx_pair) in self.selection_worker_id_dict:
                            del self.selection_worker_id_dict[mngr_idx, sys_idx_pair]
                        for sys_idx_validation in self.sys_idx_list_validation:
                            old_pvec_list, old_bitvec_type_list = self.generate_parameter_vectors(
                                    system_idx_list=sys_idx_validation)
                            bitvec_alloc_dict_list = list()
                            for _mngr_idx in range(self.N_mngr):
                                _, bitvec_alloc_dict = self.generate_bitsmartsmanager(
                                        _mngr_idx, sys_idx_validation)
                                bitvec_alloc_dict_list.append(bitvec_alloc_dict)
                            worker_id = validate_FF.remote(
                                mngr_idx_main = mngr_idx,
                                pvec_list = old_pvec_list,
                                targetcomputer = self.targetcomputer_id_dict[sys_idx_validation],
                                bitvec_dict = b_list,
                                bitvec_alloc_dict_list = bitvec_alloc_dict_list,
                                bitvec_type_list_list = old_bitvec_type_list,
                                worker_id_dict = self.minscore_worker_id_dict[mngr_idx,sys_idx_pair],
                                parm_penalty = self.parm_penalty_split,
                                verbose = self.verbose)
                            if (mngr_idx, sys_idx_pair) not in self.selection_worker_id_dict:
                                self.selection_worker_id_dict[mngr_idx, sys_idx_pair] = dict()
                            self.selection_worker_id_dict[mngr_idx, sys_idx_pair][worker_id] = sys_idx_validation

                ### ============= ###
                ### END SPLITTING ###/
                ### ============= ###

                ### ================ ###
                ### FIND BEST SPLITS ###
                ### ================ ###

                if self.verbose:
                    print(
                        "Finding best parameters.")
                if not restart:
                    self.system_idx_list_batch = self.system_idx_list_batch[::-1]

                worker_id_dict = dict()
                for mngr_idx, sys_idx_pair in self.selection_worker_id_dict:
                    for worker_id in self.selection_worker_id_dict[mngr_idx, sys_idx_pair]:
                        worker_id_dict[worker_id] = mngr_idx, sys_idx_pair

                selection_worker_id_list = list(worker_id_dict.keys())
                while selection_worker_id_list:
                    [worker_id], selection_worker_id_list = ray.wait(
                        selection_worker_id_list)
                    mngr_idx, sys_idx_pair = worker_id_dict[worker_id]
                    sys_idx_validation = self.selection_worker_id_dict[mngr_idx, sys_idx_pair][worker_id]
                    key = mngr_idx, sys_idx_validation
                    if key not in self.best_aic_dict:
                        continue
                    best_AIC = self.best_aic_dict[key]
                    try:
                        ### pvec_list should only contain vector_k at this point
                        _, pvec_list, best_bitvec_type_list, new_AIC, new_ast = ray.get(worker_id)
                        failed = False
                    except:
                        if _VERBOSE:
                            import traceback
                            print(traceback.format_exc()) 
                        failed = True
                    if not failed:
                        if isinstance(best_AIC, type(None)):
                            found_improvement_mngr = True
                        elif isinstance(new_AIC, type(None)):
                            found_improvement_mngr = False
                        else:
                            found_improvement_mngr = new_AIC < best_AIC

                        if found_improvement_mngr:
                            self.best_aic_dict[key]  = new_AIC
                            self.best_ast_dict[key]  = new_ast, sys_idx_pair
                            self.best_pvec_dict[key] = [pvec[:].copy() for pvec in pvec_list], best_bitvec_type_list

                    del self.selection_worker_id_dict[mngr_idx, sys_idx_pair][worker_id]
                    to_delete = list()
                    for worker_id in self.minscore_worker_id_dict[mngr_idx,sys_idx_pair]:
                        ast = self.minscore_worker_id_dict[mngr_idx,sys_idx_pair][worker_id]
                        if new_ast == ast:
                            to_delete.append(worker_id)
                    for worker_id in to_delete:
                        del self.minscore_worker_id_dict[mngr_idx,sys_idx_pair][worker_id]

                    if len(self.selection_worker_id_dict[mngr_idx, sys_idx_pair]) == 0:
                        del self.minscore_worker_id_dict[mngr_idx,sys_idx_pair]
                        del self.bitvec_dict[mngr_idx,sys_idx_pair]
                        del self.selection_worker_id_dict[mngr_idx,sys_idx_pair]

                        self.accepted_counter += 1

                    with open(f"{self.name}-MAIN-{iteration_idx+output_offset}-SPLIT-{self.split_iteration_idx}-ACCEPTED-{self.accepted_counter}.pickle", "wb") as fopen:
                        pickle.dump(
                                self,
                                fopen)

                del worker_id_dict

                ### =============================== ###
                ### FINAL VALIDATION AND REFINEMENT ###
                ### =============================== ###

                ### Now figure out the best ast
                best_ast_vote_dict  = dict()
                best_pvec_vote_dict = dict()
                for mngr_idx in range(self.N_mngr):
                    best_ast_vote_dict[mngr_idx]  = dict()
                    best_pvec_vote_dict[mngr_idx] = dict()
                
                ### Sometimes no value is found for a given 
                ### combination of mngr_idx, sys_idx_validation
                ### because minimizations fail or other things happen.
                ### Also, sometimes the number of types in the bitvector
                ### list and the number of physical parameter values
                ### don't match. We want to remove these cases.
                import warnings
                key_list = list(self.best_ast_dict.keys())
                for key in key_list:
                    value = self.best_ast_dict[key]
                    if isinstance(value, type(None)):
                        warnings.warn(
                            f"Deleting solution of MNGR {key[0]} and SYSTEMS {key[1]}")
                        del self.best_ast_dict[key]
                        del self.best_aic_dict[key]
                        del self.best_pvec_dict[key]
                        continue
                    elif isinstance(value[0], type(None)):
                        warnings.warn(
                            f"Deleting solution of MNGR {key[0]} and SYSTEMS {key[1]}")
                        del self.best_ast_dict[key]
                        del self.best_aic_dict[key]
                        del self.best_pvec_dict[key]
                        continue
                    _old_pvec , _ = self.generate_parameter_vectors(
                            system_idx_list=self.sys_idx_list_validation[0])
                    _pvec_list, _bitvec_type_list = self.best_pvec_dict[key]
                    for mngr_idx in range(self.N_mngr):
                        N_parms = len(_pvec_list[mngr_idx][:])/_old_pvec[mngr_idx].parameters_per_force_group
                        N_parms = int(N_parms)
                        if N_parms != len(_bitvec_type_list[mngr_idx]):
                            del self.best_ast_dict[key]
                            del self.best_aic_dict[key]
                            del self.best_pvec_dict[key]
                            break

                for mngr_idx, sys_idx_validation in self.best_ast_dict:
                    best_ast, sys_idx_pair = self.best_ast_dict[mngr_idx, sys_idx_validation]
                    if (best_ast, sys_idx_pair) in best_ast_vote_dict[mngr_idx]:
                        best_ast_vote_dict[mngr_idx][best_ast, sys_idx_pair] += 1
                    else:
                        best_ast_vote_dict[mngr_idx][best_ast, sys_idx_pair] = 1
                        best_pvec_vote_dict[mngr_idx][best_ast, sys_idx_pair] = self.best_pvec_dict[mngr_idx, sys_idx_validation]

                if self.verbose:
                    print(
                        "VALIDATION I: AIC of SMARTS patterns on same parameter set.")
                    for mngr_idx, sys_idx_validation in self.best_ast_dict:
                        aic = self.best_aic_dict[mngr_idx, sys_idx_validation]
                        new_ast, sys_idx_pair = self.best_ast_dict[mngr_idx, sys_idx_validation]
                        print(
                            f"MANAGER {mngr_idx} / VALIDATION SYSTEM {sys_idx_validation} / TRAINING SYSTEM {sys_idx_pair} / AIC {aic}")
                        try:
                            bsm, _ = self.generate_bitsmartsmanager(
                                mngr_idx, self.sys_idx_list_validation[0])
                            _, bitvec_type_list = best_pvec_vote_dict[mngr_idx][new_ast, sys_idx_pair]
                            sma_list = list()
                            for b in bitvec_type_list[mngr_idx]:
                                sma = bsm.bitvector_to_smarts(b)
                                sma_list.append(sma)
                            print(", ".join(sma_list))
                            print()
                        except Exception as e:
                            print(e)

                pvec_list_query = list()
                bitvec_type_list_query = list()
                if self.verbose:
                    print(
                        "VALIDATION I: VOTES of SMARTS patterns on same parameter set.")
                for mngr_idx in range(self.N_mngr):
                    if mngr_idx not in best_ast_vote_dict:
                        continue
                    best_ast_sysidx_list = sorted(
                            best_ast_vote_dict[mngr_idx],
                        key=best_ast_vote_dict[mngr_idx].get,
                        reverse=True)
                    if len(best_ast_sysidx_list) > MAX_VALIDATE:
                        best_ast_sysidx_list = best_ast_sysidx_list[:MAX_VALIDATE]
                    for best_ast, sys_idx_pair in best_ast_sysidx_list:
                        pvec_list, bitvec_type_list = best_pvec_vote_dict[mngr_idx][best_ast, sys_idx_pair]
                        pvec_list_query.append(pvec_list)
                        bitvec_type_list_query.append(bitvec_type_list)
                    if self.verbose:
                        for best_ast, sys_idx_pair in best_ast_sysidx_list:
                            N_votes = best_ast_vote_dict[mngr_idx][best_ast, sys_idx_pair]
                            print(
                                f"MANAGER {mngr_idx} / TRAINING SYSTEM {sys_idx_pair} / N VOTES {N_votes}")
                            ### the value of `sys_idx_validation` is irrelevant at this point,
                            ### as long as it has been used before.
                            try:
                                bsm, _ = self.generate_bitsmartsmanager(
                                    mngr_idx, self.sys_idx_list_validation[0])
                                _, bitvec_type_list = best_pvec_vote_dict[mngr_idx][best_ast, sys_idx_pair]
                                sma_list = list()
                                for b in bitvec_type_list[mngr_idx]:
                                    sma = bsm.bitvector_to_smarts(b)
                                    sma_list.append(sma)
                                print(", ".join(sma_list))
                                print()
                            except Exception as e:
                                print(e)

                if self.verbose:
                    print()
                    print(
                        "VALIDATION II: VOTES of SMARTS pattern combinations.")
                ### Figure out if we can combine the individual solutions
                worker_id_dict = dict()
                for sys_idx_validation in self.sys_idx_list_validation:
                    old_pvec_list, old_bitvec_type_list = self.generate_parameter_vectors(
                            system_idx_list=sys_idx_validation)
                    bitvec_list_alloc_dict_list = list()
                    for mngr_idx in range(self.N_mngr):
                        _, bitvec_list_alloc_dict = self.generate_bitsmartsmanager(
                            mngr_idx,
                            sys_idx_validation)
                        bitvec_list_alloc_dict_list.append(
                            bitvec_list_alloc_dict)

                    worker_id = likelihood_combine_pvec.remote(
                        old_pvec_list, pvec_list_query, bitvec_type_list_query,
                        self.parm_penalty_split, bitvec_list_alloc_dict_list, 
                        self.targetcomputer_id_dict[sys_idx_validation],
                        self.verbose)
                    worker_id_dict[worker_id] = sys_idx_validation

                best_likelihood_vote_dict  = dict()
                worker_id_list = list(worker_id_dict.keys())
                aic_min_dict = dict()
                while worker_id_list:
                    [worker_id], worker_id_list = ray.wait(worker_id_list)
                    ### The results_dict contains aic values
                    results_dict = ray.get(worker_id)

                    sys_idx_validation = worker_id_dict[worker_id]
                    if not results_dict:
                        continue
                    best_selection = min(results_dict, key=results_dict.get)
                    if best_selection in best_likelihood_vote_dict:
                        best_likelihood_vote_dict[best_selection] += 1
                        aic_min_dict[best_selection] = min(aic_min_dict[best_selection], results_dict[best_selection])
                    else:
                        best_likelihood_vote_dict[best_selection] = 1
                        aic_min_dict[best_selection] = results_dict[best_selection]

                if self.verbose:
                    for best_selection in best_likelihood_vote_dict:
                        N_votes    = best_likelihood_vote_dict[best_selection]
                        output_str = list()
                        for mngr_idx in range(self.N_mngr):
                            output_str.append(
                                f"MANAGER {mngr_idx} CANDIDATE {best_selection[mngr_idx]}")
                            try:
                                bsm, _ = self.generate_bitsmartsmanager(
                                    mngr_idx,
                                    self.sys_idx_list_validation[0])
                                candidate_idx = best_selection[mngr_idx]
                                bitvec_type_list = bitvec_type_list_query[candidate_idx]
                                sma_list = list()
                                for b in bitvec_type_list[mngr_idx]:
                                    sma = bsm.bitvector_to_smarts(b)
                                    sma_list.append(sma)
                                output_str.append(", ".join(sma_list))
                            except Exception as e:
                                print(e)

                        print(
                            " / ".join(output_str) + f" : N VOTES {N_votes} AIC MIN {aic_min_dict[best_selection]}")
                        print()

                if best_likelihood_vote_dict:
                    best_selection = max(
                        best_likelihood_vote_dict, key=best_likelihood_vote_dict.get)
                else:
                    best_selection = [None] * self.N_mngr
                if self.verbose:
                    old_pvec_list, _ = self.generate_parameter_vectors(
                            system_idx_list=self.sys_idx_list_validation[0])
                for mngr_idx in range(self.N_mngr):
                    candidate_idx = best_selection[mngr_idx]
                    found_update  = not isinstance(candidate_idx, type(None))
                    if self.verbose:
                        if found_update:
                            print(
                                f"Updating manager {mngr_idx} from manager optimization {candidate_idx}")
                            print(
                                f"Updated parameters for mngr {mngr_idx}:")
                        else:
                            print(
                                f"Not updating mngr {mngr_idx}")
                    if found_update:
                        pvec_list = pvec_list_query[candidate_idx]
                        bitvec_type_list = bitvec_type_list_query[candidate_idx]
                        self.update_best(
                            mngr_idx,
                            pvec_list[mngr_idx],
                            bitvec_type_list[mngr_idx])
                    if self.verbose:
                        old_pvec = old_pvec_list[mngr_idx]
                        if not found_update:
                            pvec_list = [pvec.vector_k for pvec in old_pvec_list]
                        try:
                            bsm, _ = self.generate_bitsmartsmanager(
                                mngr_idx, self.sys_idx_list_validation[0])
                        except Exception as e:
                            print(e)
                            continue

                        for idx, b in enumerate(bitvec_type_list[mngr_idx]):
                            try:
                                sma   = bsm.bitvector_to_smarts(b)
                            except:
                                sma   = "???"
                            start = old_pvec.parameters_per_force_group * idx
                            stop  = start + old_pvec.parameters_per_force_group
                            vec   = pvec_list[mngr_idx][start:stop]
                            vec_str = ",".join([str(v) for v in vec])
                            print(
                                f"{idx} : {vec_str} {sma}")
                if self.verbose:
                    print(
                        "SYSTEM BENCHMARK FINAL")
                    print(
                        "======================")
                    from .tools import benchmark_systems
                    sys_idx_list_validation_all = list()
                    for sys_idx_list in self.sys_idx_list_validation:
                        sys_idx_list_validation_all.extend(
                                list(sys_idx_list))
                    try:
                        _pvec_list, _ = self.generate_parameter_vectors(
                            system_idx_list = sys_idx_list_validation_all)
                        _pvec = _pvec_list[0]
                        benchmark_systems(
                            _pvec.parameter_manager.system_list)
                    except Exception as e:
                        print(e)

                    from .tools import get_rmsd
                    try:
                        rmsd_dict = get_rmsd(
                                _pvec.parameter_manager.system_list,
                                self.system_manager_loader.optdataset_dict)
                        rmsd_list = list()
                        for smi in rmsd_dict:
                            rmsd_list.extend(
                                    rmsd_dict[smi])
                        rmsd_list = np.array(rmsd_list)
                        _rmsd     = np.sqrt(np.mean(rmsd_list**2))
                        print(
                                "GLOBAL ESTIMATE FOR POSITION RMSD")
                        print(
                                "=================================")
                        print(
                                f"{_rmsd:4.2f}")
                        print()
                    except:
                        print(
                            f"Could not compute RMSD.")

                self.split_iteration_idx += 1

                ### Remove tempory data
                ### self.split_iteration_idx must not be reset
                self.system_idx_list_batch = []
                self.minscore_worker_id_dict = dict()
                self.selection_worker_id_dict = dict()
                self.best_ast_dict = dict()
                self.best_pvec_dict = dict()
                self.best_aic_dict = dict()
                self.bitvec_dict = dict()
                self.accepted_counter = 0

                if restart:
                    restart = False

                ### ================= ###
                ### END SPLIT ATTEMPT ###
                ### ================= ###

                self.clear_cache()
                self.system_manager_loader.clear_cache()

                for worker_id in self._ray_cleanup:
                    try:
                        ray.kill(worker_id)
                    except:
                        continue
                del self._ray_cleanup[:]
                gc.collect()
            
            ### Remove tempory data
            if restart:
                restart = False
            self.system_idx_list_batch = []
            self.minscore_worker_id_dict = dict()
            self.selection_worker_id_dict = dict()
            self.best_ast_dict = dict()
            self.best_pvec_dict = dict()
            self.best_aic_dict = dict()
            self.bitvec_dict = dict()
            # self.split_iteration_idx must be reset here
            self.split_iteration_idx = 0
            self.accepted_counter = 0
            for worker_id in self._ray_cleanup:
                try:
                    ray.kill(worker_id)
                except:
                    continue
            del self._ray_cleanup[:]
            gc.collect()


            ### ================== ###
            ### GARBAGE COLLECTION ###
            ### ================== ###
            self.garbage_collection(
                N_systems_validation, N_iter_validation)

            with open(f"{self.name}-MAIN-{iteration_idx+output_offset}-GARBAGE_COLLECTION.pickle", "wb") as fopen:
                pickle.dump(
                    self,
                    fopen
                )

            ### ============== ###
            ### END OUTER LOOP ###
            ### ============== ###

            self.save_traj(parm_penalty=1.)

            if self.verbose:
                print("")

            with open(f"{self.name}-MAIN-{iteration_idx+output_offset}.pickle", "wb") as fopen:
                pickle.dump(
                    self,
                    fopen
                )

            for worker_id in self._ray_cleanup:
                try:
                    ray.kill(worker_id)
                except:
                    continue
            del self._ray_cleanup[:]
            gc.collect()
        ### END LOOP OVER ITERATIONS

