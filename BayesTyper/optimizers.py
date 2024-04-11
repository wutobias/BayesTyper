import numpy as np
from scipy import stats
import copy

from .kernels import LogJumpKernel
from .vectors import ForceFieldParameterVector
from .vectors import SmartsForceFieldParameterVector
from .likelihoods import LogGaussianLikelihood
from .likelihoods import LikelihoodVectorized
from .bitvector_typing import BitSmartsManager

from .constants import (_LENGTH,
                        _ANGLE,
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
                        _EPSILON_GS
                        )
from .ray_tools import retrieve_failed_workers

import ray


@ray.remote
def generate_parameter_manager(sys_list, parm_mngr):

    import copy
    parm_mngr_cp = copy.deepcopy(parm_mngr)
    for sys in sys_list:
        parm_mngr_cp.add_system(sys)
    return parm_mngr_cp

@ray.remote
def _test_logL(
    _pvec,
    _targetcomputer):

    import time

    start_time = time.time()

    logL_list = batch_likelihood_typing(
        _pvec,
        _targetcomputer,
        [_pvec.allocations[:].tolist()],
        rebuild_from_systems=True
        )

    time_delta = time.time() - start_time
    return time_delta


def batch_likelihood_typing(
    pvec,
    targetcomputer,
    typing_list,
    rebuild_from_systems=True
    ):

    import copy
    import numpy as np

    if rebuild_from_systems:
        pvec_cp = pvec.copy(include_systems=True, rebuild_to_old_systems=False)
    else:
        pvec_cp = pvec

    N_queries = len(typing_list)

    openmm_system_list = list()
    for sys in pvec_cp.parameter_manager.system_list:
        openmm_system_list.append(
            {sys.name : [sys.openmm_system]}
            )

    init_alloc = copy.deepcopy(pvec_cp.allocations[:])
    worker_id_dict = dict()
    cache_dict = dict()
    cached_idxs = list()
    for idx in range(N_queries):
        typing = list(typing_list[idx])
        typing_tuple = tuple(typing)
        if typing_tuple in cache_dict:
            cached_idxs.append(idx)
        else:
            cache_dict[typing_tuple] = idx
            pvec_cp.allocations[:] = typing[:]
            pvec_cp.apply_changes()
            for ommdict_idx, ommdict in enumerate(openmm_system_list):
                worker_id = targetcomputer(
                    copy.deepcopy(ommdict), 
                    False
                    )
                worker_id_dict[worker_id] = (idx, ommdict_idx)
    
    logL_list = np.zeros(N_queries, dtype=float)
    worker_id_list = list(worker_id_dict.keys())
    while worker_id_list:
        worker_id, worker_id_list = ray.wait(
            worker_id_list, timeout=_TIMEOUT)
        failed = len(worker_id) == 0
        if not failed:
            try:
                _logP_likelihood = ray.get(
                    worker_id[0],
                    timeout=_TIMEOUT)
            except:
                failed = True
            if not failed:
                idx, ommdict_idx = worker_id_dict[worker_id[0]]
                logL_list[idx]  += _logP_likelihood
                del worker_id_dict[worker_id[0]]
        if failed:
            if len(worker_id) > 0:
                if worker_id[0] not in worker_id_list:
                    worker_id_list.append(worker_id[0])
            resubmit_list = retrieve_failed_workers(worker_id_list)
            for worker_id in resubmit_list:
                ray.cancel(worker_id, force=True)
                idx, ommdict_idx = worker_id_dict[worker_id]
                del worker_id_dict[worker_id]
                typing = list(typing_list[idx])
                typing_tuple = tuple(typing)
                pvec_cp.allocations[:] = typing[:]
                pvec_cp.apply_changes()
                ommdict = openmm_system_list[ommdict_idx]
                worker_id = targetcomputer(
                    copy.deepcopy(ommdict), 
                    False
                    )
                worker_id_dict[worker_id] = (idx, ommdict_idx)
            worker_id_list = list(worker_id_dict.keys())

    for idx in cached_idxs:
        typing = list(typing_list[idx])
        typing_tuple = tuple(typing)
        idx_cache = cache_dict[typing_tuple]
        logL_list[idx] = logL_list[idx_cache]

    if not rebuild_from_systems:
        pvec_cp.allocations[:] = init_alloc
        pvec_cp.apply_changes()

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


@ray.remote
def get_gradient_scores(
    ff_parameter_vector,
    targetcomputer,
    type_i, # type to be split
    selection_i = None,
    k_values_ij = None,
    grad_diff = _EPSILON_GS,
    N_trials = 5,
    N_sys_per_likelihood_batch = 4,
    ):

    ff_parameter_vector_cp = ff_parameter_vector.copy(
        include_systems = True,
        rebuild_to_old_systems = False
        )

    N_types    = ff_parameter_vector_cp.force_group_count

    N_parms    = ff_parameter_vector_cp.parameters_per_force_group
    N_typables = ff_parameter_vector_cp.allocations.size

    type_j = type_i + 1

    ff_parameter_vector_cp.duplicate(type_i)
    ff_parameter_vector_cp.swap_types(N_types, type_j)

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
                )

            grad_i = grad[first_parm_i:last_parm_i]
            grad_j = grad[first_parm_j:last_parm_j]

            norm_i = np.linalg.norm(grad_i)
            norm_j = np.linalg.norm(grad_j)

            if np.isnan(norm_i) or np.isinf(norm_i):
                continue
            if np.isnan(norm_j) or np.isinf(norm_j):
                continue

            if norm_i > 1.e-7:
                grad_i /= norm_i
            if norm_j > 1.e-7:
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


@ray.remote
def minimize_FF(
    system_list,
    targetcomputer,
    pvec_list,
    bitvec_type_list,
    bounds_list,
    parm_penalty,
    pvec_idx_min=None,
    grad_diff=_EPSILON,
    N_sys_per_likelihood_batch=4,
    bounds_penalty=10.,
    use_scipy=True,
    use_DE=False,
    verbose=False,
    get_timing=False):

    if get_timing:
        import time
        time_start = time.time()

    from openmm import unit
    from . import priors
    import numpy as np
    import copy

    pvec_list_cp   = copy.deepcopy(pvec_list)
    system_list_cp = copy.deepcopy(system_list)

    pvec_min_list  = list()
    N_parms_all    = 0
    if pvec_idx_min == None:
        pvec_idx_list = range(len(pvec_list_cp))
    elif isinstance(pvec_idx_min, list):
        pvec_idx_list = pvec_idx_min
    elif isinstance(pvec_idx_min, np.ndarray):
        pvec_idx_list = pvec_idx_min.tolist()
    else:
        pvec_idx_list = [pvec_idx_min]
    for pvec in pvec_list_cp:
        N_parms_all += pvec.size

    prior_idx_list = list()
    parm_idx_list  = list()
    N_parms = 0
    for pvec_idx in pvec_idx_list:

        pvec = pvec_list_cp[pvec_idx]
        pvec_min_list.append(pvec)

        ### Look for dead parameters
        hist = pvec.force_group_histogram
        for type_i in range(pvec.force_group_count):
            if hist[type_i] > 0:
                for idx in range(pvec.parameters_per_force_group):
                    parm_idx_list.append(
                        type_i * pvec.parameters_per_force_group + idx + N_parms)

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
        N = pvec.force_group_count
        M = pvec.parameters_per_force_group
        if isinstance(bounds, priors.BaseBounds):
            bounds.apply_pvec(pvec)
            N = pvec.force_group_count
            M = pvec.parameters_per_force_group
            if isinstance(bounds, (priors.AngleBounds, priors.BondBounds)):
                for i in range(0,N*M,2):
                    ### We only want to consider the
                    ### equilibrium bond lengths and angles
                    prior_idx_list.append(i+1+N_parms)
            else:
                for i in range(0,N*M):
                    ### We want to consider all Amplitudes
                    prior_idx_list.append(i+N_parms)
                    
        N_parms += pvec.size

    likelihood_func = LikelihoodVectorized(
        pvec_min_list,
        targetcomputer,
        N_sys_per_batch = N_sys_per_likelihood_batch
        )

    x0 = copy.deepcopy(likelihood_func.pvec[:])
    x0_ref = copy.deepcopy(likelihood_func.pvec[:])

    def penalty(x):

        penalty_val = np.sum(x[prior_idx_list]**2)
        penalty_val *= bounds_penalty
        return penalty_val

    def fun(x):

        likelihood = likelihood_func(x, parm_idx_list=parm_idx_list)
        AIC_score  = 2. * N_parms_all * parm_penalty - 2. * likelihood
        
        return AIC_score

    def grad_penalty(x):

        grad = np.zeros_like(x)
        grad[prior_idx_list] = bounds_penalty * 2. * x[prior_idx_list]

        return grad

    def grad(x):

        _grad = likelihood_func.grad(x, 
            grad_diff=grad_diff, parm_idx_list=parm_idx_list,
            use_jac=False)
        ### Multiply by -2. due to AIC definition
        _grad *= -2.

        return _grad

    if x0.size == 0:
        if get_timing:
            return fun(x0), pvec_list_cp, bitvec_type_list, time.time() - time_start
        else:
            return fun(x0), pvec_list_cp, bitvec_type_list

    if use_scipy:
        from scipy import optimize
        _fun = lambda x: fun(x)+penalty(x)
        _grad = lambda x: grad(x)+grad_penalty(x)

        if verbose:
            print(
                "Initial func value:",
                _fun(x0),
                )
                
        if use_DE:
            result = optimize.differential_evolution(
               _fun,
               [(-10,10) for _ in x0],
               maxiter=100,)
        
        else:
            result = optimize.minimize(
                _fun, 
                x0, 
                jac = _grad, 
                #method = "Newton-CG",
                method = "BFGS")
        
        best_x = result.x
        likelihood_func.apply_changes(best_x)
        best_f  = _fun(best_x)

        if verbose:
            print(
                "Final func value:",
                best_f,
                )

        if get_timing:
            return best_f, pvec_list_cp, bitvec_type_list, time.time() - time_start
        else:
            return best_f, pvec_list_cp, bitvec_type_list

    ### Step length
    alpha_incr = 1.e-2
    ### Some stopping criteria
    ### Note 08-12-2022: Decreased grad_tol and fun_tol from 1.e-2 to 1.e-4.
    ###                  After adding the Jacobian to the grad function in likelihood_func
    ###                  it seems like the force constants don't really get optimized
    ###                  anymore. This could be an issue with convergence criteria.
    grad_tol   = 1.e-2
    fun_tol    = 1.e-2
    fun_tol_line_search = 1.e-2
    max_steps_without_improvement = 5
    max_line_search_steps = 10
    max_total_steps = 100

    ### This is the main minimizer algorithm.
    grad_0 = grad(x0) + grad_penalty(x0)
    grad_0_norm = np.linalg.norm(grad_0)
    if grad_0_norm > 0.:
        grad_0 /= grad_0_norm
    best_f = fun(x0) + penalty(x0)
    best_x = x0
    step_count = 0
    criteria = 0
    criteria_msg = ""
    fun_evals = 0
    grad_evals = 0
    steps_without_improvement = 0

    if verbose:
        print(
            "Initial func value:",
            best_f,
            )
    while True:
        alpha  = alpha_incr
        improvement = 0.
        best_f0 = best_f
        do_line_search = True
        line_search_steps = 0
        while do_line_search:
            fun_evals += 1
            xk  = x0 - grad_0 * alpha
            f_k = fun(xk) + penalty(xk)
            line_improvement = f_k - best_f
            ### If we have done less than `max_line_search_steps`
            ### iterations, just continue.
            if line_search_steps < max_line_search_steps:
                if line_improvement < 0.:
                    best_f = f_k
                    best_x = xk
                    improvement = abs(best_f - best_f0)
            ### If we have already done more than `max_line_search_steps`
            ### and the objective function still decreases, continue.
            else:
                if line_improvement < 0. and abs(line_improvement) > fun_tol_line_search:
                    best_f = f_k
                    best_x = xk
                    improvement = abs(best_f - best_f0)
                else:
                    do_line_search = False
            alpha += alpha_incr
            line_search_steps += 1
        x0     = best_x
        grad_0 = grad(x0) + grad_penalty(x0)
        grad_0_norm = np.linalg.norm(grad_0)
        if grad_0_norm > 0.:
            grad_0 /= grad_0_norm
        step_count += 1
        grad_evals += 1
        if improvement < fun_tol:
            steps_without_improvement += 1
        else:
            steps_without_improvement = 0

        if grad_0_norm < grad_tol:
            criteria += 1
            criteria_msg += "Stopping due to norm. "
        if improvement < fun_tol:
            criteria += 1
            criteria_msg += "Stopping due to fun value. "
        if steps_without_improvement > max_steps_without_improvement:
            criteria += 1
            criteria_msg += "Stopping due to max steps without improvement. "
        if step_count > max_total_steps:
            criteria += 1
            criteria_msg += "Stopping due to total max steps. "
        if criteria > 1:
            if verbose:
                print(
                    criteria_msg, "# Fun evals:", fun_evals, "# Grad evals:", grad_evals
                    )
            break
        else:
            criteria = 0
            criteria_msg = ""

    likelihood_func.apply_changes(best_x)
    ### Subtract penalty out
    best_f -= penalty(best_x)

    if verbose:
        print(
            "Final func value:",
            best_f,
            )

    if get_timing:
        return best_f, pvec_list_cp, bitvec_type_list, time.time() - time_start
    else:
        return best_f, pvec_list_cp, bitvec_type_list


@ray.remote
def set_parameters_remote(
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

    MAX_AIC = 9999999999999999.
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
            rebuild_from_systems=False
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

    [best_AIC] = _calculate_AIC(
        pvec_list_cp,
        0,
        [pvec_list_cp[0].allocations[:].tolist()],
        targetcomputer
        )

    if verbose:
        print(
            "Initial best AIC:", best_AIC
            )
        print(
            f"Checking {len(worker_id_dict)} solutions..."
            )
    best_pvec_list = [pvec.copy() for pvec in pvec_list_cp]
    best_ast       = None
    best_bitvec_type_list_list = bitvec_type_list_list_cp
    found_improvement = False

    ### For each system, find the best solution
    worker_id_list = list(worker_id_dict.keys())
    while worker_id_list:

        worker_id, worker_id_list = ray.wait(
            worker_id_list, timeout=_TIMEOUT)
        failed = len(worker_id) == 0
        if not failed:
            try:
                _, _pvec_list, bitvec_type_all_list = ray.get(
                    worker_id[0], timeout=_TIMEOUT)
            except:
                failed = True
            if not failed:
            
                args = worker_id_dict[worker_id[0]]
                ast, _, _, _, _, _, _, _, _, _, _ = args
                _, _, type_ = ast

                type_i = type_[0]
                type_j = type_[1]

                ### `full_reset=True` means we will set all parameter
                ### managers to their optimized values
                ### `full_reset=False` means we will set only the main
                ### parameter manager to its optimized values and all other
                ### are set to the best value.
                for full_reset in [True, False]:
                    for mngr_idx in range(N_mngr):
                        ### For the targeted mngr we just set the
                        ### parameter values to the optimized ones.
                        ### The allocations will be set in the next step
                        if mngr_idx == mngr_idx_main:
                            allocations = [0 for _ in pvec_list_cp[mngr_idx_main].allocations]
                            pvec_list_cp[mngr_idx_main].allocations[:] = allocations
                            pvec_list_cp[mngr_idx_main].reset(
                                _pvec_list[mngr_idx_main],
                                pvec_list_cp[mngr_idx_main].allocations)
                        else:
                            ### Set all parameter values to
                            ### their optimized values.
                            if full_reset:
                                allocations = [-1 for _ in pvec_list_cp[mngr_idx].allocations]
                                bitvec_type_list_list_cp[mngr_idx] = bitvec_type_all_list[mngr_idx]
                                bitvec_hierarchy_to_allocations(
                                    bitvec_alloc_dict_list[mngr_idx], 
                                    bitvec_type_all_list[mngr_idx],
                                    allocations
                                    )
                                N_types = pvec_list_cp[mngr_idx].force_group_count
                                if allocations.count(-1) == 0 and max(allocations) < N_types:
                                    pvec_list_cp[mngr_idx].allocations[:] = allocations
                                    pvec_list_cp[mngr_idx].reset(
                                        _pvec_list[mngr_idx],
                                        pvec_list_cp[mngr_idx].allocations)
                            ### Set the parameter values to
                            ### their optimized values only for the 
                            ### targeted mngr and keep the previous
                            ### values for the other managers.
                            else:
                                bitvec_type_list_list_cp[mngr_idx] = bitvec_type_list_list[mngr_idx]
                                pvec_list_cp[mngr_idx].reset(
                                    pvec_list[mngr_idx],
                                    pvec_list[mngr_idx].allocations
                                    )
                        pvec_list_cp[mngr_idx].apply_changes()

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
                        if allocations.count(-1) == 0 and max(allocations) < N_types:
                            allocs = tuple(allocations)
                            if allocs not in alloc_list:
                                alloc_list.append(
                                    allocs
                                    )
                                bitvec_list.append(b)
                        bitvec_type_list_list_cp[mngr_idx_main].pop(type_j)

                    AIC_list = _calculate_AIC(
                        pvec_list_cp,
                        mngr_idx_main,
                        alloc_list,
                        targetcomputer
                        )
                    for idx, b in enumerate(bitvec_list):
                        new_AIC = AIC_list[idx]
                        accept = False
                        if new_AIC < best_AIC:
                            accept = True
                        if accept:
                            pvec_list_cp[mngr_idx_main].allocations[:] = alloc_list[idx]
                            pvec_list_cp[mngr_idx_main].apply_changes()
                            best_AIC       = new_AIC
                            best_pvec_list = [pvec.copy() for pvec in pvec_list_cp]
                            best_ast       = ast
                            
                            best_bitvec_type_list_list = copy.deepcopy(bitvec_type_list_list_cp)
                            best_bitvec_type_list_list[mngr_idx_main].insert(
                                type_j, b)
                            found_improvement = True

                for mngr_idx in range(N_mngr):
                    pvec_list_cp[mngr_idx].reset(
                        pvec_list_initial[mngr_idx])
                    bitvec_type_list_list_cp[mngr_idx] = copy.deepcopy(
                        bitvec_type_list_list_initial[mngr_idx])
            
                del worker_id_dict[worker_id[0]]

        if failed:
            if len(worker_id) > 0:
                if worker_id[0] not in worker_id_list:
                    worker_id_list.append(worker_id[0])
            resubmit_list = retrieve_failed_workers(worker_id_list)
            for worker_id in resubmit_list:
                args = worker_id_dict[worker_id]
                ast, _system_list_id, _targetcomputer_id, _pvec_all_id, _bitvec_type_list_id, _bounds_list, _parm_penalty_split, _mngr_idx_main, _bounds_penalty, _N_sys_per_likelihood_batch, _system_idx_list = args
                try:
                    ray.cancel(worker_id)
                except:
                    pass
                del worker_id_dict[worker_id]
                worker_id = minimize_FF.remote(
                        system_list = _system_list_id,
                        targetcomputer = _targetcomputer_id,
                        pvec_list = _pvec_all_id,
                        bitvec_type_list = _bitvec_type_list_id,
                        bounds_list = _bounds_list,
                        parm_penalty = _parm_penalty_split,
                        pvec_idx_min = [_mngr_idx_main],
                        N_sys_per_likelihood_batch = _N_sys_per_likelihood_batch,
                        bounds_penalty= _bounds_penalty,
                        use_scipy = True,
                        verbose = verbose,
                        )
                worker_id_dict[worker_id] = args
            worker_id_list = list(worker_id_dict.keys())
            import time
            time.sleep(_TIMEOUT)

    if found_improvement:
        for mngr_idx in range(N_mngr):
            pvec_list_cp[mngr_idx].reset(
                best_pvec_list[mngr_idx]
                )
            bitvec_type_list_list_cp[mngr_idx] = copy.deepcopy(
                best_bitvec_type_list_list[mngr_idx]
                )

    return found_improvement, pvec_list_cp, bitvec_type_list_list_cp, best_AIC


class BaseOptimizer(object):

    def __init__(
        self, 
        system_list, 
        name="BaseOptimizer",
        N_sys_per_likelihood_batch = 4,
        verbose=False):

        self._max_neighbor = 3

        self.bounds_list = list()
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

        self.system_list = system_list
        self._N_systems  = len(system_list)

        self.grad_diff = _EPSILON
        self.grad_diff_gs = _EPSILON_GS

        ### Note, don't make this too small.
        ### not finding the correct splits
        self.perturbation = 1.e-1

        self.verbose = verbose

        self.parm_mngr_cache_dict = dict()
        self.bsm_cache_dict  = dict()

        self._N_sys_per_likelihood_batch = N_sys_per_likelihood_batch

        self._initialize_targetcomputer()


    def _initialize_targetcomputer(self):

        from .targets import TargetComputer

        self.targetcomputer_id = ray.put(
            TargetComputer(
                self.system_list
            )
        )

        self.targetcomputer = ray.get(
            self.targetcomputer_id,
            timeout=_TIMEOUT
            )

    def add_parameters(
        self,
        parameter_manager,
        parameter_name_list = None,
        exclude_others = False,
        scale_list = None,
        bounds = None,
        bounds_penalty = 1.,
        rich_types = False,
        ):

        from . import arrays
        from . import parameters
        from . import bitvector_typing
        from .tools import remove_types
        import numpy as np

        if parameter_manager.N_systems != 0:
            raise ValueError(
                f"parameter_manager must be empty, but found {parameter_manager.N_systems} systems."
                )

        remove_types(
            self.system_list,
            [parameter_manager])

        self.parameter_manager_list.append(parameter_manager)
        parm_mngr = self.generate_parameter_manager(self.N_mngr)

        self.parameter_name_list.append(parameter_name_list)
        self.exclude_others.append(exclude_others)
        self.scaling_factor_list.append(scale_list)

        self.bounds_list.append(bounds)
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

        [pvec], _ = self.generate_parameter_vectors(
            mngr_idx_list=[self.N_mngr-1]
            )

        if rich_types:
            import copy
            _bond_ring = copy.deepcopy(
                bitvector_typing.BitSmartsManager.bond_ring)
            _bond_aromatic = copy.deepcopy(
                bitvector_typing.BitSmartsManager.bond_aromatic)
            bitvector_typing.BitSmartsManager.bond_ring = []
            bitvector_typing.BitSmartsManager.bond_aromatic = []
            if isinstance(parameter_manager, parameters.BondManager):
                bvc = bitvector_typing.BondBitvectorContainer()
            elif isinstance(parameter_manager, parameters.AngleManager):
                bvc = bitvector_typing.AngleBitvectorContainer()
            elif isinstance(parameter_manager, (parameters.ProperTorsionManager, parameters.MultiProperTorsionManager)):
                bvc = bitvector_typing.ProperTorsionBitvectorContainer()
            else:
                bvc = 0
            if bvc != 0:
                _bsm = bitvector_typing.BitSmartsManager(bvc, max_neighbor=3)
                _bsm.generate(ring_safe=True)
                _, _bitvec_list_alloc_dict = _bsm.prepare_bitvectors(max_neighbor=3)
                alloc_dict, smarts_dict, on_dict, subset_dict, bitvec_dict = _bsm.and_rows(
                    max_iter=0, generate_smarts=True)
                on_dict_sorted = sorted(on_dict.items(), key= lambda x: x[1])
                bitvec_list = list()
                for idx, _ in on_dict_sorted:
                    bitvec_list.append(
                        bitvec_dict[idx])
                allocations = [-1 for _ in pvec.allocations]
                bitvector_typing.bitvec_hierarchy_to_allocations(
                    bitvec_list_alloc_dict,
                    bitvec_list,
                    allocations)
                counts = 0
                for type_i, b in enumerate(bitvec_list):
                    if type_i in allocations:
                        if counts > 0:
                            pvec.duplicate(0)
                        counts += 1
                N_parms = pvec.parameters_per_force_group
                for type_i, b in enumerate(bitvec_list):
                    if type_i in allocations:
                        if self.verbose:
                            sma = _bsm.bitvector_to_smarts(b)
                            print(
                                f"Adding initial type {sma}")
                        self.best_bitvec_type_list[-1].append(b)
                        counts = pvec.force_group_count - 1
                        pvec[N_parms*counts:N_parms*(counts+1)] += np.random.normal(size=N_parms)
                allocations = [-1 for _ in pvec.allocations]
                bitvector_typing.bitvec_hierarchy_to_allocations(
                    bitvec_list_alloc_dict,
                    self.best_bitvec_type_list[-1],
                    allocations)
                pvec.allocations[:] = allocations[:]
                pvec.apply_changes()

            bitvector_typing.BitSmartsManager.bond_ring = _bond_ring
            bitvector_typing.BitSmartsManager.bond_aromatic = _bond_aromatic

        else:
            self.best_bitvec_type_list[-1].append(0)

        self.best_pvec_list[-1] = pvec.copy()


    @property
    def N_systems(self):
        return self._N_systems


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
        mngr_idx_list = list(),
        system_idx_list = list(),
        as_copy = False
        ):

        import copy

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

        _system_idx_list = sorted(_system_idx_list)
        pvec_list = list()
        bitvec_type_list = list()
        for mngr_idx in _mngr_idx_list:
            key = mngr_idx, tuple(_system_idx_list)
            parm_mngr = self.generate_parameter_manager(
                mngr_idx, 
                _system_idx_list
                )
            pvec = ForceFieldParameterVector(
                parm_mngr,
                self.parameter_name_list[mngr_idx],
                self.scaling_factor_list[mngr_idx],
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
            if self.best_pvec_list[mngr_idx] != None:
                pvec.allocations[:] = allocations[:]
                pvec.reset(
                    self.best_pvec_list[mngr_idx],
                    pvec.allocations
                    )

            if as_copy:
                pvec_list.append(
                    pvec.copy()
                    )
            else:
                pvec_list.append(
                    pvec
                    )

        return pvec_list, bitvec_type_list


    def calc_log_likelihood(
        self, 
        system_idx_list = list(),
        as_dict = False,
        ):
        
        import numpy as np

        if len(system_idx_list) == 0:
            _system_idx_list = list(range(self.N_systems))
        else:
            _system_idx_list = system_idx_list

        worker_id_dict = dict()
        for sys_idx in _system_idx_list:
            sys_dict = dict()
            if isinstance(sys_idx, int):
                sys = self.system_list[sys_idx]
                sys_dict[sys.name] = [sys.openmm_system]
            else:
                for _sys_idx in sys_idx:
                    sys = self.system_list[_sys_idx]
                    sys_dict[sys.name] = [sys.openmm_system]

            worker_id = self.targetcomputer(
                sys_dict,
                False
                )
            worker_id_dict[worker_id] = sys_idx

        if as_dict:
            logP_likelihood = dict()
        else:
            logP_likelihood = 0.
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
                    sys_idx = worker_id_dict[worker_id[0]]
                    if as_dict:
                        logP_likelihood[sys_idx] = _logP_likelihood
                    else:
                        logP_likelihood += _logP_likelihood
                    del worker_id_dict[worker_id[0]]
            if failed:
                if len(worker_id) > 0:
                    if worker_id[0] not in worker_id_list:
                        worker_id_list.append(worker_id[0])                    
                resubmit_list = retrieve_failed_workers(worker_id_list)
                for worker_id in resubmit_list:
                    sys_idx = worker_id_dict[worker_id]
                    ray.cancel(worker_id, force=True)
                    del worker_id_dict[worker_id]
                    if isinstance(sys_idx, int):
                        sys = self.system_list[sys_idx]
                        sys_dict[sys.name] = [sys.openmm_system]
                    else:
                        for _sys_idx in sys_idx:
                            sys = self.system_list[_sys_idx]
                            sys_dict[sys.name] = [sys.openmm_system]
                    worker_id = self.targetcomputer(
                        sys_dict,
                        False
                        )
                    worker_id_dict[worker_id] = sys_idx
                worker_id_list = list(worker_id_dict.keys())

        return logP_likelihood

    
    def update_best(
        self,         
        mngr_idx,
        pvec,
        b_list,
        ):

        import copy

        if not (pvec.force_group_count == len(b_list)):
            raise ValueError(
                f"Number of force groups in parameter vector {pvec.force_group_count} must be identical to bitvectors in b_list {len(b_list)}",
                )

        self.clear_cache([mngr_idx])

        self.best_pvec_list[mngr_idx] = pvec.copy()
        self.best_bitvec_type_list[mngr_idx]  = copy.deepcopy(b_list)


    def clear_cache(
        self, 
        mngr_idx_list = list()
        ):

        if len(mngr_idx_list) == 0:
            _mngr_idx_list = list(range(self.N_mngr))
        else:
            _mngr_idx_list = mngr_idx_list

        for mngr_idx in mngr_idx_list:
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
            parm_mngr = copy.deepcopy(self.parameter_manager_list[mngr_idx])
            s_list   = tuple()
            idx_list = tuple()
            for sys_idx in _system_idx_list:
                s_list   += (self.system_list[sys_idx],)
                idx_list += (sys_idx,)
                if len(s_list) == _CHUNK_SIZE:
                    worker_id = generate_parameter_manager.remote(
                        s_list, parm_mngr)
                    worker_id_list.append([worker_id, idx_list])
                    s_list   = tuple()
                    idx_list = tuple()
            if len(s_list) > 0:
                worker_id = generate_parameter_manager.remote(
                        s_list, parm_mngr)
                worker_id_list.append([worker_id, idx_list])
            sys_counts = 0
            for worker_id, idx_list in worker_id_list:
                _parm_mngr = ray.get(worker_id)
                parm_mngr.add_parameter_manager(_parm_mngr)
                for sys_idx in idx_list:
                    parm_mngr.system_list[sys_counts] = self.system_list[sys_idx]
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

        self.like_traj.append(self.calc_log_likelihood())
        self.aic_traj.append(self.calculate_AIC(parm_penalty=parm_penalty))

        self.pvec_traj.append(pvec_list_cp)
        self.bitvec_traj.append(bitvec_list)
        self.N_parms_traj.append(N_parms_all)


    def get_random_system_idx_list(
        self,
        N_sys_per_batch,
        N_batches,
        cluster_systems = False,
        ):
        
        if cluster_systems:
            from rdkit.Chem import rdMolDescriptors as rdmd
            from rdkit.Chem import DataStructs
            import numpy as np
            from scipy import cluster
            from scipy.spatial import distance
            
            nBits=1024
            fp_list = np.zeros((self.N_systems, nBits), dtype=np.int8)
            for sys_idx, sys in enumerate(self.system_list):
                fp = rdmd.GetMorganFingerprintAsBitVect(sys.rdmol, 3, nBits=nBits)
                arr = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp,arr)
                fp_list[sys_idx,:] = arr
                
            obs = cluster.vq.whiten(fp_list.astype(float))
            centroid, label = cluster.vq.kmeans2(obs, N_sys_per_batch, iter=100)
            label_re = [list() for _ in range(N_sys_per_batch)]
            for i in range(self.N_systems):
                k = label[i]
                label_re[k].append(i)
            system_idx_list_batch = tuple()
            for _ in range(N_batches):
                sys_list = tuple()
                for k in range(N_sys_per_batch):
                    if len(label_re[k]) > 0:
                        i = int(np.random.choice(label_re[k]))
                    else:
                        i = int(np.random.randint(0, self.N_systems))
                    sys_list += (i,)
                sys_list = list(sys_list)
                sys_list = tuple(sorted(sys_list))
                system_idx_list_batch += tuple([sys_list])

        else:
            import numpy as np
            system_idx_list = np.arange(
                self.N_systems, 
                dtype=int
                )

            if not (N_sys_per_batch < self.N_systems):
                N_batches = 1

            system_idx_list_batch = tuple()
            for _ in range(N_batches):
                np.random.shuffle(system_idx_list)
                sys_list = system_idx_list[:N_sys_per_batch].tolist()
                sys_list = tuple(sorted(sys_list))
                system_idx_list_batch += tuple([sys_list])

        return system_idx_list_batch


    def split_bitvector(
        self,
        mngr_idx,
        bitvec_type_list,
        system_idx_list,
        pvec_start=None,
        N_trials_gradient=5,
        split_all=False,
        max_on=0.1,
        max_splits=100,
        ):

        from . import arrays
        from .bitvector_typing import bitvec_hierarchy_to_allocations
        import numpy as np
        import copy
        import ray

        bsm, bitvec_alloc_dict_list = self.generate_bitsmartsmanager(
            mngr_idx,
            system_idx_list
            )
        [pvec], _ = self.generate_parameter_vectors(
            [mngr_idx],
            system_idx_list,
            as_copy=False
            )
        if not isinstance(pvec_start, type(None)):
            pvec.reset(pvec_start)

        N_types = pvec.force_group_count
        if split_all:
            type_query_list = list(range(N_types))
        else:
            type_query_list = [N_types-1]

        for _ in range(20):
            try:
                pvec_id = ray.put(pvec)
                break
            except:
                if _VERBOSE:
                    import traceback
                    print(traceback.format_exc())
                import time
                time.sleep(2)
        worker_id_dict = dict()
        alloc_bitvec_degeneracy_dict = dict()
        for type_i in type_query_list:
            type_j = type_i + 1
            selection_i = pvec.allocations.index([type_i])[0].tolist()

            if len(selection_i) == 0:
                continue
            alloc_dict, smarts_dict, on_dict, subset_dict, bitvec_dict = bsm.and_rows(
                max_iter = 3,
                allocations = selection_i,
                generate_smarts = False,
                max_neighbor = 3,
                max_on = max_on,
                duplicate_removal = False,
                verbose = self.verbose,
                )

            on_dict_sorted = sorted(on_dict.items(), key= lambda x: x[1])
            if self.verbose:
                print(
                    f"Found {len(on_dict_sorted)} candidate bitvectors for type {type_i} in mngr {mngr_idx} and systems {system_idx_list}."
                    )

            k_values_ij = np.zeros(
                (
                    max_splits, 
                    pvec.allocations.size
                    ),
                dtype=np.int16
                )

            counts = 0
            for t_idx, _ in on_dict_sorted:
                b_new = bitvec_dict[t_idx]
                b_old = bitvec_type_list[type_i]
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
                    #print(bsm.bitvector_to_smarts(b_new))
                    key   = tuple(allocations), tuple(selection_i), tuple([type_i, type_j])
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
                    f"Found {max_splits_effective} splits based on bitvectors for mngr {mngr_idx}."
                    )

            N_array_splits = 2
            if max_splits_effective > 10:
                N_array_splits = int(max_splits_effective/10)
            for split_idxs in np.array_split(np.arange(max_splits_effective), N_array_splits):
                worker_id = get_gradient_scores.remote(
                    ff_parameter_vector = pvec_id,
                    targetcomputer = self.targetcomputer_id,
                    type_i = type_i,
                    selection_i = selection_i,
                    k_values_ij = k_values_ij[split_idxs],
                    grad_diff = self.grad_diff_gs,
                    N_trials = N_trials_gradient,
                    N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                    )
                args = (pvec_id, self.targetcomputer_id, type_i, selection_i, k_values_ij[split_idxs], self.grad_diff, N_trials_gradient, self._N_sys_per_likelihood_batch, mngr_idx, system_idx_list)
                worker_id_dict[worker_id] = args

        return worker_id_dict, alloc_bitvec_degeneracy_dict
    

class ForceFieldOptimizer(BaseOptimizer):

    def __init__(
        self, 
        system_list, 
        parm_penalty_split = 1.,
        parm_penalty_merge = 1.,
        name="ForceFieldOptimizer",
        N_sys_per_likelihood_batch = 4,
        verbose=False):

        super().__init__(
            system_list, name, N_sys_per_likelihood_batch, verbose)

        self.parm_penalty_split = parm_penalty_split
        self.parm_penalty_merge = parm_penalty_merge


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
        mngr_idx_list = list(),
        ):

        import copy
        from .bitvector_typing import bitvec_hierarchy_to_allocations

        if len(mngr_idx_list) == 0:
            _mngr_idx_list = list(range(self.N_mngr))
        else:
            _mngr_idx_list = mngr_idx_list

        if self.verbose:
            print(
                "Looking for bad performing types."
                )

        found_improvement = True
        garbage_cycle_counter = 0
        while found_improvement:
            pvec_list, bitvec_type_list_list = self.generate_parameter_vectors()
            bitvec_list_alloc_dict_list = list()
            for mngr_idx in _mngr_idx_list:            
                _, bitvec_list_alloc_dict = self.generate_bitsmartsmanager(mngr_idx)
                bitvec_list_alloc_dict_list.append(
                    bitvec_list_alloc_dict
                    )
            found_improvement = False
            AIC_best  = self.calculate_AIC(
                parm_penalty=self.parm_penalty_merge
                )
            best_mngr_type = (None, None)
            if self.verbose:
                print(
                    f"Initial AIC {AIC_best} for garbage cycle {garbage_cycle_counter}"
                    )
            for mngr_idx in _mngr_idx_list:
                pvec             = pvec_list[mngr_idx]
                bitvec_type_list = bitvec_type_list_list[mngr_idx]
                bitvec_list_alloc_dict = bitvec_list_alloc_dict_list[mngr_idx]
                pvec_old         = pvec.copy()
                N_types          = pvec.force_group_count
                if N_types > 1:
                    for type_i in range(N_types):
                        type_i = N_types - type_i - 1
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

                            AIC_new = self.calculate_AIC(
                                parm_penalty=self.parm_penalty_merge
                                )
                            if AIC_new < AIC_best:
                                best_mngr_type = (mngr_idx, type_i)
                                AIC_best = AIC_new
                                found_improvement = True

                            pvec.reset(pvec_old)
                        bitvec_type_list.insert(type_i, b)

            if self.verbose:
                if found_improvement:
                    mngr_idx, type_i = best_mngr_type
                    print(
                        f"Removed bad performing type {type_i} with improved AIC {AIC_best} from mngr {mngr_idx} in garbage cycle {garbage_cycle_counter}."
                        )
                else:
                    print(
                        f"Did not remove any bad perfomring type in garbage cycle {garbage_cycle_counter}."
                        )

            if found_improvement:
                mngr_idx, type_i = best_mngr_type
                pvec_list, bitvec_type_list_list = self.generate_parameter_vectors([mngr_idx])
                _, bitvec_list_alloc_dict = self.generate_bitsmartsmanager(mngr_idx)
                pvec             = pvec_list[0]
                bitvec_type_list = bitvec_type_list_list[0]
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
                        sma = bsm.bitvector_to_smarts(b)
                        print(
                            f"Type {idx} ({pvec.force_group_histogram[idx]}): {sma}"
                            )

            garbage_cycle_counter += 1


    def get_votes(
        self,
        worker_id_dict,
        low_to_high=True,
        abs_grad_score=False,
        norm_cutoff=0.,
        grad_cutoff=0.,
        keep_N_best=10,
        ):

        import numpy as np

        if len(worker_id_dict) == 0:
            return list()

        votes_dict = dict()
        alloc_bitvec_dict = dict()

        worker_id_list = list(worker_id_dict.keys())

        score_dict  = dict()
        norm_dict   = dict()
        counts_dict = dict()
        
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(
                worker_id_list, timeout=_TIMEOUT)
            failed = len(worker_id) == 0
            if not failed:
                try:
                    result = ray.get(worker_id[0], timeout=_TIMEOUT)
                except:
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

                    del worker_id_dict[worker_id[0]]

            if failed:
                if len(worker_id) > 0:
                    if worker_id[0] not in worker_id_list:
                        worker_id_list.append(worker_id[0])
                resubmit_list = retrieve_failed_workers(worker_id_list)
                for worker_id in resubmit_list:
                    args = worker_id_dict[worker_id]
                    del worker_id_dict[worker_id]
                    ray.cancel(worker_id, force=True)
                    _pvec_id, _targetcomputer_id, _type_i, _selection_i, _k_values_ij, _grad_diff, _N_trials, _N_sys_per_likelihood_batch, _mngr_idx, _system_idx_list = args
                    worker_id = get_gradient_scores.remote(
                        ff_parameter_vector = _pvec_id,
                        targetcomputer = _targetcomputer_id,
                        type_i = _type_i,
                        selection_i = _selection_i,
                        k_values_ij = _k_values_ij,
                        grad_diff = _grad_diff,
                        N_trials = _N_trials,
                        N_sys_per_likelihood_batch = _N_sys_per_likelihood_batch
                        )
                    args = _pvec_id, _targetcomputer_id, _type_i, _selection_i, _k_values_ij, _grad_diff, _N_trials, _N_sys_per_likelihood_batch, _mngr_idx, _system_idx_list
                    worker_id_dict[worker_id] = args
                import time
                time.sleep(_TIMEOUT)
                worker_id_list = list(worker_id_dict.keys())

        to_delete = list()
        for ast in score_dict:
            score_dict[ast]   /= counts_dict[ast]
            norm_dict[ast][0] /= counts_dict[ast]
            norm_dict[ast][1] /= counts_dict[ast]
            if norm_dict[ast][0] < norm_cutoff and norm_dict[ast][1] < norm_cutoff:
                to_delete.append(ast)
            else:
                if low_to_high:
                    if score_dict[ast] > grad_cutoff:
                        to_delete.append(ast)
                else:
                    if score_dict[ast] < grad_cutoff:
                        to_delete.append(ast)

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
        system_idx_list = list(),
        votes_split_list = list(),
        ):

        use_scipy = True

        if len(system_idx_list) == 0:
            system_idx_list = list(range(self.N_systems))

        system_list = [self.system_list[sys_idx] for sys_idx in system_idx_list]
        pvec_all, bitvec_type_list = self.generate_parameter_vectors(
            [],
            system_idx_list,
            )
        for _ in range(20):
            try:
                system_list_id = ray.put(system_list)
                bitvec_type_list_id = ray.put(bitvec_type_list)
                break
            except:
                if _VERBOSE:
                    import traceback
                    print(traceback.format_exc())
                import time
                time.sleep(2)

        pvec_cp = pvec_all[mngr_idx_main].copy()
        N_types = pvec_all[mngr_idx_main].force_group_count

        pvec_all, bitvec_type_list = self.generate_parameter_vectors(
            [],
            system_idx_list,
            )

        worker_id_dict = dict()
        ### Query split candidates
        ### ======================
        for counts, ast in enumerate(votes_split_list):
            allocation, selection, type_ = ast

            pvec_all[mngr_idx_main].duplicate(type_[0])
            pvec_all[mngr_idx_main].swap_types(N_types, type_[1])
            pvec_all[mngr_idx_main].allocations[:] = list(allocation)
            pvec_all[mngr_idx_main].apply_changes()

            pvec_all_cp = [pvec.copy() for pvec in pvec_all]
            for _ in range(20):
                try:
                    pvec_all_id = ray.put(pvec_all_cp)
                    break
                except:
                    if _VERBOSE:
                        import traceback
                        print(traceback.format_exc())
                    import time
                    time.sleep(2)

            worker_id = minimize_FF.remote(
                    system_list = system_list_id,
                    targetcomputer = self.targetcomputer_id,
                    pvec_list = pvec_all_id,
                    bitvec_type_list = bitvec_type_list_id,
                    bounds_list = self.bounds_list,
                    parm_penalty = self.parm_penalty_split,
                    pvec_idx_min = [mngr_idx_main],
                    #pvec_idx_min = None,
                    N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                    bounds_penalty = self.bounds_penalty_list[mngr_idx_main],
                    use_scipy = use_scipy,
                    verbose = self.verbose,
                    )

            worker_id_dict[worker_id] = (
                ast, 
                system_list_id, 
                self.targetcomputer_id, 
                pvec_all_id, 
                bitvec_type_list_id, 
                self.bounds_list, 
                self.parm_penalty_split, 
                mngr_idx_main, 
                self.bounds_penalty_list[mngr_idx_main],
                self._N_sys_per_likelihood_batch,
                system_idx_list
                )
            pvec_all[mngr_idx_main].reset(pvec_cp)

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
        optimize_system_ordering=True,
        ### Keep the `keep_N_best` solutions for each split
        keep_N_best = 10,
        ### Prefix used for saving checkpoints
        prefix = "ForceFieldOptimizer",
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
        ):

        from .draw_bitvec import draw_bitvector_from_candidate_list
        from .bitvector_typing import bitvec_hierarchy_to_allocations

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
        if not hasattr(self, "_N_sys_per_likelihood_batch"):
            self._N_sys_per_likelihood_batch = 4

        if not restart:
            self.system_idx_list_batch = []
            self.minscore_worker_id_dict = dict()
            self.selection_worker_id_dict = dict()
            self.bitvec_dict = dict()
            self.split_iteration_idx = 0
            self.accepted_counter = 0

        if self.verbose and restart:
            print("Attempting to restart from previous run.")

        for iteration_idx in range(iterations):

            if iteration_idx > 0:
                if (iteration_idx%pair_incr_split) == 0:
                    N_sys_per_batch_split += 1

            raute_fill = ''.ljust(len(str(iteration_idx))-1,"#")
            print("ITERATION", iteration_idx)
            print(f"###########{raute_fill}")

            ### ============== ###
            ### PROCESS SPLITS ###
            ### ============== ###
            found_improvement   = True
            while self.split_iteration_idx < max_splitting_attempts:
                print(f"ATTEMPTING SPLIT {iteration_idx}/{self.split_iteration_idx}")
                found_improvement = False
                if not restart:
                    if isinstance(system_idx_dict_batch, dict):
                        self.system_idx_list_batch = system_idx_dict_batch[iteration_idx]
                    else:
                        self.system_idx_list_batch = self.get_random_system_idx_list(
                            N_sys_per_batch_split, N_batches, cluster_systems=cluster_systems)

                if optimize_system_ordering:
                    if self.verbose:
                        print(
                            "Initial ordering of systems:",
                            self.system_idx_list_batch
                            )
                        print(
                            "Optimizing system priority timings ..."
                            )

                    worker_id_dict = dict()
                    time_diff_dict = dict()
                    for sys_idx_pair in self.system_idx_list_batch:
                        time_diff_dict[sys_idx_pair] = 0.
                        pvec_list, _ = self.generate_parameter_vectors([0], sys_idx_pair)
                        pvec = pvec_list[0]
                        for _ in range(N_trials_opt):
                            worker_id = _test_logL.remote(
                                pvec, 
                                self.targetcomputer_id
                                )
                            worker_id_dict[worker_id] = sys_idx_pair

                    worker_id_list = list(worker_id_dict.keys())
                    while worker_id_list:
                        worker_id, worker_id_list = ray.wait(
                            worker_id_list, timeout=_TIMEOUT)
                        failed = len(worker_id) == 0
                        if not failed:
                            try:
                                results = ray.get(worker_id[0], timeout=_TIMEOUT)
                            except:
                                failed = True
                            if not failed:
                                sys_idx_pair = worker_id_dict[worker_id[0]]
                                time_diff_dict[sys_idx_pair] += results/N_trials_opt
                                del worker_id_dict[worker_id[0]]
                        if failed:
                            if len(worker_id) > 0:
                                if worker_id[0] not in worker_id_list:
                                    worker_id_list.append(worker_id[0])
                            resubmit_list = retrieve_failed_workers(worker_id_list)
                            for worker_id in resubmit_list:
                                del worker_id_dict[worker_id]
                                ray.cancel(worker_id, force=True)
                                pvec_list, _ = self.generate_parameter_vectors([0], sys_idx_pair)
                                pvec = pvec_list[0]
                                worker_id = _test_logL.remote(
                                    pvec, 
                                    self.targetcomputer_id
                                    )
                                worker_id_dict[worker_id] = sys_idx_pair
                        worker_id_list = list(worker_id_dict.keys())

                    self.system_idx_list_batch = [sys_idx_pair for sys_idx_pair, _ in sorted(time_diff_dict.items(), key=lambda items: items[1], reverse=True)]
                    self.system_idx_list_batch = tuple(self.system_idx_list_batch)
                    if self.verbose:
                        print(
                            "Optimized ordering of systems:",
                            self.system_idx_list_batch
                            )

                if self.verbose:
                    print(
                        "Optimize parameters."
                        )
                if not restart:
                    minimize_initial_worker_id_dict = dict()
                    for sys_idx_pair in self.system_idx_list_batch:
                        pvec_list, _ = self.generate_parameter_vectors(
                            system_idx_list=sys_idx_pair)
                        worker_id = minimize_FF.remote(
                            system_list = [self.system_list[sys_idx] for sys_idx in sys_idx_pair],
                            targetcomputer = self.targetcomputer_id,
                            pvec_list = pvec_list,
                            bitvec_type_list = list(),
                            bounds_list = self.bounds_list,
                            parm_penalty = 1.,
                            N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                            use_DE=False,
                            bounds_penalty=100.)
                        minimize_initial_worker_id_dict[worker_id] = sys_idx_pair

                    if self.verbose:
                        print(
                            "Generating splits and computing grad scores."
                        )
                    split_worker_id_dict = dict()
                    worker_id_list = list(minimize_initial_worker_id_dict.keys())
                    while worker_id_list:
                        worker_id, worker_id_list = ray.wait(
                            worker_id_list, timeout=_TIMEOUT)
                        failed = len(worker_id) == 0
                        if not failed:
                            try:
                                _, pvec_list_cp, _ = ray.get(worker_id[0])
                                sys_idx_pair = minimize_initial_worker_id_dict[worker_id[0]]
                                del minimize_initial_worker_id_dict[worker_id[0]]
                                for mngr_idx in range(self.N_mngr):
                                    split_worker_id_dict[mngr_idx,sys_idx_pair] = self.split_bitvector(                                                                   
                                        mngr_idx,                                                                                                                         
                                        self.best_bitvec_type_list[mngr_idx],                                                                                             
                                        sys_idx_pair,                                                                                                                     
                                        pvec_start=pvec_list_cp[mngr_idx],                                                                                                
                                        N_trials_gradient=N_trials_gradient,
                                        split_all=True)
                            except:
                                failed = True
                        if failed:
                            if len(worker_id) > 0:
                                if worker_id[0] not in worker_id_list:
                                    worker_id_list.append(worker_id[0])
                            resubmit_list = retrieve_failed_workers(worker_id_list)
                            for worker_id in resubmit_list:
                                sys_idx_pair = minimize_initial_worker_id_dict[worker_id]
                                del minimize_initial_worker_id_dict[worker_id]
                                try:
                                    ray.cancel(worker_id)
                                except:
                                    pass
                                pvec_list, _ = self.generate_parameter_vectors(
                                    system_idx_list=sys_idx_pair)
                                worker_id = minimize_FF.remote(
                                    system_list = [self.system_list[sys_idx] for sys_idx in sys_idx_pair],
                                    targetcomputer = self.targetcomputer_id,
                                    pvec_list = pvec_list,
                                    bitvec_type_list = list(),
                                    bounds_list = self.bounds_list,
                                    parm_penalty = 1.,
                                    N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                                    use_DE=False,
                                    bounds_penalty=1000.) 
                                minimize_initial_worker_id_dict[worker_id] = sys_idx_pair

                            import time
                            time.sleep(_TIMEOUT)
                            worker_id_list = list(minimize_initial_worker_id_dict.keys())

                    if self.verbose:
                        print(
                            "Obtaining votes and submitting parameter minimizations."
                            )
                    for mngr_idx in range(self.N_mngr):
                        for sys_idx_pair in self.system_idx_list_batch:
                            votes_split_list = self.get_votes(
                                worker_id_dict = split_worker_id_dict[mngr_idx,sys_idx_pair][0],
                                low_to_high = True,
                                abs_grad_score = False,
                                norm_cutoff = 1.e-2,
                                keep_N_best = keep_N_best,
                                )
                            self.minscore_worker_id_dict[mngr_idx,sys_idx_pair] = self.get_min_scores(
                                mngr_idx_main=mngr_idx,
                                system_idx_list=sys_idx_pair,
                                votes_split_list=votes_split_list,
                                )
                            self.bitvec_dict[mngr_idx,sys_idx_pair] = dict()
                            for ast in votes_split_list:
                                b_list = split_worker_id_dict[mngr_idx,sys_idx_pair][1][ast]
                                self.bitvec_dict[mngr_idx,sys_idx_pair][ast] = b_list

                            if self.verbose:
                                print(
                                    f"For mngr {mngr_idx} and systems {sys_idx_pair}:\n"
                                    f"Found {len(votes_split_list)} candidate split solutions ...\n",
                                    )
                            del split_worker_id_dict[mngr_idx,sys_idx_pair]

                    import pickle
                    with open(f"{prefix}-MAIN-{iteration_idx+output_offset}-SPLIT-{self.split_iteration_idx}-SELECTION.pickle", "wb") as fopen:
                        pickle.dump(
                            self,
                            fopen
                        )

                    ### =============
                    ### END SPLITTING
                    ### =============

                if self.verbose:
                    print(
                        "Finding best parameters."
                        )
                self.system_idx_list_batch = self.system_idx_list_batch[::-1]
                gibbs_dict = dict()
                gibbs_count_dict = dict()
                old_pvec_list, old_bitvec_type_list = self.generate_parameter_vectors()
                best_AIC = self.calculate_AIC(
                    parm_penalty=self.parm_penalty_split
                    )
                bitvec_alloc_dict_list = list()
                for mngr_idx in range(self.N_mngr):
                    _, bitvec_alloc_dict = self.generate_bitsmartsmanager(
                        mngr_idx,
                        )
                    bitvec_alloc_dict_list.append(
                        bitvec_alloc_dict
                        )
                        
                ### If we want to restart, first make sure that we re-run all
                ### the left-over minimization runs
                if restart:
                    _minscore_worker_id_dict = dict()
                    for mngr_idx, sys_idx_pair in self.minscore_worker_id_dict:
                        pvec_list, bitvec_type_list = self.generate_parameter_vectors(
                            system_idx_list=sys_idx_pair)
                        pvec_all_id = ray.put(pvec_list)
                        bitvec_type_list_id = ray.put(bitvec_type_list)
                        system_list_id = ray.put([self.system_list[sys_idx] for sys_idx in sys_idx_pair])
                        
                        _minscore_worker_id_dict[mngr_idx, sys_idx_pair] = dict()
                        for ast in self.bitvec_dict[mngr_idx,sys_idx_pair]:
                            worker_id = minimize_FF.remote(
                                    system_list = system_list_id,
                                    targetcomputer = self.targetcomputer_id,
                                    pvec_list = pvec_all_id,
                                    bitvec_type_list = bitvec_type_list_id,
                                    bounds_list = self.bounds_list,
                                    parm_penalty = self.parm_penalty_split,
                                    pvec_idx_min = [mngr_idx],
                                    bounds_penalty= self.bounds_penalty_list[mngr_idx],
                                    N_sys_per_likelihood_batch = self._N_sys_per_likelihood_batch,
                                    use_scipy = True,
                                    verbose = self.verbose,
                                    )
                            args = ast, system_list_id, self.targetcomputer_id, pvec_all_id, \
                                   bitvec_type_list_id, self.bounds_list, self.parm_penalty_split, \
                                   mngr_idx, self.bounds_penalty_list[mngr_idx], self._N_sys_per_likelihood_batch, \
                                   sys_idx_pair
                            _minscore_worker_id_dict[mngr_idx, sys_idx_pair][worker_id] = args
                    self.minscore_worker_id_dict = _minscore_worker_id_dict

                for mngr_idx in range(self.N_mngr):
                    self.selection_worker_id_dict[mngr_idx] = dict()
                for mngr_idx, sys_idx_pair in self.minscore_worker_id_dict:
                    b_list = self.bitvec_dict[mngr_idx, sys_idx_pair]
                    worker_id = set_parameters_remote.remote(
                        mngr_idx_main = mngr_idx,
                        pvec_list = old_pvec_list,
                        targetcomputer = self.targetcomputer_id,
                        bitvec_dict = b_list,
                        bitvec_alloc_dict_list = bitvec_alloc_dict_list,
                        bitvec_type_list_list = old_bitvec_type_list,
                        worker_id_dict = self.minscore_worker_id_dict[mngr_idx,sys_idx_pair],
                        parm_penalty = self.parm_penalty_split,
                        verbose = self.verbose,
                        )
                    self.selection_worker_id_dict[mngr_idx][worker_id] = sys_idx_pair

                for mngr_idx in range(self.N_mngr):
                    selection_worker_id_list = list(self.selection_worker_id_dict[mngr_idx].keys())
                    while selection_worker_id_list:
                        worker_id, selection_worker_id_list = ray.wait(
                            selection_worker_id_list, timeout=_TIMEOUT*100)
                        failed = len(worker_id) == 0
                        if not failed:
                            #try:
                            #    _, pvec_list, best_bitvec_type_list, new_AIC = ray.get(
                            #        worker_id[0], timeout=_TIMEOUT)
                            #except:
                            #    failed = True
                            _, pvec_list, best_bitvec_type_list, new_AIC = ray.get(
                                    worker_id[0], timeout=_TIMEOUT)
                            if not failed:
                                sys_idx_pair = self.selection_worker_id_dict[mngr_idx][worker_id[0]]
                                if self.verbose:
                                    print("mngr_idx/sys_idx_pair", mngr_idx, "/", sys_idx_pair)

                                found_improvement_mngr = new_AIC < best_AIC

                                if self.verbose:
                                    print(
                                        "Current best AIC:", best_AIC,
                                        "New AIC:", new_AIC,
                                        )
                                    if found_improvement_mngr:
                                        print(
                                            f"Found improvement for sys {sys_idx_pair}."
                                            )
                                    else:
                                        print(
                                            f"Found no improvement for sys {sys_idx_pair}.",
                                            )

                                del self.selection_worker_id_dict[mngr_idx][worker_id[0]]
                                del self.minscore_worker_id_dict[mngr_idx,sys_idx_pair]
                                del self.bitvec_dict[mngr_idx, sys_idx_pair]
                                
                                if found_improvement_mngr:
                                    found_improvement = True
                                    best_AIC = new_AIC
                                else:
                                    continue

                                if self.verbose:
                                    print("Solution globally accepted.")
                                for _mngr_idx in range(self.N_mngr):
                                    old_pvec_list[_mngr_idx].reset(
                                        pvec_list[_mngr_idx]
                                        )
                                    old_bitvec_type_list[_mngr_idx] = copy.deepcopy(
                                        best_bitvec_type_list[_mngr_idx])
                                    if self.verbose:
                                        print(
                                            f"Updated parameters for mngr {_mngr_idx}:",
                                            old_pvec_list[_mngr_idx].vector_k,
                                            f"with {old_pvec_list[_mngr_idx].force_group_count} force groups"
                                            )
                                        print(
                                            f"Updated allocations for mngr {_mngr_idx}:",
                                            old_pvec_list[_mngr_idx].allocations
                                        )
                                        print(
                                            f"Updated bitvec hashes for mngr {_mngr_idx}:",
                                            old_bitvec_type_list[_mngr_idx]
                                            )
                                        print(
                                            f"Updated SMARTS hierarchy for mngr {_mngr_idx}:"
                                            )

                                        bsm, _ = self.generate_bitsmartsmanager(_mngr_idx)
                                        for idx, b in enumerate(old_bitvec_type_list[_mngr_idx]):
                                            sma = bsm.bitvector_to_smarts(b)
                                            print(
                                                f"Type {idx} : {sma}"
                                                )

                                if found_improvement_mngr:
                                    for _mngr_idx in range(self.N_mngr):
                                        self.update_best(
                                            _mngr_idx,
                                            old_pvec_list[_mngr_idx],
                                            old_bitvec_type_list[_mngr_idx]
                                            )

                                import pickle
                                with open(f"{prefix}-MAIN-{iteration_idx+output_offset}-SPLIT-{self.split_iteration_idx}-ACCEPTED-{self.accepted_counter}.pickle", "wb") as fopen:
                                    pickle.dump(
                                        self,
                                        fopen
                                    )
                                self.accepted_counter += 1
                                
                        if failed:
                            if len(worker_id) > 0:
                                if worker_id[0] not in selection_worker_id_list:
                                    selection_worker_id_list.append(worker_id[0])
                            resubmit_list = retrieve_failed_workers(selection_worker_id_list)
                            for worker_id in resubmit_list:
                                sys_idx_pair = self.selection_worker_id_dict[mngr_idx][worker_id]
                                b_list = self.bitvec_dict[mngr_idx, sys_idx_pair]
                                ray.cancel(worker_id, force=True)
                                del self.selection_worker_id_dict[mngr_idx][worker_id]

                                worker_id = set_parameters_remote.remote(
                                    mngr_idx_main = mngr_idx,
                                    pvec_list = old_pvec_list,
                                    targetcomputer = self.targetcomputer_id,
                                    bitvec_dict = b_list,
                                    bitvec_alloc_dict_list = bitvec_alloc_dict_list,
                                    bitvec_type_list_list = old_bitvec_type_list,
                                    worker_id_dict = self.minscore_worker_id_dict[mngr_idx,sys_idx_pair],
                                    parm_penalty = self.parm_penalty_split,
                                    verbose = self.verbose,
                                    )
                                self.selection_worker_id_dict[mngr_idx][worker_id] = sys_idx_pair
                            selection_worker_id_list = list(self.selection_worker_id_dict[mngr_idx].keys())

                self.split_iteration_idx += 1

                ### Remove tempory data
                ### self.split_iteration_idx must not be reset
                self.system_idx_list_batch = []
                self.minscore_worker_id_dict = dict()
                self.selection_worker_id_dict = dict()
                self.bitvec_dict = dict()
                self.accepted_counter = 0

                if restart:
                    restart = False

                ### ================= ###
                ### END SPLIT ATTEMPT ###
                ### ================= ###
            
            ### Remove tempory data
            if restart:
                restart = False
            self.system_idx_list_batch = []
            self.minscore_worker_id_dict = dict()
            self.selection_worker_id_dict = dict()
            self.bitvec_dict = dict()
            # self.split_iteration_idx must be reset here
            self.split_iteration_idx = 0
            self.accepted_counter = 0
                
            ### ========================== ###
            ### DRAW FROM TYPING POSTERIOR ###
            ### ========================== ###
            mngr_schedule = np.arange(self.N_mngr, dtype=int)
            np.random.shuffle(mngr_schedule)
            for mngr_idx in mngr_schedule:
                if max_gibbs_attempts == 0:
                    continue
                if self.verbose:
                    print(
                        f"Drawing types from typing posterior for mngr {mngr_idx}"
                        )
                [pvec], [bitvec_list] = self.generate_parameter_vectors([mngr_idx])
                bsm, bitvec_alloc_dict_list = self.generate_bitsmartsmanager(mngr_idx)
                bitvec_list_new = draw_bitvector_from_candidate_list(
                    pvec, 
                    bitvec_list,
                    bsm,
                    self.targetcomputer_id,
                    theta=1000.,
                    alpha=10.,
                    N_iter = max_gibbs_attempts,
                    max_on=0.1,
                    verbose = self.verbose
                )
                allocations = [-1 for _ in pvec.allocations]
                bitvec_hierarchy_to_allocations(
                    bitvec_alloc_dict_list,
                    bitvec_list_new,
                    allocations
                    )
                if allocations.count(-1) == 0:
                    pvec.allocations[:] = allocations
                    pvec.apply_changes()
                    self.update_best(
                        mngr_idx,
                        pvec,
                        bitvec_list_new
                        )
                if self.verbose:
                    if allocations.count(-1) == 0:
                        print("Updated final allocations.")
                    else:
                        print("Could not final allocations.")

            ### ================== ###
            ### GARBAGE COLLECTION ###
            ### ================== ###
            self.garbage_collection()
            
            import pickle
            with open(f"{prefix}-MAIN-{iteration_idx+output_offset}-GARBAGE_COLLECTION.pickle", "wb") as fopen:
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

            import pickle
            with open(f"{prefix}-MAIN-{iteration_idx+output_offset}.pickle", "wb") as fopen:
                pickle.dump(
                    self,
                    fopen
                )
        ### END LOOP OVER ITERATIONS

