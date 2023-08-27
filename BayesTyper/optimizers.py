import numpy as np
from scipy import stats
import copy

from .kernels import LogJumpKernel
from .kernels import compute_gradient_per_forcegroup
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
                        _INACTIVE_GROUP_IDX
                        )

import ray

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
    logL_list = np.zeros(N_queries, dtype=float)

    openmm_system_list = list()
    for sys in pvec_cp.parameter_manager.system_list:
        openmm_system_list.append(
            {sys.name : [sys.openmm_system]}
            )

    init_alloc = copy.deepcopy(pvec_cp.allocations[:])
    worker_id_dict = dict()
    cache_dict = dict()
    for idx in range(N_queries):
        typing = list(typing_list[idx])
        typing_tuple = tuple(typing)
        if typing_tuple in cache_dict:
            worker_id_dict[idx] = cache_dict[typing_tuple]
        else:
            cache_dict[typing_tuple] = idx
            worker_id_dict[idx] = list()
            pvec_cp.allocations[:] = typing[:]
            pvec_cp.apply_changes()
            for ommdict in openmm_system_list:
                worker_id_dict[idx].append(
                    targetcomputer(
                        copy.deepcopy(ommdict), 
                        False
                        )
                    )
    for idx in range(N_queries):
        typing = list(typing_list[idx])
        typing_tuple = tuple(typing)
        logP_likelihood = 0.
        if isinstance(worker_id_dict[idx], int):
            logP_likelihood += logL_list[worker_id_dict[idx]]
        else:
            for worker_id in worker_id_dict[idx]:
                _logP_likelihood = ray.get(worker_id)
                logP_likelihood += _logP_likelihood
        logL_list[idx] = logP_likelihood

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


class AngleBounds(object):

    def __init__(self, parameter_name_list=list()):

        from rdkit import Chem
        from openmm import unit

        self.hybridization_type_dict_eq = {
            Chem.HybridizationType.SP3 : 109.5 * unit.degree,
            Chem.HybridizationType.SP2 : 120.0 * unit.degree,
            Chem.HybridizationType.SP  : 180.0 * unit.degree,
        }

        self.hybridization_type_dict_min = {
            Chem.HybridizationType.SP3 : (109.5 - 15.) * unit.degree,
            Chem.HybridizationType.SP2 : (120.0 - 15.) * unit.degree,
            Chem.HybridizationType.SP  : (180.0 - 15.) * unit.degree,
        }

        self.hybridization_type_dict_max = {
            Chem.HybridizationType.SP3 : (109.5 + 15.) * unit.degree,
            Chem.HybridizationType.SP2 : (120.0 + 15.) * unit.degree,
            Chem.HybridizationType.SP  : (180.0 + 0.) * unit.degree,
        }

        from .constants import _ENERGY_PER_MOL, _ANGLE_RAD, _ANGLE_DEG
        lower_k = 0.01 * _ENERGY_PER_MOL * _ANGLE_DEG**-2
        upper_k = 0.50 * _ENERGY_PER_MOL * _ANGLE_DEG**-2

        self.lower = [lower_k.in_units_of(_FORCE_CONSTANT_ANGLE),  90. * unit.degree]
        self.upper = [upper_k.in_units_of(_FORCE_CONSTANT_ANGLE), 180. * unit.degree]

        self.parameter_name_list = parameter_name_list

    def get_bounds(self, atom_list, rdmol):

        import copy
        from openmm import unit

        atom_idx = atom_list[1]

        if 'k' in self.parameter_name_list:
            lower = [self.lower[0]]
            upper = [self.upper[0]]
        elif 'angle' in self.parameter_name_list:
            lower = [self.lower[1]]
            upper = [self.upper[1]]

            atom = rdmol.GetAtomWithIdx(atom_idx)
            hyb  = atom.GetHybridization()
            if hyb in self.hybridization_type_dict_min: 
                min_angle = self.hybridization_type_dict_min[hyb]
            if hyb in self.hybridization_type_dict_max: 
                max_angle = self.hybridization_type_dict_max[hyb]

            lower[0] = min_angle
            upper[0] = max_angle
        else:
            lower = self.lower
            upper = self.upper

            atom = rdmol.GetAtomWithIdx(atom_idx)
            hyb  = atom.GetHybridization()
            if hyb in self.hybridization_type_dict_min: 
                min_angle = self.hybridization_type_dict_min[hyb]
            if hyb in self.hybridization_type_dict_max: 
                max_angle = self.hybridization_type_dict_max[hyb]

            lower[1] = min_angle
            upper[1] = max_angle

        return lower, upper


class BondBounds(object):

    def __init__(self, parameter_name_list=list()):

        from openmm import unit

        if 'k' in parameter_name_list:
            self.lower = [100000. * _FORCE_CONSTANT_BOND]
            self.upper = [600000. * _FORCE_CONSTANT_BOND]
        elif 'length' in parameter_name_list:
            self.lower = [0.05 * unit.nanometer]
            self.upper = [0.20 * unit.nanometer]            
        else:
            self.lower = [100000. * _FORCE_CONSTANT_BOND, 0.05 * unit.nanometer]
            self.upper = [600000. * _FORCE_CONSTANT_BOND, 0.20 * unit.nanometer]

    def get_bounds(self, atom_list, rdmol):

        import copy

        lower = copy.deepcopy(self.lower)
        upper = copy.deepcopy(self.upper)
        
        return lower, upper


class TorsionBounds(object):

    def __init__(self, parameter_name_list=list()):

        self._lower = [-0.0 * _FORCE_CONSTANT_TORSION]
        self._upper = [+0.0 * _FORCE_CONSTANT_TORSION]

    def get_bounds(self, atom_list, rdmol):

        return self._lower, self._upper


class DoubleTorsionBounds(object):

    def __init__(self, parameter_name_list=list()):

        self._lower = [-0.0 * _FORCE_CONSTANT_TORSION, -0.0 * _FORCE_CONSTANT_TORSION]
        self._upper = [+0.0 * _FORCE_CONSTANT_TORSION, +0.0 * _FORCE_CONSTANT_TORSION]

    def get_bounds(self, atom_list, rdmol):

        return self._lower, self._upper


class MultiTorsionBounds(object):

    def __init__(self, parameter_name_list=list()):

        N_parms = len(parameter_name_list)

        self._lower = [-0.0 * _FORCE_CONSTANT_TORSION for _ in range(N_parms)]
        self._upper = [+0.0 * _FORCE_CONSTANT_TORSION for _ in range(N_parms)]

    def get_bounds(self, atom_list, rdmol):

        return self._lower, self._upper


@ray.remote
def get_gradient_scores(
    ff_parameter_vector,
    targetcomputer,
    type_i, # type to be split
    selection_i = None,
    k_values_ij = None,
    grad_diff = 1.e-2,
    N_trials = 10,
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
        targetcomputer
        )

    initial_vec = copy.deepcopy(ff_parameter_vector_cp[first_parm_i:last_parm_i])

    grad_score_dict = dict()
    grad_norm_dict  = dict()
    allocation_list_dict = dict()
    selection_list_dict = dict()
    type_list_dict  = dict()
    for trial_idx in range(N_trials):
        ff_parameter_vector_cp[first_parm_i:last_parm_i]  = initial_vec
        ff_parameter_vector_cp[first_parm_i:last_parm_i] += np.random.normal(0, 0.1, initial_vec.size)
        ff_parameter_vector_cp[first_parm_j:last_parm_j]  = ff_parameter_vector_cp[first_parm_i:last_parm_i]

        grad_score      = list()
        grad_norm       = list()
        allocation_list = list()
        selection_list  = list()
        type_list       = list()        
        for comb_idx in range(N_comb):
            ff_parameter_vector_cp.allocations[:] = k_values_ij[comb_idx]
            ff_parameter_vector_cp.apply_changes()

            grad = likelihood_func.grad(
                ff_parameter_vector_cp[:],
                parm_idx_list=parm_idx_list,
                grad_diff=grad_diff,
                use_jac=False
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
                grad_score.append(grad_ij_diff)
            else:
                grad_score.append(grad_ij_dot)
            grad_norm.append([norm_i, norm_j])
            allocation_list.append(tuple(k_values_ij[comb_idx].tolist()))
            selection_list.append(tuple(selection_i))
            type_list.append(tuple([type_i, type_j]))

        grad_score_dict[trial_idx] = grad_score
        grad_norm_dict[trial_idx]  = grad_norm
        allocation_list_dict[trial_idx] = allocation_list
        selection_list_dict[trial_idx] = selection_list
        type_list_dict[trial_idx]  = type_list

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
    force_group_idxs=None,
    grad_diff=1.e-2,
    parallel_targets=True,
    bounds_penalty=1000.,
    use_scipy=False,
    verbose=False,
    get_timing=False):

    if get_timing:
        import time
        time_start = time.time()

    from openmm import unit
    import numpy as np
    import copy

    pvec_list_cp   = copy.deepcopy(pvec_list)
    system_list_cp = copy.deepcopy(system_list)

    pvec_min_list  = list()
    lb_bounds_list = list()
    ub_bounds_list = list()
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

    excluded_parameter_idx_list = []
    for pvec_idx in pvec_idx_list:

        pvec = pvec_list_cp[pvec_idx]
        pvec_min_list.append(pvec)

        if not isinstance(force_group_idxs, type(None)):
            N_parms = pvec.parameters_per_force_group
            for force_group_idx in range(pvec.force_group_count):
                ### Make sure we eliminate eny double entries.
                f = set(force_group_idxs[pvec_idx])
                force_group_idxs[pvec_idx] = list(f)
                for idx in force_group_idxs[pvec_idx]:
                    if force_group_idx != idx:
                        for i in range(N_parms):
                            exclude_idx = idx * N_parms + i
                            if not exclude_idx in excluded_parameter_idx_list:
                                excluded_parameter_idx_list.append(
                                    exclude_idx
                                )

        ### `lazy=True` just replaces the systems stored in
        ### in the parameter manager with the ones in `system_list_cp`.
        ### It does not completely rebuild the parameter_manager, but
        ### performs some basic sanity checks.
        pvec.rebuild_from_systems(
            lazy = True,
            system_list = system_list_cp
            )
        pvec.apply_changes()

        lb = list()
        ub = list()
        bounds = bounds_list[pvec_idx]
        for i in range(pvec.force_group_count):
            valid_ranks = list()
            atom_list_list = np.array(pvec.parameter_manager.atom_list)
            for s in pvec.allocations.index([i])[0]:
                valid_ranks.extend(
                    np.where(s == pvec.parameter_manager.force_ranks)[0].tolist()
                    )
            sys_idx_list = pvec.parameter_manager.system_idx_list[valid_ranks]
            atom_list_list = atom_list_list[valid_ranks].tolist()
            assert len(sys_idx_list) == len(atom_list_list)
            _lb = list()
            _ub = list()
            for sys_idx, atom_list in zip(sys_idx_list, atom_list_list):
                rdmol = system_list[sys_idx].rdmol
                if bounds == None:
                    minval_list = [-np.inf * pvec.vector_units[i] for i in range(pvec.parameters_per_force_group)]
                    maxval_list = [ np.inf * pvec.vector_units[i] for i in range(pvec.parameters_per_force_group)]
                else:
                    minval_list, maxval_list = bounds.get_bounds(
                        atom_list, 
                        rdmol
                        )
                _lb.append(minval_list)
                _ub.append(maxval_list)
            if len(_lb) == 0:
                _lb = [[-np.inf * pvec.vector_units[i] for i in range(pvec.parameters_per_force_group)]]
                _ub = [[ np.inf * pvec.vector_units[i] for i in range(pvec.parameters_per_force_group)]]
            _lb_best = _lb[0]
            for _l in _lb:
                for val_i, val in enumerate(_l):
                    if val < _lb_best[val_i]:
                        _lb_best[val_i] = val
            _ub_best = _ub[0]
            for _u in _ub:
                for val_i, val in enumerate(_u):
                    if val > _ub_best[val_i]:
                        _ub_best[val_i] = val

            lb.extend(_lb_best)
            ub.extend(_ub_best)
        lb = np.array(lb)
        ub = np.array(ub)
        lb = np.array(
            pvec.get_transform(lb)
            )
        ub = np.array(
            pvec.get_transform(ub)
            )

        lb_bounds_list.extend(lb.tolist())
        ub_bounds_list.extend(ub.tolist())

    lb_bounds_list = np.array(lb_bounds_list).flatten()
    ub_bounds_list = np.array(ub_bounds_list).flatten()

    likelihood_func = LikelihoodVectorized(
        pvec_min_list,
        targetcomputer,
        )

    x0 = copy.deepcopy(likelihood_func.pvec)
    x0_ref = copy.deepcopy(likelihood_func.pvec)

    def penalty(x):

        lower_diff = np.zeros_like(x)
        upper_diff = np.zeros_like(x)

        valids_lb = np.less(x, lb_bounds_list) # x < lb_bounds_list
        valids_ub = np.greater(x, ub_bounds_list) # x > ub_bounds_list
        if valids_lb.size > 0:
            lower_diff[valids_lb] = x[valids_lb] - lb_bounds_list[valids_lb]
        if valids_ub.size > 0:
            upper_diff[valids_ub] = x[valids_ub] - ub_bounds_list[valids_ub]

        ### Make sure we only use values from relevant
        ### force groups.
        if excluded_parameter_idx_list:
            lower_diff[excluded_parameter_idx_list] = 0.
            upper_diff[excluded_parameter_idx_list] = 0.

        penalty_val  = 0.
        penalty_val += bounds_penalty * np.sum(lower_diff**2)
        penalty_val += bounds_penalty * np.sum(upper_diff**2)

        return penalty_val

    def fun(x):

        ### Make sure we only use values from relevant
        ### force groups.
        if excluded_parameter_idx_list:
            x[excluded_parameter_idx_list] = x0_ref[excluded_parameter_idx_list]

        likelihood = likelihood_func(x)
        AIC_score  = 2. * N_parms_all * parm_penalty - 2. * likelihood

        return AIC_score

    def grad_penalty(x):

        grad = np.zeros_like(x)

        valids_lb = np.less(x, lb_bounds_list) # x < lb_bounds_list
        valids_ub = np.greater(x, ub_bounds_list) # x > ub_bounds_list
        if valids_lb.size > 0:
            lower_diff = x[valids_lb] - lb_bounds_list[valids_lb]
            grad[valids_lb] += bounds_penalty * 2. * lower_diff
        if valids_ub.size > 0:
            upper_diff = x[valids_ub] - ub_bounds_list[valids_ub]
            grad[valids_ub] += bounds_penalty * 2. * upper_diff

        ### Make sure we only use values from relevant
        ### force groups.
        if excluded_parameter_idx_list:
            grad[excluded_parameter_idx_list] = 0.

        return grad

    def grad(x):

        if excluded_parameter_idx_list:
            x[excluded_parameter_idx_list] = x0_ref[excluded_parameter_idx_list]

        _grad = likelihood_func.grad(x, use_jac=False)
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

        result = optimize.minimize(
            _fun, 
            x0, 
            jac = _grad, 
            method = "BFGS",
            options = {
                'gtol': 1e-6, 
                'disp': True
                },
            )
        best_x = result.x
        likelihood_func.apply_changes(best_x)
        best_f  = result.fun
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
    bitvec_type_list_list_cp = copy.deepcopy(bitvec_type_list_list)

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
        _typing_list):

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

    best_AIC = _calculate_AIC(
        pvec_list_cp,
        0,
        [pvec_list_cp[0].allocations[:].tolist()]
        )[0]

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
    for ast in worker_id_dict:
        _, _, type_ = ast

        type_i = type_[0]
        type_j = type_[1]

        worker_id = worker_id_dict[ast]
        _, _pvec_list, bitvec_type_all_list = ray.get(worker_id)

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
                    allocations = [0 for _ in pvec_list_cp[mngr_idx].allocations]
                    pvec_list_cp[mngr_idx].allocations[:] = allocations
                    pvec_list_cp[mngr_idx].reset(
                        _pvec_list[mngr_idx],
                        pvec_list_cp[mngr_idx].allocations
                        )
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
                        if allocations.count(-1) == 0:
                            pvec_list_cp[mngr_idx].allocations[:] = allocations
                            pvec_list_cp[mngr_idx].reset(
                                _pvec_list[mngr_idx],
                                pvec_list_cp[mngr_idx].allocations
                                )
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
                if allocations.count(-1) == 0:
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
                pvec_list[mngr_idx]
                )
            bitvec_type_list_list_cp[mngr_idx] = copy.deepcopy(
                bitvec_type_list_list[mngr_idx])

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
        verbose=False):

        from .targets import TargetComputer

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

        self.targetcomputer_id = ray.put(
            TargetComputer(
                self.system_list
            )
            )
        self.targetcomputer = ray.get(
            self.targetcomputer_id
            )

        self.grad_diff = 1.e-2

        ### Note, don't make this too small.
        ### Too small perturbations will lead to
        ### not finding the correct splits
        self.perturbation = 1.e-1

        self.verbose = verbose

        self.parm_mngr_cache_dict = dict()
        self.bsm_cache_dict  = dict()


    def add_parameters(
        self,
        parameter_manager,
        parameter_name_list = None,
        exclude_others = False,
        scale_list = None,
        bounds = None,
        bounds_penalty = 100.,
        ):

        from . import arrays

        if parameter_manager.N_systems != 0:
            raise ValueError(
                f"parameter_manager must be empty, but found {parameter_manager.N_systems} systems."
                )

        self.parameter_manager_list.append(parameter_manager)
        parm_mngr = self.generate_parameter_manager(self.N_mngr)

        ### Initially we want to figure out an initial type
        ### encoded as bitvector.
        from .tools import _remove_types
        _remove_types(
            parm_mngr,
            set_inactive = False
        )

        self.parameter_name_list.append(parameter_name_list)
        self.exclude_others.append(exclude_others)
        self.scaling_factor_list.append(scale_list)

        self.bounds_list.append(bounds)
        self.bounds_penalty_list.append(bounds_penalty)

        ### Must increment at the end, not before.
        self._N_mngr += 1

        bsm, _ = self.generate_bitsmartsmanager(
            self.N_mngr-1,
            max_neighbor=3
            )
        ### This adds [*:1]~[*:2] as initial bitvector
        self.best_bitvec_type_list.append(
            [0]
            )

        self.best_pvec_list.append(None)
        pvec_list, _ = self.generate_parameter_vectors(
            mngr_idx_list=[self.N_mngr-1]
            )
        self.best_pvec_list[-1] = pvec_list[0].copy()


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
                pvec.apply_changes()

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
            worker_id, worker_id_list = ray.wait(worker_id_list)
            worker_id = worker_id[0]
            _logP_likelihood = ray.get(worker_id)
            sys_idx = worker_id_dict[worker_id]
            if as_dict:
                logP_likelihood[sys_idx] = _logP_likelihood
            else:
                logP_likelihood += _logP_likelihood

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

        self.best_pvec_list[mngr_idx] = pvec.copy()
        self.best_bitvec_type_list[mngr_idx]  = copy.deepcopy(b_list)


    def generate_parameter_manager(
        self,
        mngr_idx,
        system_idx_list=list(),
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
        key = mngr_idx, tuple(_system_idx_list)
        if key in self.parm_mngr_cache_dict:
            parm_mngr = self.parm_mngr_cache_dict[key]
        else:
            parm_mngr = copy.deepcopy(
                self.parameter_manager_list[mngr_idx]
                )
            for sys_idx in _system_idx_list:
                parm_mngr.add_system(
                    self.system_list[sys_idx]
                    )
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
        ):

        system_idx_list = np.arange(
            self.N_systems, 
            dtype=int
            )
        np.random.shuffle(system_idx_list)
        res = self.N_systems%N_sys_per_batch
        N_batches = int((self.N_systems-res)/N_sys_per_batch)

        system_idx_list_batch = system_idx_list[:(self.N_systems-res)].reshape((N_batches, N_sys_per_batch)).tolist()
        if res > 0:
            system_idx_list_batch.append(system_idx_list[-res:])

        system_idx_list_batch = tuple(
            [tuple(sorted(sys_idx_pair)) for sys_idx_pair in system_idx_list_batch]
            )

        return system_idx_list_batch
    

class ForceFieldOptimizer(BaseOptimizer):

    def __init__(
        self, 
        system_list, 
        parm_penalty_split = 1.,
        parm_penalty_merge = 1.,
        name="ForceFieldOptimizer",
        verbose=False):

        super().__init__(system_list, name, verbose)

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


    def split_bitvector(
        self,
        mngr_idx,
        bitvec_type_list,
        system_idx_list,
        N_trials_gradient=10,
        split_all=False,
        max_on=0.1,
        max_splits=100,
        ):

        import arrays
        import numpy as np
        import copy
        import ray

        bsm, _ = self.generate_bitsmartsmanager(
            mngr_idx,
            system_idx_list
            )
        pvec_list, _ = self.generate_parameter_vectors(
            [mngr_idx],
            system_idx_list
            )
        pvec = pvec_list[0]
        N_types = pvec.force_group_count
        if split_all:
            type_query_list = list(range(N_types))
        else:
            type_query_list = [N_types-1]

        pvec_id = ray.put(pvec)
        worker_id_list = list()
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
            alloc  = np.zeros(pvec.allocations.size, dtype=int)
            for t_idx, _ in on_dict_sorted:
                b_new = bitvec_dict[t_idx]
                b_old = bitvec_type_list[type_i]
                if b_old == b_new:
                    continue
                check = (b_old == (b_old & b_new))
                if check:
                    selection_j = list(alloc_dict[t_idx])
                    ### We don't want to select all entries 
                    ### in the allocation vec
                    if len(selection_j) == pvec.allocations.size:
                        continue
                    if len(selection_j) == 0:
                        continue
                    alloc[:] = type_i
                    alloc[selection_j] = type_j
                    key   = tuple(alloc.tolist()), tuple(selection_i), tuple([type_i, type_j])
                    if key in alloc_bitvec_degeneracy_dict:
                        alloc_bitvec_degeneracy_dict[key].append(b_new)
                    else:
                        alloc_bitvec_degeneracy_dict[key] = [b_new]
                        k_values_ij[counts,:] = np.copy(alloc)
                        counts += 1
                if counts == max_splits:
                    break

            if counts < max_splits:
                max_splits_effective = counts
                k_values_ij = k_values_ij[:counts]
            else:
                max_splits_effective = max_splits

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
                    grad_diff = self.grad_diff,
                    )
                worker_id_list.append(worker_id)

        return worker_id_list, alloc_bitvec_degeneracy_dict


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
        worker_list,
        low_to_high=True,
        abs_grad_score=False,
        norm_cutoff=0.,
        keep_N_best=10,
        ):

        import numpy as np

        if len(worker_list) == 0:
            return list()

        votes_dict = dict()
        alloc_bitvec_dict = dict()
        for worker_id in worker_list:
            
            grad_score_dict, \
            grad_norm_dict, \
            allocation_list_dict, \
            selection_list_dict, \
            type_list_dict = ray.get(worker_id)

            score_dict = dict()
            N_trials = 0
            for trial_idx in grad_score_dict:

                grad_score = grad_score_dict[trial_idx]
                grad_norm  = grad_norm_dict[trial_idx]
                allocation_list = allocation_list_dict[trial_idx]
                selection_list  = selection_list_dict[trial_idx]
                type_list  = type_list_dict[trial_idx]

                N_trials += 1

                for g, n, a, s, t in zip(grad_score, grad_norm, allocation_list, selection_list, type_list):
                    ### Only use this split, when the norm
                    ### is larger than norm_cutoff
                    #print(
                    #    f"Gradient scores {g}",
                    #    f"Gradient norms {n}",
                    #    )

                    ast = a,s,t
                    if np.all(np.abs(n) < norm_cutoff):
                        pass
                    else:
                        ### Only consider the absolute value of the gradient score, not the sign
                        if abs_grad_score:
                            _g = [abs(gg) for gg in g]
                            g  = _g
                        if ast in score_dict:
                            score_dict[ast] += g
                        else:
                            score_dict[ast] = g

            for ast in score_dict:
                score_dict[ast] /= N_trials

            ### Sort dictionaries
            if low_to_high:
                ### For split: small values favored, i.e. small values first
                score_list  = [k for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=False)]
            else:
                ### For merge: high values favored, i.e. high values first
                score_list = [k for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)]

            ### Gather the three best (a,s,t) combinations
            ### and gather votes.
            ### Best one gets 3 votes, second 2 votes and third 1 vote
            for ast_i, ast in enumerate(score_list[:3]):
                if ast in votes_dict:
                    votes_dict[ast] += (3-ast_i)
                else:
                    votes_dict[ast] = (3-ast_i)

        ### Only keep the final best `keep_N_best` ast
        votes_list = [ast for ast, _ in sorted(votes_dict.items(), key=lambda items: items[1], reverse=True)]
        if len(votes_list) > keep_N_best:
            votes_list = votes_list[:keep_N_best]

        return votes_list


    def get_min_scores(
        self,
        mngr_idx,
        system_idx_list = list(),
        votes_split_list = list(),
        parallel_targets = False,
        ):

        use_scipy = True

        if len(system_idx_list) == 0:
            system_idx_list = list(range(self.N_systems))

        system_list_id = ray.put([self.system_list[sys_idx] for sys_idx in system_idx_list])
        worker_id_dict = dict()

        pvec_all, bitvec_type_list = self.generate_parameter_vectors(
            [],
            system_idx_list,
            )
        pvec_cp = pvec_all[mngr_idx].copy()
        N_types = pvec_all[mngr_idx].force_group_count

        ### Query split candidates
        ### ======================
        for counts, ast in enumerate(votes_split_list):
            allocation, selection, type_ = ast

            pvec_all[mngr_idx].duplicate(type_[0])
            pvec_all[mngr_idx].swap_types(N_types, type_[1])
            pvec_all[mngr_idx].allocations[:] = list(allocation)
            pvec_all[mngr_idx].apply_changes()

            worker_id = minimize_FF.remote(
                    system_list = system_list_id,
                    targetcomputer = self.targetcomputer_id,
                    pvec_list = pvec_all,
                    bitvec_type_list = bitvec_type_list,
                    bounds_list = self.bounds_list,
                    parm_penalty = self.parm_penalty_split,
                    pvec_idx_min = [mngr_idx],
                    parallel_targets = parallel_targets,
                    bounds_penalty=self.bounds_penalty_list[mngr_idx],
                    use_scipy = use_scipy,
                    verbose = self.verbose,
                    )

            worker_id_dict[ast] = worker_id
            pvec_all[mngr_idx].reset(pvec_cp)

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
        ### Optimize ordering in which systems are computed
        optimize_system_ordering=True,
        ### Keep the `keep_N_best` solutions for each split
        keep_N_best = 10,
        ### Prefix used for saving checkpoints
        prefix="ForceFieldOptimizer",
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

        #N_sys_per_batch_split = self.N_systems
        for iteration_idx in range(iterations):

            if iteration_idx > 0:
                if (iteration_idx%pair_incr_split) == 0:
                    N_sys_per_batch_split += 1
                    #N_sys_per_batch_split -= 1

            raute_fill = ''.ljust(len(str(iteration_idx))-1,"#")
            print("ITERATION", iteration_idx)
            print(f"###########{raute_fill}")

            ### ============== ###
            ### PROCESS SPLITS ###
            ### ============== ###
            split_iteration_idx = 0
            found_improvement   = True
            #while found_improvement and split_iteration_idx < max_splitting_attempts:
            while split_iteration_idx < max_splitting_attempts:
                print(f"ATTEMPTING SPLIT {iteration_idx}/{split_iteration_idx}")
                found_improvement       = False
                system_idx_list_batch   = self.get_random_system_idx_list(N_sys_per_batch_split)

                if optimize_system_ordering:
                    if self.verbose:
                        print(
                            "Initial ordering of systems:",
                            system_idx_list_batch
                            )
                        print(
                            "Optimizing system priority timings ..."
                            )

                    ### Compute average compute time for each `sys_idx_pair`
                    ### over `N_trails_opt` replicates.
                    N_trails_opt = 25
                    worker_id_dict = dict()
                    for sys_idx_pair in system_idx_list_batch:
                        worker_id_dict[sys_idx_pair] = list()
                        pvec_list, _ = self.generate_parameter_vectors([0], sys_idx_pair)
                        pvec = pvec_list[0]
                        for _ in range(N_trails_opt):
                            worker_id = _test_logL.remote(pvec, self.targetcomputer_id)
                            worker_id_dict[sys_idx_pair].append(worker_id)

                    time_diff_dict = dict()
                    for sys_idx_pair in system_idx_list_batch:
                        time_diff_dict[sys_idx_pair] = 0.
                        results = ray.get(worker_id_dict[sys_idx_pair])
                        time_diff_dict[sys_idx_pair] += sum(results)/N_trails_opt

                    system_idx_list_batch = [sys_idx_pair for sys_idx_pair, _ in sorted(time_diff_dict.items(), key=lambda items: items[1], reverse=True)]
                    system_idx_list_batch = tuple(system_idx_list_batch)
                    if self.verbose:
                        print(
                            "Optimized ordering of systems:",
                            system_idx_list_batch
                            )

                if self.verbose:
                    print(
                        "Generating splits and computing grad scores."
                        )
                split_worker_id_dict = dict()
                for mngr_idx in range(self.N_mngr):
                    for sys_idx_pair in system_idx_list_batch:
                        split_worker_id_dict[mngr_idx,sys_idx_pair] = self.split_bitvector(
                            mngr_idx,
                            self.best_bitvec_type_list[mngr_idx],
                            sys_idx_pair,
                            N_trials_gradient=N_trials_gradient,
                            split_all=True
                            )

                if self.verbose:
                    print(
                        "Obtaining votes and submitting parameter minimizations."
                        )
                worker_id_dict = dict()
                bitvec_dict = dict()
                for mngr_idx in range(self.N_mngr):
                    for sys_idx_pair in system_idx_list_batch:
                        votes_split_list = self.get_votes(
                            worker_list=split_worker_id_dict[mngr_idx,sys_idx_pair][0],
                            low_to_high=True,
                            abs_grad_score=False,
                            norm_cutoff=1.e-2,
                            keep_N_best=keep_N_best,
                            )
                        worker_id_dict[mngr_idx,sys_idx_pair] = self.get_min_scores(
                            mngr_idx=mngr_idx,
                            system_idx_list=sys_idx_pair,
                            votes_split_list=votes_split_list,
                            parallel_targets=False,
                            )
                        bitvec_dict[mngr_idx,sys_idx_pair] = dict()
                        for ast in votes_split_list:
                            b_list = split_worker_id_dict[mngr_idx,sys_idx_pair][1][ast]
                            bitvec_dict[mngr_idx,sys_idx_pair][ast] = b_list

                        if self.verbose:
                            print(
                                f"For mngr {mngr_idx} and systems {sys_idx_pair}:\n"
                                f"Found {len(votes_split_list)} candidate split solutions ...\n",
                                )

                if self.verbose:
                    print(
                        "Selecting best parameters."
                        )
                system_idx_list_batch = system_idx_list_batch[::-1]
                gibbs_dict = dict()
                gibbs_count_dict = dict()
                accepted_counter = 0
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
                for mngr_idx in range(self.N_mngr):
                    selection_worker_id_dict = dict()
                    for sys_idx_pair in system_idx_list_batch:
                        worker_id = set_parameters_remote.remote(
                            mngr_idx_main = mngr_idx,
                            pvec_list = old_pvec_list,
                            targetcomputer = self.targetcomputer_id,
                            bitvec_dict = bitvec_dict[mngr_idx, sys_idx_pair],
                            bitvec_alloc_dict_list = bitvec_alloc_dict_list,
                            bitvec_type_list_list = old_bitvec_type_list,
                            worker_id_dict = worker_id_dict[mngr_idx,sys_idx_pair],
                            parm_penalty = self.parm_penalty_split,
                            verbose = self.verbose,
                            )
                        selection_worker_id_dict[sys_idx_pair] = worker_id

                    for sys_idx_pair in selection_worker_id_dict:
                        if self.verbose:
                            print("mngr_idx/sys_idx_pair", mngr_idx, "/", sys_idx_pair)

                        worker_id = selection_worker_id_dict[sys_idx_pair]
                        _, pvec_list, best_bitvec_type_list, new_AIC = ray.get(worker_id)
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
                        with open(f"{prefix}-MAIN-{iteration_idx}-SPLIT-{split_iteration_idx}-ACCEPTED-{accepted_counter}.pickle", "wb") as fopen:
                            pickle.dump(
                                self,
                                fopen
                            )
                        accepted_counter += 1

                split_iteration_idx += 1
                
            ### ========================== ###
            ### DRAW FROM TYPING POSTERIOR ###
            ### ========================== ###
            mngr_schedule = np.arange(self.N_mngr, dtype=int)
            np.random.shuffle(mngr_schedule)
            for mngr_idx in mngr_schedule:
                if self.verbose:
                    print(
                        f"Drawing types from typing posterior for mngr {mngr_idx}"
                        )
                pvec_list, bitvec_list = self.generate_parameter_vectors([mngr_idx])
                bsm, bitvec_alloc_dict_list = self.generate_bitsmartsmanager(mngr_idx)
                bitvec_list_new = draw_bitvector_from_candidate_list(
                    pvec_list[0], 
                    bitvec_list[0], 
                    bsm,
                    self.targetcomputer,
                    theta=1000.,
                    alpha=10.,
                    N_iter = max_gibbs_attempts,
                    max_on=0.1,
                    verbose = self.verbose
                )
                allocations = [-1 for _ in pvec_list[0].allocations]
                bitvec_hierarchy_to_allocations(
                    bitvec_alloc_dict_list,
                    bitvec_list_new,
                    allocations
                    )
                if allocations.count(-1) == 0:
                    pvec_list[0].allocations[:] = allocations
                    pvec_list[0].apply_changes()
                    self.update_best(
                        mngr_idx,
                        pvec_list[0],
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
            with open(f"{prefix}-MAIN-{iteration_idx}-GARBAGE_COLLECTION.pickle", "wb") as fopen:
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
            with open(f"{prefix}-MAIN-{iteration_idx}.pickle", "wb") as fopen:
                pickle.dump(
                    self,
                    fopen
                )
        ### END LOOP OVER ITERATIONS

