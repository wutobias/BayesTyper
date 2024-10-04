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
                        _INACTIVE_GROUP_IDX,
                        _EPSILON
                        )

import ray
from .optimizers import BaseOptimizer


def calc_parameter_log_prior(
    pvec_list,
    bounds_list = None,
    sigma_parameter_list = None,
    parameter_prior="gaussian",
    ):

    import numpy as np
    from scipy import stats

    choices_parameter_prior = [
            "gaussian",
            "jeffreys"
            ]
    parameter_prior = parameter_prior.lower()

    if not parameter_prior in choices_parameter_prior:
        raise ValueError(
            f"Typing prior must be one of {choices_parameter_prior}"
            )

    N_pvec = len(pvec_list)

    if isinstance(bounds_list, type(None)):
        bounds_list = [None for _ in range(N_pvec)]
    if isinstance(sigma_parameter_list, type(None)):
        sigma_parameter_list = [None for _ in range(N_pvec)]

    log_prior_val = 0.
    if parameter_prior == "gaussian":
        for pvec_idx in range(N_pvec):

            pvec = pvec_list[pvec_idx]
            if pvec.force_group_count == 0:
                continue

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
                    rdmol = pvec.parameter_manager.system_list[sys_idx].rdmol
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
                    _lb = [bounds.lower]
                    _ub = [bounds.upper]
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
                ).astype(np.float64)
            ub = np.array(
                pvec.get_transform(ub)
                ).astype(np.float64)

            mu_list  = (lb + ub)/2.
            sig_list = lb - ub
            sig_list = np.abs(sig_list)
            x_val    = np.array(pvec[:], dtype=float)

            prior_distr_val = stats.norm(mu_list, sig_list)
            log_prior_val  += np.sum(
                prior_distr_val.logpdf(x_val)
                )
    elif parameter_prior == "jeffreys":
        for pvec_idx in range(N_pvec):
            pvec = pvec_list[pvec_idx]
            if pvec.force_group_count == 0:
                continue
            bounds = sigma_parameter_list[pvec_idx]
            log_prior_val -= pvec.force_group_count * np.log(bounds)

    else:
        raise ValueError(
            f"Prior {parameter_prior} not known."
            )

    if np.isinf(log_prior_val) or np.isnan(log_prior_val):
        return -np.inf
    else:
        return log_prior_val


def calc_log_prior(
    pvec_list,
    bounds_list = None,
    sigma_parameter_list = None,
    parameter_prior = "gaussian",
    ):

    log_prior_val = calc_parameter_log_prior(
        pvec_list,
        bounds_list = bounds_list,
        sigma_parameter_list = sigma_parameter_list,
        parameter_prior = parameter_prior,
        )

    return log_prior_val


class ForceFieldSampler(BaseOptimizer):

    def __init__(
        self, 
        system_list, 
        name = "ForceFieldSampler",
        verbose=False):

        super().__init__(system_list, name, verbose)

        self.last_step_accepted = True
        self.N_acc = 0
        self.N_all = 0

        self.last_step_likelihood = 0.
        self.last_step_prior = 0.
        self.last_step_posterior = 0.

        self.use_parameter_prior = True
        self.use_typing_prior    = True

        self.split_size = 0

        self.use_jac = False


    @property
    def jacobian_masses(self):
        return not self.use_jac


    @jacobian_masses.setter
    def jacobian_masses(self, value):
        self.use_jac = not value


    def _reload_pvec(self):
        self.pvec_list_all, _ = self.generate_parameter_vectors()


    def calc_log_prior(
        self,
        mngr_idx_list = list()
        ):

        import numpy as np

        if len(mngr_idx_list) == 0:
            _mngr_idx_list = list(range(self.N_mngr))
        else:
            _mngr_idx_list = mngr_idx_list

        pvec_list = [self.pvec_list_all[mngr_idx] for mngr_idx in _mngr_idx_list]

        alpha_list = [
            1.e+3 * np.ones(pvec.force_group_count, dtype=float) for pvec in pvec_list
            ]
        weight_list = [
            1.e+3 * np.ones(pvec.force_group_count, dtype=float) for pvec in pvec_list
            ]
        bonds_list = [
            self.bounds_list[mngr_idx] for mngr_idx in _mngr_idx_list
            ]

        log_prior_val = calc_log_prior(
            pvec_list,
            bounds_list = bonds_list,
            sigma_parameter_list = None,
            parameter_prior = "gaussian",
            )

        return log_prior_val


    def draw_parameter_vector(
        self,
        mngr_idx,
        max_on=0.1,
        theta=1.,
        alpha=1.,
        ):

        """
        RJMC with lifting, then proposing. We use
        metropolis-adjusted Langevin sampling to propagate
        the parameters.
        """

        from scipy import stats
        import numpy as np
        from . import draw_typing_vector, typing_prior
        from .bitvector_typing import bitvec_hierarchy_to_allocations

        normalize_gradients = False

        sig_langevin  = 5.e-4
        sig_symm = 1.e-1

        sig_langevin2 = sig_langevin**2
        sig_symm2 = sig_symm**2

        ### Randomly decide if we should split
        ### death, or propage.
        ### 0: split
        ### 1: death
        ### 2: propagate
        kernel_p  = [1.,1.,2.]
        kernel_p /= np.sum(kernel_p)
        kernel_choice = np.random.choice(
            [0,1,2],
            p=kernel_p
            )
        split = False
        death = False
        propagate = False
        if kernel_choice == 0:
            split = True
        elif kernel_choice == 1:
            death = True
        elif kernel_choice == 2:
            propagate = True
        else:
            raise ValueError(
                "No idea what kernel to use."
                )

        norm_distr_zero_centered_langevin = stats.norm(0, sig_langevin)
        norm_distr_zero_centered_symm = stats.norm(0, sig_symm)

        log_P_old = self.calc_log_likelihood() + np.sum(self.calc_log_prior())

        pvec_list, bitvec_list_list = self.generate_parameter_vectors([mngr_idx])
        bsm, bitvec_list_alloc_dict = self.generate_bitsmartsmanager(mngr_idx)
        pvec = pvec_list[0]
        N_types = pvec.force_group_count
        bitvec_type_list = bitvec_list_list[0]

        if split:
            if self.verbose:
                print("split ...")

            likelihood_func = LikelihoodVectorized(
                [pvec],
                self.targetcomputer
                )

            logL_old = likelihood_func()                
            logP_prior_old = 0.
            for _mngr_idx in range(self.N_mngr):
                if mngr_idx == _mngr_idx:
                   _bitvec_type_list = bitvec_type_list
                   _bsm = bsm
                else:
                    _, _bitvec_type_list = self.generate_parameter_vectors(_mngr_idx)
                    _bsm, _ = self.generate_bitsmartsmanager(_mngr_idx)
                for _type_i, _b in enumerate(_bitvec_type_list):
                    logP_prior_old += typing_prior(
                                    _b, _bsm, 
                                    _type_i, 
                                    theta, alpha
                                    )
            logP_prior_old += self.calc_log_prior()
            log_P_old = logL_old + logP_prior_old
            
            ### STEP 1:
            ### Figure out the type we want to select for lifting.
            ### This is done by computing gradient scores and treating
            ### them as probabilities. In addition, we use a prior on
            ### the bitvectors to enable lifting to less specific bitvector
            ### spaces.

            ### Compute gradient scores.
            worker_id_list, alloc_bitvec_degeneracy_dict = self.split_bitvector(
                mngr_idx,
                bitvec_type_list,
                [i for i in range(self.N_systems)]
                )

            ast_logP_dict = dict()
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

                for ast in score_dict:
                    _, _, type_ = ast
                    type_i = type_[0]
                    for b in alloc_bitvec_degeneracy_dict[ast]:
                        bitvec_type_list.insert(type_i, b)
                        allocations = [-1 for _ in pvec.allocations]
                        bitvec_hierarchy_to_allocations(
                            bitvec_list_alloc_dict,
                            bitvec_type_list,
                            allocations,
                            )
                        ### If true, this bitvec is valid
                        if allocations.count(-1) == 0:
                            logP_prior = typing_prior(
                                b, bsm, type_i, theta, alpha
                                )
                            ast_logP_dict[(ast,b)] = np.logaddexp(
                                logP_prior, np.log(score_dict[ast])
                                )
                        bitvec_type_list.pop(type_i)

            logP_sum = 0.
            for i, ast_b in enumerate(ast_logP_dict.keys()):
                logp = ast_logP_dict[ast_b]
                if i == 0:
                    logP_sum = logp
                else:
                    logP_sum = np.logaddexp(logp, logP_sum)
            P_values = list()
            ast_b_keys = list()
            for ast_b in ast_prob_dict:
                ast_logP_dict[ast_b] -= logP_sum
                ast_b_keys.append(ast_b)
                P_values.append(ast_logP_dict[ast_b])
            ast_b_keys = np.array(ast_b_keys, dtype=object)
            P_values = np.array(P_values, dtype=float)
            new_ast_b  = np.random.choice(ast_b_keys, p=P_values)
            logP_type_selection = ast_logP_dict[new_ast_b]
            ast, b_new = new_ast_b
            _, _, type_ = ast
            type_i = type_[0]
            type_j = type_[1]
            bitvec_type_list.insert(type_i, b_new)
            allocations = [-1 for _ in pvec.allocations]
            bitvec_hierarchy_to_allocations(
                bitvec_list_alloc_dict,
                bitvec_type_list,
                allocations,
                )
            pvec.duplicate(type_i)
            pvec.swap_types(N_types, type_j)
            pvec.allocations[:] = allocations
            pvec.apply_changes()


            ### STEP 2: We propose a new parameter vector
            ### in the lifted space.
            likelihood_func = LikelihoodVectorized(
                [pvec],
                self.targetcomputer
                )
            x0 = likelihood_func.pvec

            grad_fwd = likelihood_func.grad(
                x0[:], 
                grad_diff=self.grad_diff,
                use_jac=self.use_jac
                )
            grad_fwd_norm = np.linalg.norm(grad_fwd)
            if normalize_gradients:
                if grad_fwd_norm > 0.:
                    grad_fwd /= grad_fwd_norm

            W  = norm_distr_zero_centered_langevin.rvs(x0.size)
            if self.jacobian_masses:
                W *= np.ones_like(W) / np.array(likelihood_func.jacobian)**2
            x1 = x0 + 0.5 * sig_langevin2 * grad_fwd + sig_langevin2 * W
            grad_bkw = likelihood_func.grad(
                x1[:], 
                grad_diff=self.grad_diff,
                use_jac=self.use_jac
                )
            grad_bkw_norm = np.linalg.norm(grad_bkw)
            if normalize_gradients:
                if grad_bkw_norm > 0.:
                    grad_bkw /= grad_bkw_norm
            pvec.apply_changes()
            likelihood_func.apply_changes(x1)

            logL_new = likelihood_func()
            logP_prior_new = 0.
            for _mngr_idx in range(self.N_mngr):
                if mngr_idx == _mngr_idx:
                   _bitvec_type_list = bitvec_type_list
                   _bsm = bsm
                else:
                    _, _bitvec_type_list = self.generate_parameter_vectors(_mngr_idx)
                    _bsm, _ = self.generate_bitsmartsmanager(_mngr_idx)
                for _type_i, _b in enumerate(_bitvec_type_list):
                    logP_prior_new += typing_prior(
                                    _b, _bsm, 
                                    _type_i, 
                                    theta, alpha
                                    )
            logP_prior_new += self.calc_log_prior()
            log_P_new = logL_new + logP_prior_new

            fwd_diff = x1 - x0 - 0.5 * sig_langevin2 * grad_fwd
            bkw_diff = x0 - x1 - 0.5 * sig_langevin2 * grad_bkw

            ### Note the prefactors are omitted since they are the
            ### same in fwd and bkw.
            logQ_fwd = np.sum(fwd_diff**2)
            logQ_bkw = np.sum(bkw_diff**2)

            log_alpha  = log_P_new - log_P_old
            log_alpha += logQ_bkw  - logQ_fwd
            log_alpha -= logP_type_selection
            log_alpha -= np.log(N_types+1.)
            log_alpha -= np.log(N_types)

            if self.verbose:
                sma = bsm.bitvector_to_smarts(b_new)
                print(
                    "mngr:", mngr_idx,
                    "type i:", type_i,
                    "type j:", type_j,
                )
                for idx, b in enumerate(bitvec_type_list):
                    sma = bsm.bitvector_to_smarts(b)
                    print(
                        f"Type {idx}({pvec.allocations.count(idx)})", sma
                    )

                print(
                    "log_alpha", log_alpha, "\n",
                    "log_P_new", log_P_new,  "\n",
                    "log_P_old", log_P_old,  "\n",
                    "logQ_bkw,", logQ_bkw,  "\n",
                    "logQ_fwd", logQ_fwd, "\n",
                    "log(type_selection)", logP_type_selection, "\n",
                    )

        elif propagate:
            if self.verbose:
                print("propagate ...")

            likelihood_func = LikelihoodVectorized(
                [pvec],
                self.targetcomputer
                )

            logL_old = likelihood_func()
            logP_prior_old = 0.
            for _mngr_idx in range(self.N_mngr):
                if mngr_idx == _mngr_idx:
                   _bitvec_type_list = bitvec_type_list
                   _bsm = bsm
                else:
                    _, _bitvec_type_list = self.generate_parameter_vectors(_mngr_idx)
                    _bsm, _ = self.generate_bitsmartsmanager(_mngr_idx)
                for _type_i, _b in enumerate(_bitvec_type_list):
                    logP_prior_old += typing_prior(
                                    _b, _bsm, 
                                    _type_i, 
                                    theta, alpha
                                    )
            logP_prior_old += self.calc_log_prior()
            log_P_old = logL_old + logP_prior_old

            x0 = likelihood_func.pvec
            grad_fwd = likelihood_func.grad(
                x0[:], 
                grad_diff=self.grad_diff,
                use_jac=self.use_jac
                )
            grad_fwd_norm = np.linalg.norm(grad_fwd)
            if normalize_gradients:
                if grad_fwd_norm > 0.:
                    grad_fwd /= grad_fwd_norm

            W  = norm_distr_zero_centered_langevin.rvs(x0.size)
            if self.jacobian_masses:
                W *= np.ones_like(W) / np.array(likelihood_func.jacobian)**2
            x1 = x0 + 0.5 * sig_langevin2 * grad_fwd + sig_langevin2 * W
            grad_bkw = likelihood_func.grad(
                x1[:], 
                grad_diff=self.grad_diff,
                use_jac=self.use_jac
                )
            grad_bkw_norm = np.linalg.norm(grad_bkw)
            if normalize_gradients:
                if grad_bkw_norm > 0.:
                    grad_bkw /= grad_bkw_norm
            pvec.apply_changes()
            likelihood_func.apply_changes(x1)

            logL_new = likelihood_func()
            logP_prior_new = 0.
            for _mngr_idx in range(self.N_mngr):
                if mngr_idx == _mngr_idx:
                   _bitvec_type_list = bitvec_type_list
                   _bsm = bsm
                else:
                    _, _bitvec_type_list = self.generate_parameter_vectors(_mngr_idx)
                    _bsm, _ = self.generate_bitsmartsmanager(_mngr_idx)
                for _type_i, _b in enumerate(_bitvec_type_list):
                    logP_prior_new += typing_prior(
                                    _b, _bsm, 
                                    _type_i, 
                                    theta, alpha
                                    )
            logP_prior_new += self.calc_log_prior()
            log_P_new = logL_new + logP_prior_new

            fwd_diff = x1 - x0 - 0.5 * sig_langevin2 * grad_fwd
            bkw_diff = x0 - x1 - 0.5 * sig_langevin2 * grad_bkw

            ### Note the prefactors are omitted since they are the
            ### same in fwd and bkw.
            logQ_fwd = np.sum(fwd_diff**2)
            logQ_bkw = np.sum(bkw_diff**2)

            log_alpha  = log_P_new - log_P_old
            log_alpha += logQ_bkw  - logQ_fwd

            if self.verbose:
                sma = bsm.bitvector_to_smarts(b_new)
                print(
                    "mngr:", mngr_idx,
                    "type i:", type_i,
                    "type j:", type_j,
                )
                for idx, b in enumerate(bitvec_type_list):
                    sma = bsm.bitvector_to_smarts(b)
                    print(
                        f"Type {idx}({pvec.allocations.count(idx)})", sma
                    )

                print(
                    "log_alpha", log_alpha, "\n",
                    "log_P_new", log_P_new,  "\n",
                    "log_P_old", log_P_old,  "\n",
                    "logQ_bkw,", logQ_bkw,  "\n",
                    "logQ_fwd", logQ_fwd, "\n",
                    )

        #elif death:
            

        is_accepted = False
        if np.isinf(log_alpha):
            is_accepted = False
        else:
            ### Means alpha > 1.
            if log_alpha > 0.:
                is_accepted = True
            else:
                u = np.random.random()
                log_u = np.log(u)
                if log_u < log_alpha:
                    is_accepted = True
                else:
                    is_accepted = False

        if not is_accepted:
            for mngr_idx in range(self.N_mngr):
                self.pvec_list_all[mngr_idx].reset(
                    pvec_list_old[mngr_idx],
                    pvec_list_old[mngr_idx].allocations
                    )
                self.pvec_list_all[mngr_idx].apply_changes()

            self.last_step_likelihood = self.calc_log_likelihood()
            self.last_step_prior = np.sum(self.calc_log_prior())
            self.last_step_posterior = self.last_step_likelihood + self.last_step_prior

        if is_accepted:
            self.last_step_accepted = True
            print("Accepted.")
            return alloc_omit_list
        else:
            self.last_step_accepted = False
            print("Rejected.")
            alloc_omit_list = [list() for _ in range(self.N_mngr)]
            return alloc_omit_list

    def run(
        self,
        iterations=100
        ):

        self.pvec_list_all = self.generate_parameter_vectors(
            mngr_idx_list=[], 
            system_idx_list=[]
            )

        if self.N_all == 0:
            self.last_step_accepted = True

            ct = ComputeTarget(
                self.system_list, 
                parallel_targets=True
                )
            likelihood_func = LikelihoodVectorized(ct)

            self.last_step_likelihood = likelihood_func()
            self.last_step_prior = np.sum(self.calc_log_prior())
            self.last_step_posterior = self.last_step_likelihood + self.last_step_prior
            self.N_acc = 0
            self.N_all = 0

            pvec_list_cp = copy.deepcopy(self.pvec_list_all)
            for pvec in pvec_list_cp:
                print(
                    pvec.allocations,
                    pvec.vector_k,
                    )
                del pvec.parameter_manager.system_list

            self.pvec_traj.append(
                pvec_list_cp
                )

        alloc_omit_list = [list() for _ in range(self.N_mngr)]
        for _ in range(iterations):
            print()
            print(
                f"ITERATION: {self.N_all}"
                )

            print("Re-Type ...")

            self.draw_typing_vector(
                alloc_omit_list=alloc_omit_list
                )

#            try:
#                self.draw_typing_vector(
#                    alloc_omit_list=alloc_omit_list
#                    )
#            except Exception as e:
#                import warnings
#                warnings.warn(
#                    f"Error drawing typing vector at step {self.N_all} with {e}"
#                    )
#                for mngr_idx in range(self.N_mngr):
#                    self.pvec_list_all[mngr_idx].reset(
#                        self.pvec_traj[-1][mngr_idx],
#                        self.pvec_traj[-1][mngr_idx].allocations
#                        )
#                    self.pvec_list_all[mngr_idx].apply_changes()
#                continue
            
            alloc_omit_list = self.draw_parameter_vector()
#            try:
#                alloc_omit_list = self.draw_parameter_vector()
#            except Exception as e:
#                import warnings
#                warnings.warn(
#                    f"Error drawing parameter vector at step {self.N_all} with {e}"
#                    )
#                for mngr_idx in range(self.N_mngr):
#                    self.pvec_list_all[mngr_idx].reset(
#                        self.pvec_traj[-1][mngr_idx],
#                        self.pvec_traj[-1][mngr_idx].allocations
#                        )
#                    self.pvec_list_all[mngr_idx].apply_changes()
#                continue
            
            pvec_list_cp = copy.deepcopy(self.pvec_list_all)
            for pvec in pvec_list_cp:
                print(
                    pvec.allocations,
                    pvec.vector_k,
                    )
                del pvec.parameter_manager.system_list

            self.pvec_traj.append(
                pvec_list_cp
                )
            if self.last_step_accepted:
                self.N_acc += 1
            self.N_all += 1
