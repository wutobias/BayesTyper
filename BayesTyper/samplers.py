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
        mngr_idx
        ):

        """
        RJMC with lifting, then proposing. We use
        metropolis-adjusted Langevin sampling to propagate
        the parameters.
        """

        from scipy import stats
        import numpy as np
        from . import draw_typing_vector, typing_prior

        normalize_gradients = False

        sig_langevin  = 5.e-4
        sig_symm = 1.e-1

        sig_langevin2 = sig_langevin**2
        sig_symm2 = sig_symm**2

        ### Randomly decide if we should split
        ### merge, or propage.
        ### 0: split
        ### 1: merge
        ### 2: propagate
        kernel_p  = [1.,1.,2.]
        kernel_p /= np.sum(kernel_p)
        kernel_choice = np.random.choice(
            [0,1,2],
            p=kernel_p
            )
        split = False
        merge = False
        propagate = False
        if kernel_choice == 0:
            split = True
        elif kernel_choice == 1:
            merge = True
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
        pvec = pvec_list[0]
        bitvec_list = bitvec_list_list[0]

        if split:
            print("split ...")
            ### First do the lifting move.
            ### Randomly select one parameter type

            N_types   = pvec.force_group_count
            type_i = np.random.randint(0, N_types)
            type_j = type_i + 1

            pvec.duplicate(type_i)
            pvec.swap_types(N_types, type_j)
            pvec.apply_changes()

            selection_i = list()
            for i, a in enumerate(pvec.allocations):
                if a == type_i:
                    selection_i.append(i)

            _, bitvec_alloc_dict_list = bsm.prepare_bitvectors()

            bsm.and_rows(
                allocations=selection_i
                )

            ### Split the parameter type randomly
            alloc     = pvec.allocations.index([type_i])[0]
            ### Note: We canot use `pvec.force_group_histogram` here, since it
            ###       will not contain counts from `_INACTIVE_GROUP_IDX`
            N_bonds_i = alloc.size

            ### Case if we consider all possible splits
            if self.split_size == 0:
                pvec.allocations[alloc] = np.random.choice(
                    [type_i, pvec.force_group_count-1],
                    size=alloc.size,
                    replace=True
                    )
                alloc_omit_list[pvec_idx] = alloc
                N_possibilities = 2**N_bonds_i - 2
            ### Case if we consider only splits of size `self.split_size`
            else:
                from scipy import special
                _split_size = self.split_size
                if self.split_size >= N_bonds_i:
                    _split_size = N_bonds_i
                alloc_new = np.random.choice(
                    alloc,
                    size=_split_size,
                    replace=True
                    )
                pvec.allocations[alloc_new] = pvec.force_group_count-1
                ### Must still write old `alloc` in `alloc_omit_list` here.
                alloc_omit_list[pvec_idx] = alloc
                N_possibilities = special.binom(
                    N_bonds_i, 
                    _split_size
                    )
            pvec.apply_changes()
            d_new = norm_distr_zero_centered_symm.rvs(
                pvec.parameters_per_force_group
                )
            first_parm = pvec.parameters_per_force_group * (
                pvec.force_group_count - 1
                )
            last_parm  = first_parm + pvec.parameters_per_force_group
            pvec[first_parm:last_parm] += d_new
            pvec.apply_changes()

            alpha_list  = [1.e+3 * np.ones(pvec.force_group_count, dtype=float)]
            weight_list = [1.e+3 * np.ones(pvec.force_group_count, dtype=float)]

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

            likelihood_func.apply_changes(x1)
            log_P_new = self.calc_log_likelihood() + np.sum(self.calc_log_prior())

            fwd_diff = x1 - x0 - 0.5 * sig_langevin2 * grad_fwd
            bkw_diff = x0 - x1 - 0.5 * sig_langevin2 * grad_bkw

            ### Note the prefactors are omitted since they are the
            ### same in fwd and bkw.
            logQ_fwd = np.sum(fwd_diff**2) + norm_distr_zero_centered_symm.logpdf(d_new).sum()
            logQ_bkw = np.sum(bkw_diff**2)

            log_alpha  = log_P_new - log_P_old
            log_alpha += logQ_bkw  - logQ_fwd
            log_alpha += np.log(N_possibilities)
            log_alpha -= np.log(N_types+1.)
            log_alpha -= np.log(N_types)

            print(
                "Mngr"                        , pvec_idx, "\n",
                "Allocations"                 , pvec.allocations, "\n",
                "log_alpha"                   , log_alpha, "\n",
                "log_P_new"                   , log_P_new,  "\n",
                "-log_P_old"                  , -log_P_old,  "\n",
                "logQ_bkw,"                   , logQ_bkw,  "\n",
                "-logQ_fwd"                   , -logQ_fwd, "\n",
                "log(N_possibilities)"        , np.log(N_possibilities), "\n",
                "-log(N_types)-log(N_types+1)", -np.log(N_types), -np.log(N_types+1.), "\n",
                )

        elif merge:

            print("merge ...")
            pvec_idx = np.random.choice(
                np.arange(self.N_mngr)
                )
            pvec = self.pvec_list_all[pvec_idx]
            sele = np.arange(
                pvec.force_group_count,
                dtype=int
                )
            if death_move:
                _sele = np.append(sele, _INACTIVE_GROUP_IDX)
                sele = _sele
            ### type_i: The type that type_j is merged into
            ### type_j: The type we want to eliminate
            type_i, type_j = np.random.choice(
                sele,
                size=2,
                replace=False
                )
            ### If type_j is `_INACTIVE_GROUP_IDX`, we must
            ### swap type_i and type_j. The inactive type cannot
            ### be eliminated.
            if type_j == _INACTIVE_GROUP_IDX:
                type_j = type_i
                type_i = _INACTIVE_GROUP_IDX

            ### First do the propagation
            x0 = likelihood_func.pvec
            likelihood_func = LikelihoodVectorized(
                [pvec],
                self.targetcomputer
                )

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
            x1 = x0 + 0.5 * sig_langevin2 * grad_fwd + sig_langevin2 * W

            ### Then compute backwards propability.
            grad_bkw = likelihood_func.grad(
                x1, 
                grad_diff=self.grad_diff,
                use_jac=self.use_jac
                )
            grad_bkw_norm = np.linalg.norm(grad_bkw)
            if normalize_gradients:
                if grad_bkw_norm > 0.:
                    grad_bkw /= grad_bkw_norm

            fwd_diff = x1 - x0 - 0.5 * sig_langevin2 * grad_fwd
            bkw_diff = x0 - x1 - 0.5 * sig_langevin2 * grad_bkw

            ### Then do the lifting (i.e. contraction) move.
            ### Randomly select two of the parameter types
            N_types   = pvec.force_group_count
            N_bonds_i = pvec.allocations.index([type_i])[0].size
            N_bonds_j = pvec.allocations.index([type_j])[0].size

            if type_i == _INACTIVE_GROUP_IDX:
                value_i      = [0. * _FORCE_CONSTANT_TORSION for _ in range(pvec.parameters_per_force_group)]
            else:
                first_parm_i = pvec.parameters_per_force_group * type_i
                last_parm_i  = first_parm_i + pvec.parameters_per_force_group
                value_i      = pvec.vector_k[first_parm_i:last_parm_i]
            first_parm_j = pvec.parameters_per_force_group * type_j
            last_parm_j  = first_parm_j + pvec.parameters_per_force_group
            value_j      = pvec.vector_k[first_parm_j:last_parm_j]

            ### Important. Compute the difference using the transformation
            ### on type_j (would also work with transformation on type_i).
            real_cp    = copy.deepcopy(pvec.vector_k)
            real_cp[first_parm_j:last_parm_j] = value_j
            d_old      = pvec.get_transform(real_cp)[first_parm_j:last_parm_j]
            real_cp[first_parm_j:last_parm_j] = value_i
            d_old     -= pvec.get_transform(real_cp)[first_parm_j:last_parm_j]
            d_old      = np.array(d_old, dtype=float)

            alloc_j = pvec.allocations.index([type_j])[0]

            pvec.allocations[alloc_j] = type_i
            pvec.apply_changes()
            pvec.remove(type_j)
            pvec.apply_changes()

            alloc_omit_list[pvec_idx] = alloc_j

            likelihood_func.apply_changes(x1)
            log_P_new = self.calc_log_likelihood() + np.sum(self.calc_log_prior())

            ### Note the prefactors are omitted since they are the
            ### same in fwd and bkw.
            logQ_fwd = np.sum(fwd_diff**2)
            logQ_bkw = np.sum(bkw_diff**2) + norm_distr_zero_centered_symm.logpdf(d_old).sum()

            if self.split_size == 0:
                N_possibilities = 2**N_bonds_i - 2
            ### Case if we consider only splits of size `self.split_size`
            else:
                from scipy import special
                _split_size = self.split_size
                if self.split_size >= N_bonds_i:
                    _split_size = N_bonds_i
                N_possibilities = special.binom(
                    N_bonds_i, 
                    _split_size
                    )

            log_alpha  = log_P_new - log_P_old
            log_alpha += logQ_bkw  - logQ_fwd
            log_alpha += np.log(N_types)
            log_alpha += np.log(N_types-1.)
            log_alpha -= np.log(N_possibilities)

            print(
                "Mngr"                        , pvec_idx, "\n",
                "Allocations"                 , pvec.allocations, "\n",
                "log_alpha"                   , log_alpha, "\n",
                "log_P_new"                   , log_P_new,  "\n",
                "-log_P_old"                  , -log_P_old,  "\n",
                "logQ_bkw"                    , logQ_bkw,  "\n",
                "-logQ_fwd"                   , -logQ_fwd, "\n",
                "log(N_types)+log(N_types-1)" , np.log(N_types), np.log(N_types-1.), "\n",
                "-log(N_possibilities)"       , -np.log(N_possibilities), "\n",
                )

        elif propagate:
            print("propagate ...")
            pvec_idx = np.random.choice(
                np.arange(self.N_mngr)
                )
            pvec = self.pvec_list_all[pvec_idx]
            ### First do the propagation
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
            x1 = x0 + 0.5 * sig_langevin2 * grad_fwd + sig_langevin2 * W

            ### Then compute backwards propability.
            grad_bkw = likelihood_func.grad(
                x1[:], 
                grad_diff=self.grad_diff,
                use_jac=self.use_jac
                )
            grad_bkw_norm = np.linalg.norm(grad_bkw)
            if normalize_gradients:
                if grad_bkw_norm > 0.:
                    grad_bkw /= grad_bkw_norm

            fwd_diff = x1 - x0 - 0.5 * sig_langevin2 * grad_fwd
            bkw_diff = x0 - x1 - 0.5 * sig_langevin2 * grad_bkw

            likelihood_func.apply_changes(x1)
            log_P_new = self.calc_log_likelihood() + np.sum(self.calc_log_prior())

            ### Note the prefactors are omitted since they are the
            ### same in fwd and bkw.
            logQ_fwd = np.sum(fwd_diff**2)
            logQ_bkw = np.sum(bkw_diff**2)

            log_alpha  = log_P_new - log_P_old
            log_alpha += logQ_bkw  - logQ_fwd

            print(
                "delta"      , x1 - x0, "\n",
                "log_alpha"  , log_alpha, "\n",
                "log_P_new"  , log_P_new,  "\n",
                "-log_P_old" , -log_P_old,  "\n",
                "logQ_bkw"   , logQ_bkw,  "\n",
                "-logQ_fwd"  , -logQ_fwd, "\n",
                )

        elif death:

            print("death ...")
            ### Find empty types
            pvec_idx = np.random.choice(
                np.arange(self.N_mngr)
                )
            if self.pvec_list_all[pvec_idx].force_group_count == 0:
                print("Rejected.")
                return 
            pvec = self.pvec_list_all[pvec_idx]
            ### This is the type we want to duplicate from
            valids_empty = np.where(pvec.force_group_histogram == 0)[0]
            N_all   = pvec.force_group_count
            N_empty = valids_empty.size
            if N_empty == 0:
                print("Rejected.")
                return 
            ### Type selected for death move
            type_i = np.random.choice(
                valids_empty,
                replace=True
                )
            ### Type selected for birth move. We will duplicate from
            ### this type and propagate using the symmetric kernel.
            type_j = np.random.choice(
                np.r_[:type_i, (type_i+1):pvec.force_group_count]
                )
            ### P_death: 1./N_empty
            ### P_birth: 1./(N_all-1) * P(x)
            first_parm_i = pvec.parameters_per_force_group * type_i
            last_parm_i  = first_parm_i + pvec.parameters_per_force_group
            value_i      = pvec.vector_k[first_parm_i:last_parm_i]
            first_parm_j = pvec.parameters_per_force_group * type_j
            last_parm_j  = first_parm_j + pvec.parameters_per_force_group
            value_j      = pvec.vector_k[first_parm_j:last_parm_j]

            real_cp    = copy.deepcopy(pvec.vector_k)
            ### d_new: distance for birth proposal
            d_old      = pvec.get_transform(real_cp)[first_parm_i:last_parm_i]
            real_cp[first_parm_i:last_parm_i] = value_j
            d_old     -= pvec.get_transform(real_cp)[first_parm_i:last_parm_i]
            d_old      = np.array(d_old, dtype=float)

            logQ_fwd = 0.
            logQ_bkw = norm_distr_zero_centered_symm.logpdf(d_old).sum()

            pvec.remove(type_i)

            likelihood_func = LikelihoodVectorized(
                [pvec],
                self.targetcomputer
                )
            x1 = likelihood_func.pvec

            likelihood_func.apply_changes(x1)
            log_P_new = self.calc_log_likelihood() + np.sum(self.calc_log_prior())

            log_alpha  = log_P_new - log_P_old
            log_alpha += logQ_bkw  - logQ_fwd
            log_alpha += np.log(N_empty)
            log_alpha -= np.log(N_all-1.)

            print(
                "Mngr"        , pvec_idx, "\n",
                "log_alpha"   , log_alpha, "\n",
                "log_P_new"   , log_P_new,  "\n",
                "-log_P_old"  , -log_P_old,  "\n",
                "logQ_bkw"    , logQ_bkw,  "\n",
                "-logQ_fwd"   , -logQ_fwd, "\n",
                "log(N_empty)", np.log(N_empty), "\n",
                "-log(N_all-1)", -np.log(N_all-1.), "\n",
                )

        elif birth:
            pass

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