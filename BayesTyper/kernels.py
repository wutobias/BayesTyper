import numpy as np
from scipy import stats, special
from scipy import optimize
import copy

from .priors import LogGaussianAllocationPrior
from .priors import LogGaussianPrior
from .priors import LogBasePrior
from .vectors import BaseVector

from .constants import (_UNIT_QUANTITY,
                        _SEED,
                        _INACTIVE_GROUP_IDX
                        )

def compute_gradient_per_forcegroup(
    func, 
    ff_vec, 
    force_group_idx_list=None,
    grad_diff=1.e-2,
    use_jac=True
    ):

    if type(force_group_idx_list) == type(None):
        force_group_idx_list= np.arange(
            ff_vec.force_group_count, 
            dtype=int
            )

    N_parms   = ff_vec.parameters_per_force_group
    grad_list = np.zeros(
        len(force_group_idx_list) * N_parms, dtype=float
        )

    grad_idx = 0
    for force_group_idx in force_group_idx_list:
        first_parm = force_group_idx * N_parms
        last_parm  = first_parm + N_parms
        val        = ff_vec[first_parm:last_parm]
        val_0      = ff_vec.vector_0[first_parm:last_parm]
        for parm_idx in range(N_parms):
            val[parm_idx] += grad_diff
            ff_vec.set_parameters_by_force_group(
                force_group_idx,
                val,
                val_0
                )
            logP_plus  = func(ff_vec[:])

            val[parm_idx] -= 2. * grad_diff,
            ff_vec.set_parameters_by_force_group(
                force_group_idx,
                val,
                val_0
                )
            logP_minus = func(ff_vec[:])

            val[parm_idx] += grad_diff
            ff_vec.set_parameters_by_force_group(
                force_group_idx,
                val,
                val_0
                )

            grad_list[grad_idx]  = logP_plus - logP_minus
            grad_list[grad_idx] *= 1./(grad_diff * 2.)
            if use_jac:
                grad_list[grad_idx] *= 1./ff_vec.jacobian[grad_idx]

            grad_idx += 1

    return grad_list


class LogGibbsKernel(LogBasePrior):

    def __init__(
        self,
        log_posterior_callback,
        verbose: bool = False,
        ):

        self.log_posterior_callback = log_posterior_callback
        self.verbose  = verbose

    def compute(self, x, co_vec, co_vec_vals):

        return (0., 0.)

    def __call__(self, x, co_vec, co_vec_vals):

        return self.compute(x, co_vec, co_vec_vals)


class LogJumpKernel(LogBasePrior):

    def __init__(
        self,
        K_max: int,
        verbose: bool = False,
        ):

        self._K_max  = K_max
        self.verbose = verbose

        super().__init__(K_max)

    @property
    def K_max(self):
        return self._K_max

    def compute(self, x, co_vec, co_vec_vals):

        return (0., 0.)

    def __call__(self, x, co_vec, co_vec_vals):

        return self.compute(x, co_vec, co_vec_vals)


class LogKernel(LogBasePrior):

    def __init__(
        self,
        N_parms: int,
        verbose: bool = False,
        ):

        self._N_parms = N_parms
        self.verbose  = verbose

        super().__init__(N_parms)

    def compute(self, x):

        return (0., 0.)

    def __call__(self, x):

        return self.compute(x)


class LogGaussianKernel(LogKernel):

    def __init__(
        self, 
        N_parms):

        super().__init__(N_parms)

        self.sig2_vec = BaseVector(dtype=float)
        self.sig2_vec.append(np.ones(N_parms))

        self.sig2_vec.call_back = self.update

    @property
    def sig2(self):
        return self.sig2_vec.vector_values

    @property
    def mu(self):
        return self.mu_vec.vector_values

    def update(self):

        self._N_parms = self.sig2_vec.size

    def compute(self, parameter_vector):

        sig  = np.sqrt(self.sig2)
        diff = np.random.normal(loc=0,
                                scale=sig,
                                size=self.N_parms)

        pre  = -0.5 * self.N_parms * np.log(2.*np.pi) 
        pre -=  0.5 * np.sum(np.log(self.sig2))
        exp  = -0.5 * np.sum(diff**2/self.sig2)

        logP                 = pre + exp
        parameter_vector[:] += diff
        parameter_vector.apply_changes()

        return logP, logP

    def __call__(self, x):
        return self.compute(x)


class LogUniformAllocationKernel(LogKernel):

    def __init__(
        self,
        K_max,
        N_sele = 2):

        if isinstance(K_max, int):
            self.K_list  = np.arange(K_max, dtype=int)
        elif isinstance(K_max, list):
            self.K_list  = np.array(K_max, dtype=int)
        elif isinstance(K_max, np.ndarray):
            self.K_list  = np.copy(K_max)
        else:
            raise ValueError(
                f"Type of `K_max` {type(K_max)} not understood.")

        self.N_sele = N_sele

    def compute(self, allocation_vector):

        valids     = np.where(
            allocation_vector.vector_values != _INACTIVE_GROUP_IDX
            )[0]
        np.random.shuffle(valids)
        alloc_sele  = valids[:self.N_sele]
        logP_alloc  = -np.log(special.comb(valids.size, self.N_sele))
        logP_alloc -= self.N_sele * np.log(self.K_list.size)

        for r in alloc_sele:
            k_prop               = np.random.choice(self.K_list, 1)[0]
            allocation_vector[r] = k_prop

        allocation_vector.apply_changes()

        return logP_alloc, logP_alloc


class LogMALAKernel(LogKernel):

    def __init__(
        self, 
        N_parms,
        log_posterior_callback,
        sig: float = 0.1
        ):

        super().__init__(N_parms)

        self.sig  = sig
        self.log_posterior_callback = log_posterior_callback

        self.grad_diff = 1.e-2

    @property
    def sig(self):
        return self._sig

    @property
    def sig2(self):
        return self._sig2

    @sig.setter
    def sig(self, value):

        self._sig  = value
        self._sig2 = value**2
        self.norm_distr = stats.norm(0, value)

    def update(self):

        self._N_parms = self.sig2_vec.size

    def compute(self, ff_parameter_vector):

        grad_fwd = compute_gradient_per_forcegroup(
            self.log_posterior_callback,
            ff_parameter_vector,
            grad_diff=self.grad_diff
            )
        d_i        = self.norm_distr.rvs(ff_parameter_vector.size)
        values_old = np.copy(ff_parameter_vector[:])
        grad_fwd_norm = np.linalg.norm(grad_fwd)
        if grad_fwd_norm > 0.:
            grad_fwd /= grad_fwd_norm

        ff_parameter_vector[:] += 0.5 * self.sig2 * grad_fwd + self.sig * d_i
        ff_parameter_vector.apply_changes()

        ### Compute backwards move
        grad_bkw = compute_gradient_per_forcegroup(
            self.log_posterior_callback,
            ff_parameter_vector,
            grad_diff=self.grad_diff
            )

        grad_bkw_norm = np.linalg.norm(grad_bkw)
        if grad_bkw_norm > 0.:
            grad_bkw /= grad_bkw_norm

        ###                y                        x               0.5 sig^2 grad
        forward_diff  = ff_parameter_vector[:] - values_old - 0.5 * self.sig2 * grad_fwd
        backward_diff = values_old - ff_parameter_vector[:] - 0.5 * self.sig2 * grad_bkw

        logP_fwd = np.sum(self.norm_distr.logpdf(forward_diff))
        logP_rev = np.sum(self.norm_distr.logpdf(backward_diff))
        return logP_fwd, logP_rev

    def __call__(self, x):
        return self.compute(x)


class LogMinimizeKernel(LogKernel):

    def __init__(
        self, 
        N_parms,
        log_posterior_callback,
        sig: float = 0.1
        ):

        super().__init__(N_parms)

        self.sig  = sig
        self.log_posterior_callback = log_posterior_callback

        self.grad_diff = 1.e-2

    @property
    def sig(self):
        return self._sig

    @property
    def sig2(self):
        return self._sig2

    @sig.setter
    def sig(self, value):

        self._sig  = value
        self._sig2 = value**2
        self.norm_distr = stats.norm(0, value)

    def update(self):

        self._N_parms = self.sig2_vec.size

    def compute(self, ff_parameter_vector):

        def fun(x):

            ff_parameter_vector[:] = x
            ff_parameter_vector.apply_changes()

            ### Important: `return -1. * posterior`
            ###            because we want to minimize things.
            posterior = self.log_posterior_callback()
            return -1. * posterior

        ### Compute forwards move
        ### =====================
        values_old     = copy.deepcopy(ff_parameter_vector[:])
        x0             = copy.deepcopy(ff_parameter_vector[:])
        result_forward = optimize.minimize(
            fun, 
            x0, 
            method='Nelder-Mead',
            tol = 1e-6,
            options={'disp': True},
            #method='BFGS',
            #options={'gtol': 1e-6, 'disp': True}
            )
        d_i = self.norm_distr.rvs(ff_parameter_vector.size)
        ff_parameter_vector[:] = result_forward.x
        ff_parameter_vector.apply_changes()
        print(f"Forward posterior {self.log_posterior_callback()}")
        print(ff_parameter_vector.vector_k)
        ff_parameter_vector[:] = result_forward.x + d_i
        ff_parameter_vector.apply_changes()
        print(f"Forward posterior + d_i {self.log_posterior_callback()}")
        print(ff_parameter_vector.vector_k)

        ### Compute backwards move
        ### ======================
        values_new      = copy.deepcopy(ff_parameter_vector[:])
        x0              = copy.deepcopy(ff_parameter_vector[:])
        result_backward = optimize.minimize(
            fun, 
            x0,
            method='Nelder-Mead',
            tol = 1e-6,
            options={'disp': True},
            #method='BFGS',
            #options={'gtol': 1e-6, 'disp': True}
            )

        ff_parameter_vector[:] = result_backward.x
        ff_parameter_vector.apply_changes()
        print(f"Backward posterior {self.log_posterior_callback()}")
        ff_parameter_vector[:] = values_new
        ff_parameter_vector.apply_changes()

        ###                y            x          NOT-SURE-IF-CORRECT
        forward_diff  = values_new - values_old - (result_forward.x - values_old)
        backward_diff = values_old - values_new - (result_backward.x - values_new)

        logP_fwd = np.sum(self.norm_distr.logpdf(forward_diff))
        logP_rev = np.sum(self.norm_distr.logpdf(backward_diff))
        return logP_fwd, logP_rev

    def __call__(self, x):
        return self.compute(x)


class LogMergeSplitKernel(LogJumpKernel):

    __doc__ = """

    This kernel performs Merge-Split proposals using random perturbations
    and random merging and splitting.

    """

    def __init__(
        self,
        K_max: int,
        sig: float = 0.01,
        verbose: bool = False,
        ):

        self.sig = sig
        self.norm_distr = stats.norm(0, self.sig)

        super().__init__(K_max, verbose)

    def compute(
        self, 
        ff_parameter_vector,
        co_vector_list: list = list(),
        co_vector_vals_per_group: list = list()):

        P_split = 0.5
        w_split = 0.1

        assert len(co_vector_vals_per_group) == len(co_vector_list)

        K_max_current = ff_parameter_vector.force_group_count - 1
        N_types = K_max_current + 1
        force_group_histogram = ff_parameter_vector.force_group_histogram
        
        N_parms    = ff_parameter_vector.parameters_per_force_group
        N_typables = ff_parameter_vector.allocations.size

        do_split = True
        logP_fwd = 0.
        logP_rev = 0.

        ### Figure out if we should split or merge
        ### ... if we are at maximum K (K_max), do merge always
        if K_max_current >= self.K_max:
            do_split = False
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 0.
        ### ... if we are at minimum K (1), do split always
        elif N_types == 1:
            do_split = True
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 1.
        ### ... if each of the types is allocated only once,
        ### we cannot split. That would create empty types.
        ### We must merge here.
        elif np.sum(force_group_histogram) == N_types:
            do_split = False
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 0.
        else:
            u_merge_split = np.random.random()
            if u_merge_split < P_split:
                do_split = True
            else:
                do_split = False
            logP_fwd = np.log(P_split)
            logP_rev = np.log(1.-P_split)

        ### ============== ###
        ### TYPE SPLITTING ###
        ### ============== ###
        if do_split:

            ### ======================================= ###
            ### FIRST PROPOSE THE NEW ALLOCATION VECTOR ###
            ### ======================================= ###

            ### All types that we can split potentially
            splittables = np.where(
                    force_group_histogram > 1
                    )[0]

            ### Pick one type at random
            i = np.random.choice(splittables)

            ### Propose a splitting.
            ### This is drawn from Multi bernoulli experiment
            ### The disribution is binomial
            n_i = force_group_histogram[i]
            ### 0: no split, 1: split
            xk = [0, 1]
            pk = [1.-w_split, w_split]
            ber = stats.rv_discrete(values=(xk, pk))
            split_selection = ber.rvs(size=n_i)
            no_split_selection = 1-split_selection
            split_selection = split_selection.astype(bool)
            no_split_selection = no_split_selection.astype(bool)
            n_i_new = np.sum(split_selection)
            
            selection_i = ff_parameter_vector.allocations.index([i])[0]
            selection_j = selection_i[split_selection]
            selection_i = selection_i[no_split_selection]

            ### ===================================== ###
            ### SECOND PROPOSE A NEW PARAMETER VECTOR ###
            ### ===================================== ###

            ### First perturbation of all parameters
            d_i = self.norm_distr.rvs(ff_parameter_vector.size)
            ff_parameter_vector[:] += d_i
            ff_parameter_vector.apply_changes()

            ### Then split parameter i
            first_parm = i * N_parms
            last_parm  = first_parm + N_parms
            value_list_i   = ff_parameter_vector[first_parm:last_parm]
            value_list_i_0 = ff_parameter_vector.vector_0[first_parm:last_parm]
            scaling_j      = ff_parameter_vector.scaling_vector[first_parm:last_parm]

            ### Propose a distance for splitting out parameters
            ### This is drawn from Gaussian
            ### _0 are the reference values in the transformation
            d_j = self.norm_distr.rvs(N_parms)
            value_list_j_0 = value_list_i_0 + d_j * scaling_j
            value_list_j = np.array(value_list_i).copy()

            ### ====================== ###
            ### THIRD APPLY EVERYTHING ###
            ### ====================== ###

            ff_parameter_vector.duplicate(i)
            ff_parameter_vector.apply_changes()

            ### We will *not* change the vector_k values
            ### Only the reference values will be changed.
            ff_parameter_vector.set_parameters_by_force_group(
                K_max_current + 1,
                value_list_j,
                value_list_j_0
                )

            if n_i_new > 0:
                ff_parameter_vector.allocations[selection_j] = K_max_current + 1

            ### ============================== ###
            ### COMPUTE PROPOSAL PROBABILITIES ###
            ### ============================== ###

            ### Probability that particular k was selected
            ### ------------------------------------------
            ### 1/K
            if N_types > 1:
                logP_fwd -= np.log(N_types)
            ### 1/K * 1/(K-1)
            logP_rev -= np.log(N_types + 1)

            ### Probability that particular splitting was obtained
            ### --------------------------------------------------

            ### Binomial distribution probability
            logP_fwd += np.log(special.comb(n_i, n_i_new))
            logP_fwd += n_i*np.log(w_split)
            logP_fwd += n_i_new*np.log(1.-w_split)

            ### Merging move will be deterministic, once (i,j) has been selected
            ### So now change of logP_rev here

            ### Type volume correction
            ### ----------------------
            logP_fwd += (N_typables * (np.log(N_types) - np.log(N_types+1)))

            ### Split out distance
            ### ------------------
            logP_fwd += np.sum(
                self.norm_distr.logpdf(d_i)
                )
            logP_rev += np.sum(
                self.norm_distr.logpdf(d_i)
                )
            logP_fwd += np.sum(
                self.norm_distr.logpdf(d_j)
                )

            ### Note: logP_rev does not change after here, since a mergeing move
            ### will always merge all components considered. Therefore 
            ### we would have logP_rev += np.log(1)

            ### ========================= ###
            ### LASTLY, CO-TYPE OTHER VEC ###
            ### ========================= ###

            ### Merge/split all other co-typed vectors
            for co_vector_idx in range(len(co_vector_list)):
                co_vector  = co_vector_list[co_vector_idx]
                first_parm = i * co_vector_vals_per_group[co_vector_idx]
                last_parm  = first_parm + co_vector_vals_per_group[co_vector_idx]
                co_vector.append(co_vector.vector_values[first_parm:last_parm])
                co_vector.apply_changes()

            if self.verbose:
                print(
                    "split:", 
                    i, 
                    ff_parameter_vector.vector_0,
                    ff_parameter_vector.vector_k,
                    )

        ### ============ ###
        ### TYPE MERGING ###
        ### ============ ###
        else:

            ### =============================== ###
            ### FIRST SET NEW ALLOCATION VECTOR ###
            ### =============================== ###

            ### Randomly select the to-be-merged types
            i = np.random.randint(
                low=0,
                high=K_max_current,
                size=1,
                dtype=int)[0]

            j = K_max_current

            ### Move all in k2 into k1
            selection_i = ff_parameter_vector.allocations.index([i])[0]
            selection_j = ff_parameter_vector.allocations.index([j])[0]

            n_i = selection_i.size
            n_j = selection_j.size

            ### =============================== ###
            ### SECOND SET NEW PARAMETER VECTOR ###
            ### =============================== ###

            ### Check the distance between i and j *before*
            ### the perturbation
            first_parm = i * N_parms
            last_parm  = first_parm + N_parms
            value_list_i_0 = ff_parameter_vector.vector_0[first_parm:last_parm]
            value_list_i_0 = np.array(value_list_i_0).copy()

            first_parm = j * N_parms
            last_parm  = first_parm + N_parms
            value_list_j_0 = ff_parameter_vector.vector_0[first_parm:last_parm]
            value_list_j_0 = np.array(value_list_j_0).copy()
            scaling_j      = ff_parameter_vector.scaling_vector[first_parm:last_parm]

            d_j = (value_list_j_0 - value_list_i_0)/ scaling_j
            d_i = self.norm_distr.rvs(ff_parameter_vector.size - N_parms)
            ### Set last N_parms parameters to zero. These are the ones
            ### that will be "merged off".
            ff_parameter_vector[:-N_parms] += d_i
            ff_parameter_vector.apply_changes()

            ### ====================== ###
            ### THIRD APPLY EVERYTHING ###
            ### ====================== ###

            ff_parameter_vector.allocations[selection_j] = i

            ### Must call apply_changes to update the parameter manager, 
            ### before we can remove the force_group.
            ### Important to remove the force group only after we call
            ### `set_parameters_by_force_group`
            ff_parameter_vector.apply_changes()
            ff_parameter_vector.remove(j)

            ### ============================== ###
            ### COMPUTE PROPOSAL PROBABILITIES ###
            ### ============================== ###

            ### Probability that particular (k1,k2) was selected
            ### ------------------------------------------------
            logP_fwd -= np.log(N_types)
            logP_rev -= np.log(N_types - 1)

            ### Probability that particiular merge was done
            ### -------------------------------------------

            ### The merge is deterministic. However the reverse
            ### move (i.e. the split) is not.
            logP_rev += np.log(special.comb(n_i+n_j, n_j))
            logP_rev += (n_i+n_j)*np.log(w_split)
            logP_rev += n_j*np.log(1.-w_split)

            ### Correction type volume
            ### ----------------------

            logP_fwd += (N_typables * (np.log(N_types) - np.log(N_types-1)))

            ### Split out distance
            ### ------------------
            logP_fwd += np.sum(
                self.norm_distr.logpdf(d_i)
                )
            logP_rev += np.sum(
                self.norm_distr.logpdf(d_i)
                )
            logP_rev += np.sum(
                self.norm_distr.logpdf(d_j)
                )

            ### ========================= ###
            ### LASTLY, CO-TYPE OTHER VEC ###
            ### ========================= ###

            for co_vector_idx in range(len(co_vector_list)):
                co_vector   = co_vector_list[co_vector_idx]
                first_parm = i * co_vector_vals_per_group[co_vector_idx]
                last_parm  = first_parm + co_vector_vals_per_group[co_vector_idx]
                for _ in range(first_parm, last_parm):
                    co_vector.remove(first_parm)
                co_vector.apply_changes()

            if self.verbose:
                print(
                    "merge:", 
                    i, j, 
                    ff_parameter_vector.vector_0,
                    ff_parameter_vector.vector_k,
                    )

        ff_parameter_vector.apply_changes()

        return logP_fwd, logP_rev


class LogGibbsKernelSequential(LogGibbsKernel):

    __doc__ = """

    This kernel performs Merge-Split proposals using random perturbations
    and merging and splitting with a Gibbs kernel.

    """

    def __init__(
        self,
        log_posterior_callback = None,
        verbose: bool = False,
        ):

        super().__init__(log_posterior_callback, verbose)

    def compute(
        self, 
        ff_parameter_vector,
        ):

        K_max_current = ff_parameter_vector.force_group_count - 1
        N_types = K_max_current + 1
        
        N_typables = ff_parameter_vector.allocations.size

        logP_fwd = 0.

        ### Loop over all allocations
        for z_i in range(N_typables):
            ### Loop over all types
            k_i_old = ff_parameter_vector.allocations[z_i]
            ### First build the probability vector for each
            log_pk = np.zeros(N_types, dtype=float)
            xk     = np.arange(N_types, dtype=int)
            for k_i in range(N_types):
                ff_parameter_vector.allocations[z_i] = k_i
                ff_parameter_vector.apply_changes()
                log_pk[k_i] = self.log_posterior_callback()
            pk   = np.exp(log_pk)
            norm = math.fsum(pk)
            pk  /= norm
            log_pk   -= np.log(norm)
            ### Sample from conditionals
            gibbs = stats.rv_discrete(values=(xk, pk))
            k_i_new = gibbs.rvs()
            ff_parameter_vector.allocations[z_i] = k_i_new
            ff_parameter_vector.apply_changes()

            logP_fwd += log_pk[k_i_new]

            if self.verbose:
                print(f"Allocated z_i={z_i}: {k_i_old}->{k_i_new}")

        return logP_fwd, logP_fwd


class LogMALAMergeSplitKernel(LogJumpKernel):

    __doc__ = """

    This kernel performs Merge-Split proposals using gradients 
    and random perturbations.
    """

    def __init__(
        self,
        K_max: int,
        log_posterior_callback = None,
        sig: float = 0.1,
        verbose: bool = False,
        ):

        self.sig  = sig
        self.log_posterior_callback = log_posterior_callback

        self.grad_diff = 1.e-2

        super().__init__(K_max, verbose)

    @property
    def sig(self):
        return self._sig

    @property
    def sig2(self):
        return self._sig2

    @sig.setter
    def sig(self, value):

        self._sig  = value
        self._sig2 = value**2
        self.norm_distr = stats.norm(0, value)

    def compute(
        self, 
        ff_parameter_vector,
        co_vector_list: list = list(),
        co_vector_vals_per_group: list = list()):

        P_split = 0.5

        assert len(co_vector_vals_per_group) == len(co_vector_list)

        N_types = ff_parameter_vector.force_group_count
        K_max_current = N_types - 1
        force_group_histogram = ff_parameter_vector.force_group_histogram
        
        N_parms    = ff_parameter_vector.parameters_per_force_group
        N_typables = ff_parameter_vector.allocations.size

        do_split = True
        logP_fwd = 0.
        logP_rev = 0.

        ### Figure out if we should split or merge
        ### ... if we are at maximum K (K_max), do merge always
        if K_max_current >= self.K_max:
            do_split = False
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 0.
        ### ... if we are at minimum K (1), do split always
        elif N_types == 1:
            do_split = True
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 1.
        ### ... if each of the types is allocated only once,
        ### we cannot split. That would create empty types.
        ### We must merge here.
        elif np.sum(force_group_histogram) == N_types:
            do_split = False
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 0.
        else:
            u_merge_split = np.random.random()
            if u_merge_split < P_split:
                do_split = True
            else:
                do_split = False
            logP_fwd = np.log(P_split)
            logP_rev = np.log(1.-P_split)

        ### ============== ###
        ### TYPE SPLITTING ###
        ### ============== ###
        if do_split:

            ### ============================ ###
            ### FIRST SELECT A TYPE TO SPLIT ###
            ### ============================ ###

            ### All types that we can split potentially
            splittables = np.where(
                    force_group_histogram > 1
                    )[0]

            ### Pick one type at random
            type_i = np.random.choice(splittables)
            type_j = K_max_current + 1

            ### Create dummy type for gradient assessment later
            ff_parameter_vector.duplicate(type_i)
            ff_parameter_vector.apply_changes()

            ### ======================== ###
            ### SECOND CO-TYPE OTHER VEC ###
            ### ======================== ###

            ### Merge/split all other co-typed vectors
            for co_vector_idx in range(len(co_vector_list)):
                co_vector  = co_vector_list[co_vector_idx]
                first_parm = type_i * co_vector_vals_per_group[co_vector_idx]
                last_parm  = first_parm + co_vector_vals_per_group[co_vector_idx]
                co_vector.append(co_vector.vector_values[first_parm:last_parm])
                co_vector.apply_changes()

            ### ===================================================== ###
            ### THIRD PROPOSE ALLOCATION/PARAMETERS FOR THE SPLITTING ###
            ### ===================================================== ###

            selection_i = ff_parameter_vector.allocations.index([type_i])[0]
            N_allocs    = selection_i.size
            ### Generate all possible splitting solutions
            ### The `[1:-1]` removes the two solutions that
            ### would lead to empty types.
            k_values_ij = np.array(
                np.meshgrid(
                    *[[type_i, type_j] for _ in range(N_allocs)]
                    )
                ).T.reshape(-1,N_allocs)[1:-1]
            N_comb = k_values_ij.shape[0]
            grad_i = np.zeros((N_comb, N_parms), dtype=float)
            grad_j = np.zeros((N_comb, N_parms), dtype=float)
            grad_ij_dot  = np.zeros(N_comb, dtype=float)
            grad_ij_diff = np.zeros(N_comb, dtype=float)
            ### Loop over all 2**N_allocs combinations
            ### and find the one that minimizes the following 
            ### gradient dot product:
            ###     grad_i grad_j

            first_parm = type_i * N_parms
            last_parm  = first_parm + N_parms

            value_list_i   = ff_parameter_vector[first_parm:last_parm]
            value_list_i_0 = ff_parameter_vector.vector_0[first_parm:last_parm]
            scaling        = ff_parameter_vector.scaling_vector[first_parm:last_parm]

            for comb_idx in range(N_comb):
                ff_parameter_vector.allocations[selection_i] = k_values_ij[comb_idx]
                ff_parameter_vector.apply_changes()
                grad_i[comb_idx] = compute_gradient_per_forcegroup(
                        self.log_posterior_callback,
                        ff_parameter_vector,
                        force_group_idx_list=[type_i],
                        grad_diff=self.grad_diff
                        )
                grad_j[comb_idx] = compute_gradient_per_forcegroup(
                        self.log_posterior_callback,
                        ff_parameter_vector,
                        force_group_idx_list=[type_j],
                        grad_diff=self.grad_diff
                        )

                norm_i = np.linalg.norm(grad_i[comb_idx])
                norm_j = np.linalg.norm(grad_j[comb_idx])

                grad_ij_dot[comb_idx] = np.dot(
                    grad_i[comb_idx], 
                    grad_j[comb_idx]
                    )
                grad_ij_diff[comb_idx] = np.linalg.norm(
                    grad_i[comb_idx] - grad_j[comb_idx],                    
                    )

                if self.verbose:
                    print(
                        k_values_ij[comb_idx],
                        grad_i[comb_idx], 
                        grad_j[comb_idx],
                        grad_ij_dot[comb_idx],
                        grad_ij_diff[comb_idx],
                        )

                if norm_i > 0.:
                    grad_i[comb_idx] /= norm_i
                if norm_j > 0.:
                    grad_j[comb_idx] /= norm_j

            if N_parms == 1:
                best_comb_idx = np.argmax(grad_ij_diff)
            else:
                best_comb_idx = np.argmin(grad_ij_dot)            
            best_grad_j   = grad_j[best_comb_idx]
            d_j           = self.norm_distr.rvs(N_parms)
            ff_parameter_vector.allocations[selection_i] = k_values_ij[best_comb_idx]
            ff_parameter_vector.apply_changes()

            value_list_j    = np.array(value_list_i).copy()
            value_list_j_0  = copy.copy(value_list_i_0)
            value_list_j_0 += scaling * (0.5 * self.sig2 * best_grad_j + self.sig * d_j)

            forward_diff_j  = (value_list_j_0 - value_list_i_0) / scaling - 0.5 * self.sig2 * best_grad_j
            if self.verbose:
                print(
                    forward_diff_j,
                    scaling,
                    value_list_j_0,
                    value_list_i_0,
                    best_grad_j)

            ff_parameter_vector.set_parameters_by_force_group(
                type_j,
                value_list_j,
                value_list_j_0
                )

            ### ====================================== ###
            ### PROPOSE NEW VECTOR FOR EVERYTHING ELSE ###
            ### ====================================== ###

            values_old = np.copy(ff_parameter_vector[:])

            ### Compute forward move
            grad_fwd = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                grad_diff=self.grad_diff
                )
            d_i = self.norm_distr.rvs(ff_parameter_vector.size)

            norm_fwd = np.linalg.norm(grad_fwd)
            if norm_fwd > 0.:
                grad_fwd /= norm_fwd

            ff_parameter_vector[:] += 0.5 * self.sig2 * grad_fwd + self.sig * d_i
            ff_parameter_vector.apply_changes()

            ### Compute backwards move
            grad_bkw = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                grad_diff=self.grad_diff
                )

            norm_bkw = np.linalg.norm(grad_bkw)
            if norm_bkw > 0.:
                grad_bkw /= norm_bkw

            ###                y                        x               0.5 sig^2 grad
            forward_diff  = ff_parameter_vector[:] - values_old - 0.5 * self.sig2 * grad_fwd
            backward_diff = values_old - ff_parameter_vector[:] - 0.5 * self.sig2 * grad_bkw

            ### ============================== ###
            ### COMPUTE PROPOSAL PROBABILITIES ###
            ### ============================== ###

            ### Probability that particular type_i was selected
            ### -----------------------------------------------
            ### 1/K
            logP_fwd -= np.log(N_types)
            logP_rev -= np.log(N_types)

            ### Probability that particular splitting was obtained
            ### --------------------------------------------------

            ### We already computed this during the Gibbs moves ...

            ### Type volume correction
            ### ----------------------
            logP_rev += (N_typables * (np.log(N_types) - np.log(N_types+1)))

            ### Note: logP_rev does not change after here, since a mergeing move
            ### will always merge all components considered. Therefore 
            ### we would have logP_rev += np.log(1)

            ### Note: self.norm_distr is zero-centered, i.e. loc=0.
            ###       Therefore we use shifted input here (P(y) with y=x-loc)
            logP_fwd += np.sum(self.norm_distr.logpdf(forward_diff_j))
            logP_fwd += np.sum(self.norm_distr.logpdf(forward_diff))
            logP_rev += np.sum(self.norm_distr.logpdf(backward_diff))

            if self.verbose:
                print(
                    "split:", 
                    type_i,
                    ff_parameter_vector.allocations.index([type_i])[0], 
                    ff_parameter_vector.allocations.index([type_j])[0],
                    ff_parameter_vector.vector_0,
                    ff_parameter_vector.vector_k,
                    )

        ### ============ ###
        ### TYPE MERGING ###
        ### ============ ###

        else:

            ### ========================== ###
            ### FIRST SELECT TYPE TO MERGE ###
            ### ========================== ###

            type_list = np.arange(
                    N_types, dtype=int
                    )

            ### Randomly select the to-be-merged types
            type_i = np.random.choice(type_list)
            type_j = np.random.choice(
                np.delete(
                    type_list, type_i
                    )
                )

            ### Select allocations in i and j
            selection_i  = ff_parameter_vector.allocations.index([type_i])[0]
            selection_j  = ff_parameter_vector.allocations.index([type_j])[0]
            selection_ij = np.append(selection_i, selection_j)

            n_i = selection_i.size
            n_j = selection_j.size

            ### =============================== ###
            ### SECOND PROPOSE PARAMETER VECTOR ###
            ### =============================== ###

            values_old = np.copy(ff_parameter_vector[:])

            ### Compute forward move
            grad_fwd = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                force_group_idx_list=np.arange(N_types-1, dtype=int),
                grad_diff=self.grad_diff
                )
            d_i = self.norm_distr.rvs(ff_parameter_vector.size-N_parms)

            norm_fwd = np.linalg.norm(grad_fwd)
            if norm_fwd > 0.:
                grad_fwd /= norm_fwd

            ff_parameter_vector[:-N_parms] += 0.5 * self.sig2 * grad_fwd + self.sig * d_i
            ff_parameter_vector.apply_changes()

            ### Compute backwards move
            grad_bkw = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                force_group_idx_list=np.arange(N_types-1, dtype=int),
                grad_diff=self.grad_diff
                )

            grad_norm = np.linalg.norm(grad_bkw)
            if grad_norm > 0.:
                grad_bkw /= grad_norm

            ###                y                        x               0.5 sig^2 grad
            forward_diff  = ff_parameter_vector[:-N_parms] - values_old[:-N_parms] - 0.5 * self.sig2 * grad_fwd
            backward_diff = values_old[:-N_parms] - ff_parameter_vector[:-N_parms] - 0.5 * self.sig2 * grad_bkw

            ### ====================================== ###
            ### THIRD COMPUTE REVERSE (SPLITTING) MOVE ###
            ### ====================================== ###

            first_parm = type_i * N_parms
            last_parm  = first_parm + N_parms

            value_list_i   = ff_parameter_vector[first_parm:last_parm]
            value_list_i_0 = ff_parameter_vector.vector_0[first_parm:last_parm]
            scaling        = ff_parameter_vector.scaling_vector[first_parm:last_parm]
            scaled_diff    = scaling * self.grad_diff

            value_list_j   = ff_parameter_vector[first_parm:last_parm]
            value_list_j_0 = ff_parameter_vector.vector_0[first_parm:last_parm]

            ff_parameter_vector.set_parameters_by_force_group(
                type_j,
                value_list_i,
                value_list_i_0
                )

            grad_j = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                force_group_idx_list=[type_j],
                grad_diff=self.grad_diff
                )

            norm_j = np.linalg.norm(grad_j)
            if norm_j > 0.:
                grad_j /= norm_j

            backward_diff_j  = (value_list_j_0 - value_list_i_0)/scaling - 0.5 * self.sig2 * grad_j

            ff_parameter_vector.allocations[selection_j] = type_i
            ### Must call apply_changes to update the parameter manager, 
            ### before we can remove the force_group.
            ### Important to remove the force group only after we call
            ### `set_parameters_by_force_group`
            ff_parameter_vector.apply_changes()
            ff_parameter_vector.remove(type_j)

            ### ======================== ###
            ### THIRD, CO-TYPE OTHER VEC ###
            ### ======================== ###

            ### Merge/split all other co-typed vectors
            for co_vector_idx in range(len(co_vector_list)):
                co_vector   = co_vector_list[co_vector_idx]
                first_parm = type_j * co_vector_vals_per_group[co_vector_idx]
                last_parm  = first_parm + co_vector_vals_per_group[co_vector_idx]
                for _ in range(first_parm, last_parm):
                    co_vector.remove(first_parm)
                co_vector.apply_changes()

            ### ============================== ###
            ### COMPUTE PROPOSAL PROBABILITIES ###
            ### ============================== ###

            ### Probability that particular type_i was selected
            ### ------------------------------------------------
            logP_fwd -= np.log(N_types)
            logP_fwd -= np.log(N_types - 1)
            logP_rev -= np.log(N_types - 1)
            logP_rev -= np.log(N_types)

            ### Correction type volume
            ### ----------------------

            logP_rev += (N_typables * (np.log(N_types) - np.log(N_types-1)))

            ### Note: self.norm_distr is zero-centered, i.e. loc=0.
            ###       Therefore we use shifted input here (P(y) with y=x-loc)
            logP_fwd            += np.sum(self.norm_distr.logpdf(forward_diff))
            logP_rev            += np.sum(self.norm_distr.logpdf(backward_diff))
            logP_rev            += np.sum(self.norm_distr.logpdf(backward_diff_j))

            if self.verbose:
                print(
                    "merge:", 
                    type_i, type_j,
                    selection_i, selection_j,
                    ff_parameter_vector.vector_0,
                    ff_parameter_vector.vector_k,
                    )

        ff_parameter_vector.apply_changes()

        return logP_fwd, logP_rev


class LogMALABirthDeathKernel(LogJumpKernel):

    __doc__ = """

    This kernel performs Birth-Death proposals using gradients
    and random perturbations.
    """

    def __init__(
        self,
        K_max: int,
        log_posterior_callback = None,
        sig: float = 0.1,
        verbose: bool = False,
        ):

        self.sig  = sig
        self.sig2 = sig**2
        self.log_posterior_callback = log_posterior_callback
        self.norm_distr = stats.norm(0, self.sig)

        self.grad_diff = 1.e-2

        super().__init__(K_max, verbose)

    def compute(
        self, 
        ff_parameter_vector,
        co_vector_list: list = list(),
        co_vector_vals_per_group: list = list()):

        P_split = 0.5

        assert len(co_vector_vals_per_group) == len(co_vector_list)

        N_types = ff_parameter_vector.force_group_count
        K_max_current = N_types - 1
        force_group_histogram = ff_parameter_vector.force_group_histogram
        
        N_parms    = ff_parameter_vector.parameters_per_force_group
        N_typables = ff_parameter_vector.allocations.size

        do_split = True
        logP_fwd = 0.
        logP_rev = 0.

        ### Figure out if we should split or merge
        ### ... if we are at maximum K (K_max), do merge always
        if K_max_current >= self.K_max:
            do_split = False
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 0.
        ### ... if we are at minimum K (1), do split always
        elif N_types == 1:
            do_split = True
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 1.
        ### ... if each of the types is allocated only once,
        ### we cannot split. That would create empty types.
        ### We must merge here.
        elif np.sum(force_group_histogram) == N_types:
            do_split = False
            logP_fwd = 0.
            logP_rev = np.log(P_split)
            P_split  = 0.
        else:
            u_merge_split = np.random.random()
            if u_merge_split < P_split:
                do_split = True
            else:
                do_split = False
            logP_fwd = np.log(P_split)
            logP_rev = np.log(1.-P_split)

        ### ============== ###
        ### TYPE SPLITTING ###
        ### ============== ###
        if do_split:

            ### ============================ ###
            ### FIRST SELECT A TYPE TO SPLIT ###
            ### ============================ ###

            ### All types that we can split potentially
            splittables = np.where(
                    force_group_histogram > 1
                    )[0]

            ### Pick one type at random
            type_i = np.random.choice(splittables)
            type_j = K_max_current + 1

            ### Create dummy type for gradient assessment later
            ff_parameter_vector.duplicate(type_i)
            ff_parameter_vector.apply_changes()

            ### ======================== ###
            ### SECOND CO-TYPE OTHER VEC ###
            ### ======================== ###

            ### Merge/split all other co-typed vectors
            for co_vector_idx in range(len(co_vector_list)):
                co_vector  = co_vector_list[co_vector_idx]
                first_parm = type_i * co_vector_vals_per_group[co_vector_idx]
                last_parm  = first_parm + co_vector_vals_per_group[co_vector_idx]
                co_vector.append(co_vector.vector_values[first_parm:last_parm])
                co_vector.apply_changes()

            ### ===================================================== ###
            ### THIRD PROPOSE ALLOCATION/PARAMETERS FOR THE SPLITTING ###
            ### ===================================================== ###

            selection_i = ff_parameter_vector.allocations.index([type_i])[0]
            N_allocs    = selection_i.size
            ### Generate all possible splitting solutions
            ### The `[1:-1]` removes the two solutions that
            ### would lead to empty types.
            k_values_ij = np.array(
                np.meshgrid(
                    *[[type_i, type_j] for _ in range(N_allocs)]
                    )
                ).T.reshape(-1,N_allocs)[1:-1]
            N_comb = k_values_ij.shape[0]
            grad_i = np.zeros((N_comb, N_parms), dtype=float)
            grad_j = np.zeros((N_comb, N_parms), dtype=float)
            grad_ij_dot  = np.zeros(N_comb, dtype=float)
            grad_ij_diff = np.zeros(N_comb, dtype=float)
            ### Loop over all 2**N_allocs combinations
            ### and find the one that minimizes the following 
            ### gradient dot product:
            ###     grad_i grad_j

            first_parm = type_i * N_parms
            last_parm  = first_parm + N_parms

            value_list_i   = ff_parameter_vector[first_parm:last_parm]
            value_list_i_0 = ff_parameter_vector.vector_0[first_parm:last_parm]
            scaling        = ff_parameter_vector.scaling_vector[first_parm:last_parm]

            for comb_idx in range(N_comb):
                ff_parameter_vector.allocations[selection_i] = k_values_ij[comb_idx]
                ff_parameter_vector.apply_changes()
                grad_i[comb_idx] = compute_gradient_per_forcegroup(
                        self.log_posterior_callback,
                        ff_parameter_vector,
                        force_group_idx_list=[type_i],
                        grad_diff=self.grad_diff
                        )
                grad_j[comb_idx] = compute_gradient_per_forcegroup(
                        self.log_posterior_callback,
                        ff_parameter_vector,
                        force_group_idx_list=[type_j],
                        grad_diff=self.grad_diff
                        )

                norm_i = np.linalg.norm(grad_i[comb_idx])
                norm_j = np.linalg.norm(grad_j[comb_idx])

                grad_ij_dot[comb_idx] = np.dot(
                    grad_i[comb_idx], 
                    grad_j[comb_idx]
                    )
                grad_ij_diff[comb_idx] = np.linalg.norm(
                    grad_i[comb_idx] - grad_j[comb_idx],                    
                    )

                if self.verbose:
                    print(
                        k_values_ij[comb_idx],
                        grad_i[comb_idx], 
                        grad_j[comb_idx],
                        grad_ij_dot[comb_idx],
                        grad_ij_diff[comb_idx],
                        )

                if norm_i > 0.:
                    grad_i[comb_idx] /= norm_i
                if norm_j > 0.:
                    grad_j[comb_idx] /= norm_j

            if N_parms == 1:
                best_comb_idx = np.argmax(grad_ij_diff)
            else:
                best_comb_idx = np.argmin(grad_ij_dot)            
            best_grad_j   = grad_j[best_comb_idx]
            d_j           = self.norm_distr.rvs(N_parms)
            ff_parameter_vector.allocations[selection_i] = k_values_ij[best_comb_idx]
            ff_parameter_vector.apply_changes()

            value_list_j    = np.array(value_list_i).copy()
            value_list_j_0  = copy.copy(value_list_i_0)
            value_list_j_0 += scaling * (0.5 * self.sig2 * best_grad_j + self.sig * d_j)

            forward_diff_j  = (value_list_j_0 - value_list_i_0) / scaling - 0.5 * self.sig2 * best_grad_j
            if self.verbose:
                print(
                    forward_diff_j,
                    scaling,
                    value_list_j_0,
                    value_list_i_0,
                    best_grad_j)

            ff_parameter_vector.set_parameters_by_force_group(
                type_j,
                value_list_j,
                value_list_j_0
                )

            ### ====================================== ###
            ### PROPOSE NEW VECTOR FOR EVERYTHING ELSE ###
            ### ====================================== ###

            values_old = np.copy(ff_parameter_vector[:])

            ### Compute forward move
            grad_fwd = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                grad_diff=self.grad_diff
                )
            d_i = self.norm_distr.rvs(ff_parameter_vector.size)

            norm_fwd = np.linalg.norm(grad_fwd)
            if norm_fwd > 0.:
                grad_fwd /= norm_fwd

            ff_parameter_vector[:] += 0.5 * self.sig2 * grad_fwd + self.sig * d_i
            ff_parameter_vector.apply_changes()

            ### Compute backwards move
            grad_bkw = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                grad_diff=self.grad_diff
                )

            norm_bkw = np.linalg.norm(grad_bkw)
            if norm_bkw > 0.:
                grad_bkw /= norm_bkw

            ###                y                        x               0.5 sig^2 grad
            forward_diff  = ff_parameter_vector[:] - values_old - 0.5 * self.sig2 * grad_fwd
            backward_diff = values_old - ff_parameter_vector[:] - 0.5 * self.sig2 * grad_bkw

            ### ============================== ###
            ### COMPUTE PROPOSAL PROBABILITIES ###
            ### ============================== ###

            ### Probability that particular type_i was selected
            ### -----------------------------------------------
            ### 1/K
            logP_fwd -= np.log(N_types)
            logP_rev -= np.log(N_types)

            ### Probability that particular splitting was obtained
            ### --------------------------------------------------

            ### We already computed this during the Gibbs moves ...

            ### Type volume correction
            ### ----------------------
            logP_rev += (N_typables * (np.log(N_types) - np.log(N_types+1)))

            ### Note: logP_rev does not change after here, since a mergeing move
            ### will always merge all components considered. Therefore 
            ### we would have logP_rev += np.log(1)

            ### Note: self.norm_distr is zero-centered, i.e. loc=0.
            ###       Therefore we use shifted input here (P(y) with y=x-loc)
            logP_fwd            += np.sum(self.norm_distr.logpdf(forward_diff_j))
            logP_fwd            += np.sum(self.norm_distr.logpdf(forward_diff))
            logP_rev            += np.sum(self.norm_distr.logpdf(backward_diff))

            if self.verbose:
                print(
                    "split:", 
                    type_i,
                    ff_parameter_vector.allocations.index([type_i])[0], 
                    ff_parameter_vector.allocations.index([type_j])[0],
                    ff_parameter_vector.vector_0,
                    ff_parameter_vector.vector_k,
                    )

        ### ============ ###
        ### TYPE MERGING ###
        ### ============ ###

        else:

            ### ========================== ###
            ### FIRST SELECT TYPE TO MERGE ###
            ### ========================== ###

            type_list = np.arange(
                    N_types, dtype=int
                    )

            ### Randomly select the to-be-merged types
            type_i = np.random.choice(type_list)
            type_j = np.random.choice(
                np.delete(
                    type_list, type_i
                    )
                )

            ### Select allocations in i and j
            selection_i  = ff_parameter_vector.allocations.index([type_i])[0]
            selection_j  = ff_parameter_vector.allocations.index([type_j])[0]
            selection_ij = np.append(selection_i, selection_j)

            n_i = selection_i.size
            n_j = selection_j.size

            ### =============================== ###
            ### SECOND PROPOSE PARAMETER VECTOR ###
            ### =============================== ###

            values_old = np.copy(ff_parameter_vector[:])

            ### Compute forward move
            grad_fwd = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                force_group_idx_list=np.arange(N_types-1, dtype=int),
                grad_diff=self.grad_diff
                )
            d_i = self.norm_distr.rvs(ff_parameter_vector.size-N_parms)

            norm_fwd = np.linalg.norm(grad_fwd)
            if norm_fwd > 0.:
                grad_fwd /= norm_fwd

            ff_parameter_vector[:-N_parms] += 0.5 * self.sig2 * grad_fwd + self.sig * d_i
            ff_parameter_vector.apply_changes()

            ### Compute backwards move
            grad_bkw = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                force_group_idx_list=np.arange(N_types-1, dtype=int),
                grad_diff=self.grad_diff
                )

            grad_norm = np.linalg.norm(grad_bkw)
            if grad_norm > 0.:
                grad_bkw /= grad_norm

            ###                y                        x               0.5 sig^2 grad
            forward_diff  = ff_parameter_vector[:-N_parms] - values_old[:-N_parms] - 0.5 * self.sig2 * grad_fwd
            backward_diff = values_old[:-N_parms] - ff_parameter_vector[:-N_parms] - 0.5 * self.sig2 * grad_bkw

            ### ====================================== ###
            ### THIRD COMPUTE REVERSE (SPLITTING) MOVE ###
            ### ====================================== ###

            first_parm = type_i * N_parms
            last_parm  = first_parm + N_parms

            value_list_i   = ff_parameter_vector[first_parm:last_parm]
            value_list_i_0 = ff_parameter_vector.vector_0[first_parm:last_parm]
            scaling        = ff_parameter_vector.scaling_vector[first_parm:last_parm]
            scaled_diff    = scaling * self.grad_diff

            value_list_j   = ff_parameter_vector[first_parm:last_parm]
            value_list_j_0 = ff_parameter_vector.vector_0[first_parm:last_parm]

            ff_parameter_vector.set_parameters_by_force_group(
                type_j,
                value_list_i,
                value_list_i_0
                )

            grad_j = compute_gradient_per_forcegroup(
                self.log_posterior_callback,
                ff_parameter_vector,
                force_group_idx_list=[type_j],
                grad_diff=self.grad_diff
                )

            norm_j = np.linalg.norm(grad_j)
            if norm_j > 0.:
                grad_j /= norm_j

            backward_diff_j  = (value_list_j_0 - value_list_i_0)/scaling - 0.5 * self.sig2 * grad_j

            ff_parameter_vector.allocations[selection_j] = type_i
            ### Must call apply_changes to update the parameter manager, 
            ### before we can remove the force_group.
            ### Important to remove the force group only after we call
            ### `set_parameters_by_force_group`
            ff_parameter_vector.apply_changes()
            ff_parameter_vector.remove(type_j)

            ### ======================== ###
            ### THIRD, CO-TYPE OTHER VEC ###
            ### ======================== ###

            ### Merge/split all other co-typed vectors
            for co_vector_idx in range(len(co_vector_list)):
                co_vector   = co_vector_list[co_vector_idx]
                first_parm = type_j * co_vector_vals_per_group[co_vector_idx]
                last_parm  = first_parm + co_vector_vals_per_group[co_vector_idx]
                for _ in range(first_parm, last_parm):
                    co_vector.remove(first_parm)
                co_vector.apply_changes()

            ### ============================== ###
            ### COMPUTE PROPOSAL PROBABILITIES ###
            ### ============================== ###

            ### Probability that particular type_i was selected
            ### ------------------------------------------------
            logP_fwd -= np.log(N_types)
            logP_fwd -= np.log(N_types - 1)
            logP_rev -= np.log(N_types - 1)
            logP_rev -= np.log(N_types)

            ### Correction type volume
            ### ----------------------

            logP_rev += (N_typables * (np.log(N_types) - np.log(N_types-1)))

            ### Note: self.norm_distr is zero-centered, i.e. loc=0.
            ###       Therefore we use shifted input here (P(y) with y=x-loc)
            logP_fwd            += np.sum(self.norm_distr.logpdf(forward_diff))
            logP_rev            += np.sum(self.norm_distr.logpdf(backward_diff))
            logP_rev            += np.sum(self.norm_distr.logpdf(backward_diff_j))

            if self.verbose:
                print(
                    "merge:", 
                    type_i, type_j,
                    selection_i, selection_j,
                    ff_parameter_vector.vector_0,
                    ff_parameter_vector.vector_k,
                    )

        ff_parameter_vector.apply_changes()

        return logP_fwd, logP_rev

