import numpy as np
from scipy.special import gamma as gamma_function
from scipy.stats import multivariate_normal
import copy

from .constants import (_UNIT_QUANTITY,
                        _SEED,
                        _INACTIVE_GROUP_IDX
                        )

from .vectors import BaseVector

class LogBasePrior(object):

    def __init__(self, N_parms):

        self.is_log   = True
        self._N_parms = N_parms

    @property
    def N_parms(self):
        return self._N_parms

    def compute(self, x):

        return 0.

    def __call__(self, x):

        return self.compute(x)


class BasePrior(LogBasePrior):

    def __init__(self, N_parms):

        super.__init__(N_parms)
        self.is_log = False

    def __call__(self, x):

        return np.exp(self.compute(x))


class Restraints(BasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)
        self.sig2_vec = BaseVector(dtype=float)
        self.sig2_vec.append(np.ones(N_parms))

    @property
    def sig2(self):
        return self.sig2_vec.vector_values

    def compute(self, diffs):

        rss  = np.sum(diffs.vector_values**2/self.sig2)
        return rss


class LogGaussianPrior(LogBasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)

        self.sig2_vec = BaseVector(dtype=float)
        self.sig2_vec.append(np.ones(N_parms))

        self.mu_vec = BaseVector(dtype=float)
        self.mu_vec.append(np.zeros(N_parms))

    @property
    def N_parms(self):
        return self.sig2_vec.size

    @property
    def sig2(self):
        return np.abs(self.sig2_vec.vector_values)

    @property
    def mu(self):
        return self.mu_vec.vector_values

    def compute(self, x):

        if x.vector_values.shape != self.mu.shape:
            raise ValueError(f"x is shape {x.vector_values.shape} but must be {self.mu.shape}")

        diffs = self.mu - x.vector_values
        pre   = -0.5 * self.N_parms * np.log(2.*np.pi)
        pre  -=  0.5 * np.sum(np.log(self.sig2))
        exp   = -0.5 * np.sum(diffs**2/self.sig2)

        P = pre + exp

        return P


class GaussianPrior(LogGaussianPrior, BasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)


class LogUniformPrior(LogBasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)

        self.is_log  = True

        self.factors_vec = BaseVector(dtype=float)
        self.factors_vec.append(np.ones(N_parms))

    def compute(self, x):

        if np.any(x.vector_values <= 0.):
            P = -np.inf
        else:
            log = np.log(self.factors)
            P   = np.sum(log)

        return P

    @property
    def factors(self):

        return self.factors_vec.vector_values


class UniformPrior(LogUniformPrior, BasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)


class LogRectangularPrior(LogBasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)

        self.center_upper_vec = BaseVector(dtype=float)
        self.center_lower_vec = BaseVector(dtype=float)

        self.center_upper_vec.append(np.ones(self.N_parms))
        self.center_lower_vec.append(-np.ones(self.N_parms))

    def compute(self, x):

        L_max = -1.e10

        valids   = np.where((x.vector_values < self.center_upper) * (x.vector_values > self.center_lower))[0]

        ### Here we just build a very steep wall. That's better
        ### when for gradients.

        invalids_upper = np.where(x.vector_values > self.center_upper)[0]
        invalids_lower = np.where(x.vector_values < self.center_lower)[0]
        diff_upper     = np.copy(x.vector_values[invalids_upper])
        diff_upper    -= self.center_upper[invalids_upper]
        diff_lower     = np.copy(x.vector_values[invalids_lower])
        diff_lower    -= self.center_lower[invalids_lower]
        
        rect_diff      = self.center_upper - self.center_lower
        log_rect_diff  = np.log(rect_diff)

        #if invalids_lower.size > 0 or invalids_upper.size > 0:
        #    L = L_max
        #else:
        #    L = -np.sum(valids_log)

        L  = -1. * 1.e+1 * (np.sum(diff_upper**2) + np.sum(diff_lower**2))
        L -= np.sum(log_rect_diff)

        return L

    @property
    def center_upper(self):

        return self.center_upper_vec.vector_values

    @property
    def center_lower(self):

        return self.center_lower_vec.vector_values


class RectangularPrior(LogRectangularPrior, LogBasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)


class LogInverseGammaPrior(LogBasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)

        self.is_log    = True
        self.alpha_vec = BaseVector(dtype=float)
        self.beta_vec  = BaseVector(dtype=float)

        self.alpha_vec.append(np.ones(self.N_parms, dtype=float))
        self.beta_vec.append(np.ones(self.N_parms, dtype=float))

    def compute(self, x):

        if x.vector_values.shape != self.alpha.shape:
            raise ValueError(f"x is shape {x.vector_values.shape} but must be {self.alpha.shape}")
        if x.vector_values.shape != self.beta.shape:
            raise ValueError(f"x is shape {x.vector_values.shape} but must be {self.beta.shape}")

        pre   = np.sum(self.alpha * np.log(self.beta))
        gam   = gamma_function(self.alpha)
        power = -self.alpha - 1.
        exp   = -self.beta / x.vector_values

        if np.any(x.vector_values <= 0.):
            P = -np.inf
        elif np.any(gam <= 0.):
            P = -np.inf
        else:
            log = pre - np.log(gam) + power*np.log(x.vector_values) + exp
            P   = np.sum(log)

        return P

    @property
    def alpha(self):
        return self.alpha_vec.vector_values

    @property
    def beta(self):
        return self.beta_vec.vector_values


class InverseGammaPrior(LogInverseGammaPrior, BasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)


class LogDirichletPrior(LogBasePrior):

    def __init__(self, N_parms):

        super().__init__(N_parms)

        self.is_log    = True
        self.alpha_vec = BaseVector(dtype=float)

        self.alpha_vec.append(np.ones(self.N_parms, dtype=float)*1.5)

    def compute(self, x):

        x_copy  = np.copy(x.vector_values)
        x_copy /= np.sum(x.vector_values)

        if np.any(x_copy <= 0.):
            P = -np.inf
        else:
            logB  = np.sum(np.log(gamma_function(self.alpha)))
            logB  = logB - np.log(np.sum(gamma_function(self.alpha)))
            power = self.alpha-1.
            P     = -logB + np.sum(power * np.log(x_copy))

        return P

    @property
    def alpha(self):
        return self.alpha_vec.vector_values


class DirichletPrior(LogDirichletPrior, BasePrior):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class LogGaussianAllocationPrior(LogBasePrior):

    def __init__(self, allocation_vecgmm) -> None:

        self.allocation_vecgmm = allocation_vecgmm

        super().__init__(
            self.allocation_vecgmm.N_parms
            )

    def update(self):

        self.allocation_vecgmm.update()

    def compute(self, allocation_vector) -> float:

        self.update()

        unique_types = self.allocation_vecgmm.unique_types
        x_types      = allocation_vector.vector_values
 
        ### 'allocation_vector.vector_values' has shape 
        ### (N_unique_force_objects). Each element corresponds
        ### to a unique force object with positive integer value 
        ### between 0 and N_unique_force_objects.

        ### We don't want to have more allocation indices 
        ### than components in the GMM
        assert np.max(x_types) <= np.max(unique_types)
        assert allocation_vector.size == self.allocation_vecgmm.max_force_rank + 1
        assert allocation_vector.size == self.allocation_vecgmm.unique_force_ranks.size

        ### This call to update is important as it updates the 
        ### allocation_vecgmm with the new values in the w, mu 
        ### and sig vectors (and potentially some other things
        ### internally).
        self.update()

        P_list = np.zeros(
            (
                allocation_vector.size, 
                unique_types.size
                ),
            dtype=float
            )
        P = 0.
        for i in range(unique_types.size):
            P_list[:,i] = self.allocation_vecgmm.get_k(
                unique_types[i]
                )
        for r in range(allocation_vector.size):
            k  = allocation_vector[r]
            if k == _INACTIVE_GROUP_IDX:
                continue
            P += np.log(P_list[r,k])
            P -= np.log(np.sum(P_list[r]))
        return P


class GaussianAllocationPrior(LogGaussianAllocationPrior):

    def __init__(self, allocation_vecgmm):

        super().__init__(allocation_vecgmm)

        self.is_log = False

    def __call__(self, allocation_vector):

        return np.exp(self.compute(allocation_vector))