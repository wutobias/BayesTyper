import numpy as np
from scipy import stats
import copy

from .kernels import LogJumpKernel
from .kernels import compute_gradient_per_forcegroup
from .vectors import ForceFieldParameterVector
from .vectors import SmartsForceFieldParameterVector
from .likelihoods import LogGaussianLikelihood
from .likelihoods import LikelihoodVectorized

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


class TypingConstraints(object):

    def __init__(self, constraints_list):

        """
        This class handles typing constraints in the 
        form of SMIRNOFF type SMARTS strings.
        The input is a list of constraints given as SMARTS
        strings. The order of the SMARTS strings defines
        the hierarchy of the typing constraints.
        """

        from rdkit import Chem

        self._constraints_list = constraints_list
        self._num_constraints  = len(constraints_list)

        self._rdmol_list = list()
        for constraint in constraints_list:
            self._rdmol_list.append(
                Chem.MolFromSmarts(constraint)
                )

    @property
    def rdmol_list(self):

        return self._rdmol_list

    @property
    def constraints_list(self):

        return self._constraints_list

    @property
    def num_constraints(self):

        return self._num_constraints


    def is_pvec_valid_hierarchy(self, pvec, verbose=False):

        """
        Check if allocations in parameter vector
        are configured w.r.t. to hierarchy in constraints list.
        If yes, then return `True`. Else return `False`.
        """

        type_encoding, constraints_encoding = self.get_binary_encoding(pvec)

        for t in range(pvec.force_group_count):
            ty = type_encoding[t]
            ty_sum = np.sum(ty)
            if ty_sum > 0:
                parent_idx = -1
                for constraint_i in range(self.num_constraints):
                    pa = constraints_encoding[constraint_i]
                    type_product = np.sum(pa * ty)
                    ### If these two match, then we have found a
                    ### potential parent type.
                    if type_product == ty_sum:
                        parent_idx = constraint_i
                    ### if `parent_idx > -1`, it means that we already
                    ### have found the parent index. At that point the 
                    ### type product `type_product` cannot be greater
                    ### than zero anymore. Because that would mean that
                    ### some groups want to go deeper in the typing tree.
                    else:
                        if parent_idx > -1 and type_product > 0:
                            if verbose:
                                print(
                                    f"Type {t} has parent type {parent_idx} but {type_product}/{ty_sum} incomplete matches were also found for parent type {constraint_i}"
                                    )
                            return False
                ### This means we have not found a parent type.
                if parent_idx == -1:
                    if verbose:
                        print(
                            f"Could not find parent type for type {t}"
                            )
                    return False

        if verbose:
            print(
                "Hierarchy looks good to me."
                )

        return True


    def retrieve_hierarchy(self, pvec):

        type_encoding, constraints_encoding = self.get_binary_encoding(pvec)

        parent_assignment_dict = {constraint_i:[] for constraint_i in range(self.num_constraints)}
        parent_assignment_dict.update({-1:[]})

        for t in range(pvec.force_group_count):
            ty = type_encoding[t]
            ty_sum = np.sum(ty)
            parent_idx = -1
            if ty_sum > 0:
                for constraint_i in range(self.num_constraints):
                    pa = constraints_encoding[constraint_i]
                    type_product = np.sum(pa * ty)
                    if type_product == ty_sum:
                        parent_idx = constraint_i
            parent_assignment_dict[parent_idx].append(t)

        return parent_assignment_dict


    def resolve_ordering(self, pvec):

        """
        Resolve any mismatches in the type ordering in the parameter vector `pvec`
        according to the hierarchy defined in the constraints.
        """

        import copy

        type_encoding, constraints_encoding = self.get_binary_encoding(pvec)

        resolve_dict = {constraint_i:list() for constraint_i in range(self.num_constraints)}

        for constraint_i in range(self.num_constraints):
            pa = constraints_encoding[constraint_i]
            for t in range(pvec.force_group_count):    
                ch = type_encoding[t]    
                type_product_vec = pa * ch
                type_product = np.sum(type_product_vec)
                if type_product > 0:
                    resolve_dict[constraint_i].append(t)

        type_map  = dict()
        type_reverse_map = dict()
        allocations_new = copy.deepcopy(pvec.allocations[:])
        type_max = 0
        for constraint_i in range(self.num_constraints):
            for t in resolve_dict[constraint_i]:
                if not t in type_map:
                    type_map[t] = type_max
                    type_reverse_map[type_max] = t
                    type_max   += 1
                sele = pvec.allocations.index([t])[0]
                allocations_new[sele] = type_map[t]

        for t in range(type_max):
            pvec.duplicate(type_reverse_map[t])
        allocations_new    += type_max
        pvec.allocations[:] = allocations_new
        pvec.apply_changes()
        for _ in range(type_max):
            pvec.remove(0)


    def query_available_types(self, pvec):

        """
        For each allocation entry in pvec, return the
        available types based on the current state of the
        allocation vector.
        """

        parent_assignment_dict = self.retrieve_hierarchy(pvec)

        available_types_for_z = {z:list() for z in range(pvec.allocations.size)}

        for constraint_i in parent_assignment_dict:
            if parent_assignment_dict[constraint_i]:
                idxs_list = pvec.allocations.index(
                    parent_assignment_dict[constraint_i]
                    )
                for idxs in idxs_list:
                    for z in idxs:
                        available_types_for_z[z].extend(
                            parent_assignment_dict[constraint_i]
                            )

        histogram = pvec.force_group_histogram
        for t in range(pvec.force_group_count):
            if histogram[t] == 0:
                for z in range(pvec.allocations.size):
                    available_types_for_z[z].append(t)

        return available_types_for_z


    def get_binary_encoding(self, pvec):

        """
        Returns binary encoding of the allocated types
        in the parameter vector and the constraints.
        """

        import numpy as np

        type_encoding = np.zeros(
            (
                pvec.force_group_count,
                pvec.allocations.size
            ),
            dtype=bool
            )

        constraints_encoding = np.zeros(
            (
                self.num_constraints,
                pvec.allocations.size
            ),
            dtype=bool
            )

        allocation_indices = pvec.allocations.index(
            range(pvec.force_group_count)
            )
        ### Get binary encoding of actual type assignment
        for type_i in range(pvec.force_group_count):
            valid_allocations = allocation_indices[type_i]
            type_encoding[type_i, valid_allocations] = True

        ### Get binary encoding of constraint type assignment
        for z in range(pvec.allocations.size):
            selection = np.where(
                pvec.parameter_manager.force_ranks == z
                )[0]
            sys_idx = pvec.parameter_manager.system_idx_list[selection[0]]
            sys = pvec.parameter_manager.system_list[sys_idx]
            rdmol = sys.rdmol
            atom_list = pvec.parameter_manager.atom_list[selection[0]]
            atom_list = tuple(atom_list)
            for constraint_i, constraint_mol in enumerate(self.rdmol_list):
                matches = rdmol.GetSubstructMatches(
                    constraint_mol
                    )
                for match in matches:
                    match = tuple(match)
                    if match == atom_list:
                        constraints_encoding[constraint_i,z] = True
                    elif match[::-1] == atom_list:
                        constraints_encoding[constraint_i,z] = True
                    else:
                        continue

        return type_encoding, constraints_encoding


    def apply_constraints(self, pvec):

        """
        Given a parameter vector, unravel the allocations
        according to the constraints (i.e. SMARTS defintions
        and their hierarchy) and apply the constraints.
        """

        _, constraints_encoding = self.get_binary_encoding(pvec)

        diff = pvec.force_group_count - self.num_constraints
        ### Case when we have more constraints then types
        if diff < 0:
            for _ in range(abs(diff)):
                pvec.duplicate(0)

        ### First figure out the weighted average parameter values
        ### for each type after applying the constraints
        new_value_list = list()
        for constraint_i in range(self.num_constraints):
            sele = constraints_encoding[constraint_i]
            value_list = list()
            for z, s in enumerate(sele):
                if s:
                    values = pvec.get_parameters_by_force_group(
                        pvec.allocations[z],
                        get_all_parms=False
                        )
                    value_list.append(values)
            new_values = np.mean(value_list, axis=0)
            new_value_list.append(new_values)

        ### Allocate everything and set the parameter values
        for constraint_i in range(self.num_constraints):
            pvec.set_parameters_by_force_group(
                constraint_i,
                [0. for _ in range(pvec.parameters_per_force_group)],
                new_value_list[constraint_i]
                )
            sele = constraints_encoding[constraint_i]
            pvec.allocations[sele] = constraint_i
            pvec.apply_changes()

        ### Case when we have less constraints then types
        if diff > 0:
            ### Remove any type that is left over.
            for _ in range(diff):
                pvec.remove(-1)


class AllocationScorePrior(object):

    def __init__(
        self, 
        chemistry_constraint_group_list = list(),
        parameter_manager = None,
        ):

        """
        This works in principal similar to `FingerprintAllocationScorer` but instead
        the result is interpreted as a log-probability. Also this class uses more caching
        of intermediate results and some pre-processing so that at runtime it is much faster
        and more appropriate if many iterations/calls are necessary (as for example in 
        sampling procedures).
        """

        from rdkit import Chem

        self.chemistry_constraint_group_list = chemistry_constraint_group_list

        self.hybridization_type_dict = {
            Chem.HybridizationType.SP3 : 0,
            Chem.HybridizationType.SP2 : 1,
            Chem.HybridizationType.SP  : 2,
            Chem.HybridizationType.S   : 3,
        }

        self.fingerprint_dict = dict()
        self.chemistry_dict   = dict()

        from .parameters import ParameterManager
        if isinstance(parameter_manager, ParameterManager):
            self.pre_process(parameter_manager)


    def pre_process(
        self, 
        parameter_manager, 
        maxLength = 2
        ):

        max_rank = parameter_manager.max_rank
        if max_rank < 0:
            raise ValueError(
                "ParameterManager not properly initialized. Cannot pre-process."
                )

        from rdkit.Chem import AllChem as Chem

        system_list = parameter_manager.system_list

        self.fingerprint_dict = dict()
        self.chemistry_dict   = dict()
        self.chemistry_constraint_group_dict = dict()

        for constraint_group_key in np.unique(self.chemistry_constraint_group_list):
            valids = np.where(
                self.chemistry_constraint_group_list == constraint_group_key
                )[0]
            self.chemistry_constraint_group_dict[constraint_group_key] = valids

        for rank in range(max_rank+1):
            sele = np.where(rank == parameter_manager.force_ranks)[0][0]
            atom_list = parameter_manager.atom_list[sele]
            atom_list = np.array(atom_list)
            sys_idx = parameter_manager.system_idx_list[sele]
            rdmol = system_list[sys_idx].rdmol
            fp = Chem.GetHashedAtomPairFingerprintAsBitVect(
                rdmol,
                maxLength = maxLength,
                fromAtoms = atom_list.tolist()
                )
            self.fingerprint_dict[rank] = fp
            self.chemistry_dict[rank] = dict()

            for constraint_group_key in self.chemistry_constraint_group_dict:
                self.chemistry_dict[rank][constraint_group_key] = dict()
                self.chemistry_dict[rank][constraint_group_key]["hybridization"] = list()
                self.chemistry_dict[rank][constraint_group_key]["atoms"] = list()
                self.chemistry_dict[rank][constraint_group_key]["bonds"] = list()
                list_idx_list = self.chemistry_constraint_group_dict[constraint_group_key]
                N_atom_idxs   = len(list_idx_list)
                for atom_idx_i in range(N_atom_idxs):
                    atom_idx1 = atom_list[list_idx_list[atom_idx_i]]
                    atom = rdmol.GetAtomWithIdx(
                        int(atom_idx1)
                        )
                    self.chemistry_dict[rank][constraint_group_key]["atoms"].append(
                        atom.GetAtomicNum()
                        )
                    self.chemistry_dict[rank][constraint_group_key]["hybridization"].append(
                        atom.GetHybridization()
                        )
                    for atom_idx_j in range(N_atom_idxs):
                        if atom_idx_j == atom_idx_i:
                            continue
                        atom_idx2 = atom_list[list_idx_list[atom_idx_j]]
                        bond = rdmol.GetBondBetweenAtoms(
                            int(atom_idx1),
                            int(atom_idx2),
                            )
                        if bond == None:
                            self.chemistry_dict[rank][constraint_group_key]["bonds"].append(
                                0.
                                )
                        else:
                            self.chemistry_dict[rank][constraint_group_key]["bonds"].append(
                                bond.GetBondTypeAsDouble()
                                )
                ### Within a group the order of atoms does not matter.
                ### Sort them by element such that any group is selected deterministically.
                self.chemistry_dict[rank][constraint_group_key]["atoms"] = sorted(
                    self.chemistry_dict[rank][constraint_group_key]["atoms"]
                    )
                self.chemistry_dict[rank][constraint_group_key]["hybridization"] = sorted(
                    self.chemistry_dict[rank][constraint_group_key]["hybridization"]
                    )
                self.chemistry_dict[rank][constraint_group_key]["bonds"] = sorted(
                    self.chemistry_dict[rank][constraint_group_key]["bonds"]
                    )


    def __call__(
        self,
        allocations,
        ):

        import numpy as np
        from rdkit import DataStructs

        max_alloc = max(allocations)
        N_allocs  = allocations.size
        allocation_score_prior = 0.
        for z in range(N_allocs):
            t  = allocations[z]
            fp = self.fingerprint_dict[z]
            fp_list_self  = list()
            fp_list_other = list()
            found_constraint_violation = False
            for _z in range(N_allocs):
                if z == _z:
                    continue
                _t = allocations[_z]
                if t == _t:
                    is_match_list = list()
                    for constraint_group_key in self.chemistry_constraint_group_dict:
                        ### The wildcard (i.e. == 0) is always True
                        if constraint_group_key == 0:
                            is_match_list.append(True)
                        else:
                            constraint_values_1 = self.chemistry_dict[z][constraint_group_key]
                            constraint_values_2 = self.chemistry_dict[_z][constraint_group_key]
                            if (constraint_values_1["atoms"] == constraint_values_2["atoms"]) and \
                            (constraint_values_1["hybridization"] == constraint_values_2["hybridization"]) and \
                            (constraint_values_1["bonds"] == constraint_values_2["bonds"]):
                                is_match_list.append(True)
                            else:
                                is_match_list.append(False)

                    if not all(is_match_list):
                        found_constraint_violation = True
                    else:
                        fp_list_self.append(
                            self.fingerprint_dict[_z]
                            )
                else:
                    fp_list_other.append(
                        self.fingerprint_dict[_z]
                        )
                if found_constraint_violation:
                    break

            if not found_constraint_violation:
                sim_self = 0.
                sim_other = 0.
                sim_self_min = 0.
                sim_other_min = 0.
                sim_self_max = 0.
                sim_other_max = 0.
                if fp_list_self:
                    sim_self = DataStructs.BulkTanimotoSimilarity(
                        fp,
                        fp_list_self,
                    )
                    sim_self_min = np.min(sim_self)
                    sim_self_max = np.max(sim_self)
                    sim_self = np.mean(sim_self)
                if fp_list_other:
                    sim_other = DataStructs.BulkTanimotoSimilarity(
                        fp,
                        fp_list_other,
                    )
                    sim_other_min = np.min(sim_other)
                    sim_other_max = np.max(sim_other)
                    sim_other = np.mean(sim_other)
                sim_total  = sim_self + 1. - sim_other
                sim_total += sim_self_min + 1. - sim_other_min
                sim_total += sim_self_max + 1. - sim_other_max
                allocation_score_prior += sim_total

        allocation_score_prior *= 20.

        return allocation_score_prior


def calc_typing_log_prior(
    allocations_list,
    force_group_count_list,
    alpha_list = None,
    weight_list = None,
    sigma_list = None,
    allocation_scorer_list = None,
    typing_prior="multinomial-dirichlet",
    ):

    import numpy as np
    import copy
    from scipy import special
    from scipy import stats

    choices_typing_prior = [
            "multinomial",
            "multinomial-dirichlet",
            "multinomial-conditional",
            "multinomial-dirichlet-conditional",
            "jeffreys",
            "allocation-score"
            ]
    typing_prior = typing_prior.lower()

    if not typing_prior in choices_typing_prior:
        raise ValueError(
            f"Typing prior must be one of {choices_typing_prior}"
            )

    allocations_list_cp = copy.deepcopy(allocations_list)

    N_pvec = len(allocations_list_cp)

    if isinstance(alpha_list, type(None)):
        alpha_list = [None for _ in range(N_pvec)]
    if isinstance(weight_list, type(None)):
        weight_list = [None for _ in range(N_pvec)]
    if isinstance(sigma_list, type(None)):
        sigma_list = [None for _ in range(N_pvec)]
    if isinstance(allocation_scorer_list, type(None)):
        allocation_scorer_list = [None for _ in range(N_pvec)]

    def get_force_group_histogram(a, f):

        hist = np.zeros(
            f, dtype=int
            )
        for i in range(f):
            alloc_pos = np.where(a == i)[0]
            hist[i]   = alloc_pos.size

        return hist

    log_prior_alloc = 0.
    if typing_prior == "multinomial" or typing_prior == "multinomial-conditional":
        for pvec_idx in range(N_pvec):

            allocations = allocations_list_cp[pvec_idx]
            force_group_count = force_group_count_list[pvec_idx]
            if force_group_count == 0:
                continue

            if typing_prior == "multinomial-conditional":
                histogram = get_force_group_histogram(allocations, force_group_count)
                for z_i, z in enumerate(allocations):
                    if z == _INACTIVE_GROUP_IDX:
                        continue
                    x = float(histogram[z] - 1.)
                    w = weight_list[pvec_idx][z]
                    logp  = x * np.log(w)
                    logp -= np.log(x + 1.)
                    logN_list = np.zeros(force_group_count, dtype=float)
                    for t in range(force_group_count):
                        if t == z:
                            x = float(histogram[t] - 1)
                        else:
                            x = float(histogram[t])
                        w = weight_list[pvec_idx][t]
                        logN_list[t] += x * np.log(w)
                        logN_list[t] -= np.log(x + 1.)
                    logN = logN_list[0]
                    for p in logN_list[1:]:
                        logN = np.logaddexp(p, logN)                    
                    log_prior_alloc += logp
                    log_prior_alloc -= logN
            else:
                mn = stats.multinomial(
                    force_group_count,
                    weight_list[pvec_idx]
                    )
                log_prior_alloc += mn.logpmf(
                    force_group_histogram
                    )

    elif typing_prior == "multinomial-dirichlet" or typing_prior == "multinomial-dirichlet-conditional":
        for pvec_idx in range(N_pvec):

            allocations = allocations_list_cp[pvec_idx]
            force_group_count = force_group_count_list[pvec_idx]

            if typing_prior == "multinomial-dirichlet-conditional":
                histogram = get_force_group_histogram(allocations, force_group_count)
                for z_i, z in enumerate(allocations):
                    if z == _INACTIVE_GROUP_IDX:
                        continue
                    x  = float(histogram[z] - 1)
                    a  = alpha_list[pvec_idx][z]
                    logp  = np.log(x + a)
                    logp -= np.log(x + 1.)
                    logN_list = np.zeros(force_group_count, dtype=float)
                    for t in range(force_group_count):
                        if t == z:
                            x = float(histogram[t] - 1)
                        else:
                            x = float(histogram[t])
                        a  = alpha_list[pvec_idx][t]
                        logN_list[t] += np.log(x + a)
                        logN_list[t] -= np.log(x + 1.)
                    logN = logN_list[0]
                    for p in logN_list[1:]:
                        logN = np.logaddexp(p, logN)                    
                    log_prior_alloc += logp
                    log_prior_alloc -= logN
            else:
                def get_log_beta(x):
                    
                    import numpy as np
                    from scipy import special
                    
                    b1 = np.sum(special.gammaln(x))
                    b2 = special.gammaln(np.sum(x))
                    return b1-b2

                from scipy import special
                log_prior_alloc -= get_log_beta(alpha_list[pvec_idx])
                log_prior_alloc += get_log_beta(alpha_list[pvec_idx] + force_group_histogram)
                log_prior_alloc += np.sum(special.gammaln(force_group_histogram + 1))

    elif typing_prior == "jeffreys":
        for pvec_idx in range(N_pvec):

            allocations = allocations_list_cp[pvec_idx]
            force_group_count = force_group_count_list[pvec_idx]

            log_prior_alloc -= np.sum(
                np.log(
                    sigma_list[pvec_idx]
                    )
                )
    elif typing_prior == "allocation-score":
        log_prior_alloc = 0.
        for pvec_idx in range(N_pvec):

            allocation_scorer = copy.deepcopy(
                allocation_scorer_list[pvec_idx]
                )
            allocations = allocations_list_cp[pvec_idx]
            log_prior_alloc += allocation_scorer(allocations)

    else:
        raise ValueError(
            f"Prior {typing_prior} not known."
            )

    if np.isinf(log_prior_alloc) or np.isnan(log_prior_alloc):
        return -np.inf
    else:
        return log_prior_alloc


def calc_parameter_log_prior(
    pvec_list,
    bounds_list = None,
    sigma_parameter_list = None,
    parameter_prior="gaussian",
    ):

    import numpy as np

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
    alpha_list = None,
    weight_list = None,
    sigma_parameter_list = None,
    sigma_list = None,
    parameter_prior = "gaussian",
    typing_prior = "multinomial-dirichlet",
    ):

    log_prior_alloc = calc_typing_log_prior(
        [pvec.allocations[:] for pvec in pvec_list],
        [pvec.force_group_count for pvec in pvec_list],
        alpha_list = alpha_list,
        weight_list = weight_list,
        sigma_list = sigma_list,
        typing_prior = typing_prior,
        )

    log_prior_val = calc_parameter_log_prior(
        pvec_list,
        bounds_list = bounds_list,
        sigma_parameter_list = sigma_parameter_list,
        parameter_prior = parameter_prior,
        )

    return log_prior_val, log_prior_alloc


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
    cache_dict = dict()
    for idx in range(N_queries):
        typing = typing_list[idx]
        typing_tuple = tuple(typing)
        if typing_tuple in cache_dict:
            logL_list[idx] = logL_list[cache_dict[typing_tuple]]
        else:
            cache_dict[typing_tuple] = idx
            pvec_cp.allocations[:] = typing[:]
            pvec_cp.apply_changes()
            worker_id_list = [targetcomputer(ommdict, False) for ommdict in openmm_system_list]
            logP_likelihood = 0.
            while worker_id_list:
                worker_id, worker_id_list = ray.wait(worker_id_list)
                _logP_likelihood = ray.get(worker_id[0])
                logP_likelihood += _logP_likelihood
            logL_list[idx] = logP_likelihood
    pvec_cp.allocations[:] = init_alloc
    pvec_cp.apply_changes()

    return logL_list


def draw_typing_vector(
    pvec_list_in,
    targetcomputer,
    typing_constraints = None,
    switching = None,
    N_attempts = 1,
    alpha_list = None,
    weight_list = None,
    sigma_list = None,
    allocation_scorer_list = None,
    typing_prior = "multinomial-dirichlet-conditional",
    parallel_targets = True,
    draw_maximum = False,
    smarts_typing = None
    ):

    import copy
    import numpy as np
    import ray

    pvec_list = copy.deepcopy(pvec_list_in)

    N_pvec = len(pvec_list)
    if isinstance(alpha_list, type(None)):
        alpha_list = [None for _ in range(N_pvec)]
    if isinstance(weight_list, type(None)):
        weight_list = [None for _ in range(N_pvec)]
    if isinstance(sigma_list, type(None)):
        sigma_list = [None for _ in range(N_pvec)]
    if isinstance(allocation_scorer_list, type(None)):
        allocation_scorer_list = [None for _ in range(N_pvec)]
    if isinstance(switching, type(None)):
        switching = [False for _ in range(N_pvec)]
    if isinstance(smarts_typing, type(None)):
        smarts_typing = [False for _ in range(N_pvec)]
    if isinstance(typing_constraints, type(None)):
        typing_constraints = [False for _ in range(N_pvec)]

    if parallel_targets:
        _batch_likelihood_typing = ray.remote(batch_likelihood_typing)
        _calc_typing_log_prior   = ray.remote(calc_typing_log_prior)
    else:
        _batch_likelihood_typing = batch_likelihood_typing
        _calc_typing_log_prior   = calc_typing_log_prior

    pvec_pvec_list = list()
    pvec_sysidx_list = list()
    system_list = pvec_list[0].parameter_manager.system_list
    for pvec_idx in range(N_pvec):
        pvec_pvec_list.append(list())
        pvec_sysidx_list.append(list())
        if not smarts_typing[pvec_idx]:
            pvec_big  = pvec_list[pvec_idx]
            for sys_idx in range(pvec_big.N_systems):
                to_delete = np.arange(pvec_big.N_systems, dtype=int)[
                    sys_idx != np.arange(pvec_big.N_systems, dtype=int)
                ].tolist()
                pvec = pvec_big.copy(
                    include_systems=True, 
                    rebuild_to_old_systems=True
                    )
                system_list
                valids_sysidx = np.where(
                    pvec.parameter_manager.system_idx_list == sys_idx
                    )
                valids_forceranks = np.unique(
                    pvec.parameter_manager.force_ranks[valids_sysidx]
                    ).tolist()
                pvec_sysidx_list[-1].append(valids_forceranks)
                pvec.remove_system(to_delete)
                pvec_pvec_list[-1].append(pvec)

    ### Here we synchronize the prior calculations. This takes longer
    ### but is more accurate. Except for "jeffreys" prior where it 
    ### makes exactly no difference.
    for _ in range(N_attempts):
        pvec_schedule = np.arange(N_pvec)
        np.random.shuffle(pvec_schedule)
        for pvec_idx in pvec_schedule:
            if smarts_typing[pvec_idx]:
                ff_parameter_vector  = pvec_list[pvec_idx]
                type_schedule = np.arange(
                    ff_parameter_vector.force_group_count, 
                    dtype=int
                    )
                np.random.shuffle(type_schedule)
                for type_j in type_schedule:
                    selection_j = ff_parameter_vector.get_smarts_indices(int(type_j))
                    active_selection_j = np.where(
                        ff_parameter_vector.smarts_allocations[selection_j] != _INACTIVE_GROUP_IDX
                        )[0]
                    N_actives   = active_selection_j.size
                    for pos_i in range(N_actives):
                        pos_1 = selection_j[pos_i]
                        for pos_2 in selection_j:
                            if pos_2 == pos_1:
                                continue
                            if ff_parameter_vector.smarts_allocations[pos_2] != _INACTIVE_GROUP_IDX:
                                continue
                            allocations_start = copy.deepcopy(ff_parameter_vector.allocations)
                            max_alloc = ff_parameter_vector.get_max_allocation(pos_2)
                            val       = _INACTIVE_GROUP_IDX
                            worker_id_logP_dict = dict()
                            allocations_list = list()
                            state_start = ff_parameter_vector.copy()
                            state_list = list()
                            while (max_alloc > 0 and val < max_alloc) or val == _INACTIVE_GROUP_IDX:
                                ff_parameter_vector.set_allocation(pos_2, val)

                                ff_parameter_vector.apply_changes(
                                    flush_to_systems=False,
                                    flush_to_allocations=True
                                    )

                                allocations = ff_parameter_vector.allocations
                                diff = allocations[:] - allocations_start[:]
                                
                                allocations_list.append(
                                    allocations[:]
                                    )
                                if parallel_targets:
                                    worker_id_logP = _calc_typing_log_prior.remote(
                                        [np.array(allocations[:])],
                                        [ff_parameter_vector.force_group_count],
                                        alpha_list = [alpha_list[pvec_idx]],
                                        weight_list = [weight_list[pvec_idx]],
                                        sigma_list = [sigma_list[pvec_idx]],
                                        allocation_scorer_list = [allocation_scorer_list[pvec_idx]],
                                        typing_prior = typing_prior,
                                        )
                                    worker_id_logP_dict[val] = worker_id_logP
                                else:
                                    logP = _calc_typing_log_prior(
                                        [np.array(allocations[:])],
                                        [ff_parameter_vector.force_group_count],
                                        alpha_list = [alpha_list[pvec_idx]],
                                        weight_list = [weight_list[pvec_idx]],
                                        sigma_list = [sigma_list[pvec_idx]],
                                        allocation_scorer_list = [allocation_scorer_list[pvec_idx]],
                                        typing_prior = typing_prior,
                                        )
                                    worker_id_logP_dict[val] = logP

                                state_list.append(
                                    ff_parameter_vector.copy()
                                    )
                                if val == _INACTIVE_GROUP_IDX:
                                    val = 0
                                else:
                                    val += 1
                                max_alloc = ff_parameter_vector.get_max_allocation(pos_2)

                            if len(allocations_list) == 0:
                                continue
                            if parallel_targets:
                                worker_id = _batch_likelihood_typing.remote(
                                    ff_parameter_vector,
                                    targetcomputer,
                                    allocations_list,
                                    rebuild_from_systems=True,
                                    )
                                logL_list = ray.get(worker_id)
                            else:
                                logL_list = _batch_likelihood_typing(
                                    ff_parameter_vector,
                                    targetcomputer,
                                    allocations_list,
                                    rebuild_from_systems=False,
                                    )
                            logP_list = np.copy(logL_list)

                            for counts, val in enumerate(worker_id_logP_dict):
                                if parallel_targets:
                                    worker_id = worker_id_logP_dict[val]
                                    logP = ray.get(worker_id)
                                else:
                                    logP = worker_id_logP_dict[val]
                                #logP_list[counts] += logP
                            logP_sum = logP_list[0]
                            for p in logP_list[1:]:
                                logP_sum = np.logaddexp(p, logP_sum)
                            logP_list -= logP_sum
                            P_list = np.exp(logP_list)

                            if not np.isnan(P_list).any():
                                if draw_maximum:
                                    state_new = state_list[np.argmax(P_list)]
                                else:
                                    idx_new = np.random.choice(
                                        np.arange(len(state_list)), 
                                        size=None, 
                                        p=P_list
                                        )
                                    state_new = state_list[idx_new]
                                ff_parameter_vector.reset(state_new)
                                ff_parameter_vector.apply_changes(True, True)
                            else:
                                ff_parameter_vector.reset(state_start)
                                ff_parameter_vector.apply_changes(True, True)

            else:
                pvec_big  = pvec_list[pvec_idx]
                N_systems = pvec_big.N_systems
                system_schedule = np.arange(N_systems, dtype=int)
                np.random.shuffle(system_schedule)
                allocation_schedule_dict = dict()
                max_alloc = 0
                ### Create sampling schedule and figure out
                ### maximum allocation size.
                for sys_idx in system_schedule:
                    pvec = pvec_pvec_list[pvec_idx][sys_idx]
                    _max_alloc = pvec.allocations.size

                    pvec_allocation_schedule = np.arange(_max_alloc)
                    np.random.shuffle(pvec_allocation_schedule)
                    allocation_schedule_dict[sys_idx] = pvec_allocation_schedule.tolist()
                    if _max_alloc > max_alloc:
                        max_alloc = _max_alloc
                ### Fill empty allocations for those pvec that
                ### have fewer allocations than max_alloc
                for sys_idx in system_schedule:
                    _max_alloc = len(allocation_schedule_dict[sys_idx])
                    diff = abs(_max_alloc - max_alloc)
                    if diff > 0:
                        [allocation_schedule_dict[sys_idx].append(None) for _ in range(diff)]

                type_list_dict = dict()
                for z_i in range(max_alloc):
                    worker_id_dict = dict()
                    logL_dict      = {sys_idx : list() for sys_idx in system_schedule}
                    for sys_idx in system_schedule:
                        allocations_list = list()
                        z = allocation_schedule_dict[sys_idx][z_i]
                        if z == None:
                            continue
                        pvec = pvec_pvec_list[pvec_idx][sys_idx]
                        tc = typing_constraints[pvec_idx]
                        if isinstance(tc, type(None)):
                            available_types = {}
                            for z in range(pvec.allocations.size):
                                available_types[z] = np.arange(
                                    pvec.force_group_count, 
                                    dtype=int
                                    ).tolist()
                                if switching[pvec_idx]:
                                    available_types[z].append(
                                        _INACTIVE_GROUP_IDX
                                        )
                        else:
                            available_types = tc.query_available_types(pvec)
                            
                        type_list = available_types[z]
                        type_list_dict[(sys_idx, z)] = type_list
                        for t in type_list:
                            allocations = np.copy(pvec.allocations[:])
                            allocations[z] = t
                            allocations_list.append(allocations)
                        if len(allocations_list) > 0:
                            if parallel_targets:
                                worker_id = _batch_likelihood_typing.remote(
                                    pvec,
                                    targetcomputer,
                                    allocations_list,
                                    rebuild_from_systems=True,
                                    )
                                worker_id_dict[worker_id] = (sys_idx, z)
                            else:
                                logL_list = _batch_likelihood_typing(
                                    pvec,
                                    targetcomputer,
                                    allocations_list,
                                    rebuild_from_systems=False,
                                    )
                                logL_dict[sys_idx].append((logL_list, z))

                    if parallel_targets:
                        worker_list = list(worker_id_dict.keys())
                        while worker_list:
                            worker_id, worker_list = ray.wait(worker_list)
                            worker_id = worker_id[0]
                            sys_idx, z = worker_id_dict[worker_id]
                            logL_list = ray.get(worker_id)
                            logL_dict[sys_idx].append((logL_list, z))
                    for sys_idx in system_schedule:
                        if not sys_idx in logL_dict:
                            continue
                        pvec = pvec_pvec_list[pvec_idx][sys_idx]
                        force_ranks = pvec_sysidx_list[pvec_idx][sys_idx]
                        for logL_list, z in logL_dict[sys_idx]:
                            type_list = type_list_dict[(sys_idx, z)]
                            logP_list = np.copy(logL_list)
                            worker_id_logP_dict = dict()
                            for t_idx in range(len(type_list)):
                                t_old = pvec_big.allocations[force_ranks[z]]
                                pvec_big.allocations[force_ranks[z]] = type_list[t_idx]
                                if parallel_targets:
                                    worker_id_logP = _calc_typing_log_prior.remote(
                                        [np.copy(pvec_big.allocations[:])],
                                        [pvec_big.force_group_count],
                                        alpha_list = [alpha_list[pvec_idx]],
                                        weight_list = [weight_list[pvec_idx]],
                                        sigma_list = [sigma_list[pvec_idx]],
                                        allocation_scorer_list = [allocation_scorer_list[pvec_idx]],
                                        typing_prior = typing_prior,
                                        )
                                    worker_id_logP_dict[t_idx] = worker_id_logP
                                else:
                                    logP = _calc_typing_log_prior(
                                        [np.copy(pvec_big.allocations[:])],
                                        [pvec_big.force_group_count],
                                        alpha_list = [alpha_list[pvec_idx]],
                                        weight_list = [weight_list[pvec_idx]],
                                        sigma_list = [sigma_list[pvec_idx]],
                                        allocation_scorer_list = [allocation_scorer_list[pvec_idx]],
                                        typing_prior = typing_prior,
                                        )
                                    logP_list[t_idx] += logP
                                pvec_big.allocations[force_ranks[z]] = t_old
                            if parallel_targets:
                                for t_idx in range(len(type_list)):
                                    logP = ray.get(worker_id_logP_dict[t_idx])
                                    logP_list[t_idx] += logP
                            logP_sum = logP_list[0]
                            for p in logP_list[1:]:
                                logP_sum = np.logaddexp(p, logP_sum)
                            logP_list -= logP_sum
                            P_list = np.exp(logP_list)

                            type_start = pvec.allocations[z]
                            if not np.isnan(P_list).any():
                                type_list = type_list_dict[(sys_idx, z)]
                                if draw_maximum:
                                    type_new = type_list[np.argmax(P_list)]
                                else:
                                    type_new = np.random.choice(
                                        type_list, 
                                        size=None, 
                                        p=P_list
                                        )
                                pvec.allocations[z] = type_new
                                pvec_big.allocations[force_ranks[z]] = type_new
                                ### No need to call apply_changes at this point
                                ### for pvec_big
                                pvec.apply_changes()
                            else:
                                pvec.allocations[z] = type_start
                                pvec_big.allocations[force_ranks[z]] = type_start
                                pvec.apply_changes()

    allocations_list = list()
    state_list = list()
    for pvec_idx in range(N_pvec):
        pvec_big = pvec_list[pvec_idx]
        allocations_list.append(
            copy.deepcopy(pvec_big.allocations[:])
            )
        state_list.append(
            pvec_big.copy()
            )

    return allocations_list, state_list


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


def pre_smirksify(
    molecules, 
    cluster_definitions, 
    max_layers=3
    ):

    from chemper.smirksify import ClusterGraph, SMIRKSifier
    import ray

    N_molecules = len(molecules)
    N_clusters  = len(cluster_definitions)

    def get_smirks_len(clust):
        
        cluster_graph = ClusterGraph(
            molecules,
            clust,
            layers=1
        )
        smirks = cluster_graph.as_smirks(compress=False)
        return len(smirks)


    def generate_cluster_definitions(merge_list):
        cluster_definitions_new = list()
        for merge_idx in merge_list:
            new_name  = ""
            new_clust = [list() for _ in range(N_molecules)]
            for cluster_idx in merge_idx:
                name, cluster = cluster_definitions[cluster_idx]
                if new_name:
                    new_name = f"{new_name}/{name}"
                else:
                    new_name = f"{name}"
                for mol_idx in range(N_molecules):
                    new_clust[mol_idx].extend(
                        cluster[mol_idx]
                    )

            cluster_definitions_new.append(
                (new_name, new_clust)
            )
        ### Sort from longest do shortest SMIRKS pattern.
        ### This should generate the most reliable ordering
        ### of SMIRKS patterns. See also ChemPer paper: chemrxiv.8304578.v1
        new_merge_cluster_definitions = sorted(
            zip(merge_list, cluster_definitions_new), 
            key=lambda x: get_smirks_len(x[1][1]),
            reverse=True
        )

        new_merge_list = [m for m, c in new_merge_cluster_definitions]
        cluster_definitions_new = [c for m, c in new_merge_cluster_definitions]

        return cluster_definitions_new, new_merge_list
    

    def will_it_smirksify(cluster_definitions_new):

        import os, sys

        ### Redirect print output from SMIRKSifier.
        ### We want to suppress the print("WARNING")
        ### messages.
        sys.stdout = open(os.devnull, 'w')
        smirksifier = SMIRKSifier(
            molecules,
            cluster_definitions_new,
            max_layers=max_layers,
            strict_smirks=False,
            verbose=False
        )
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        return smirksifier.checks
    
    will_it_smirksify_remote = ray.remote(will_it_smirksify)

    ### First try to find all clusters that work.
    add_list = list()
    merge_list =  list()
    for cluster_idx in range(N_clusters):
        add_list.append(
            [cluster_idx]
        )
        result = will_it_smirksify(
            generate_cluster_definitions(
                add_list
            )[0]
        )
        if not result:
            merge_list.append(
                cluster_idx
            )
            add_list.pop(-1)
    ### Then try find the clusters that will
    ### have a conflict with each other.
    ### Then merge them.
    for cluster_idx_merge in merge_list:
        worker_id_dict = dict()
        len_add_list = len(add_list)
        for idx in range(len_add_list):
            if cluster_idx_merge in add_list[idx]:
                continue
            add_list[idx].append(
                cluster_idx_merge
            )
            worker_id = will_it_smirksify_remote.remote(
                generate_cluster_definitions(
                    add_list
                )[0]
            )
            worker_id_dict[worker_id] = idx
            add_list[idx].pop(-1)
        not_ready = list(worker_id_dict.keys())
        while not_ready:
            ready, not_ready = ray.wait(not_ready)
            for worker_id in ready:
                result = ray.get(worker_id)
                if result:
                    idx = worker_id_dict[worker_id]
                    add_list[idx].append(
                        cluster_idx_merge
                    )
                    not_ready = list()
                    break

    cluster_definitions_new, add_list = generate_cluster_definitions(add_list)
    return add_list, cluster_definitions_new


def unravel_constraints(
    scorer, 
    parameter_manager, 
    max_attempts=100,
    apply_to_parameter_manager=True,
    ):

    from .vectors import ForceFieldParameterVector
    import numpy as np

    __doc__ = """
    Returns allocation vector corresponding to a `ForceField.allocations` vector
    that cannot be further merged (i.e. the number of types reduced) without
    violating the constraints defined in `scorer`.
    """

    alloc = np.arange(
        parameter_manager.max_rank+1, 
        dtype=int
        )
    found_new_split = True
    for _ in range(max_attempts):
        N_types   = np.unique(alloc).size
        found_new_split = False
        for type_i in range(N_types):
            for type_j in range(type_i+1,N_types):
                sele_i = np.where(alloc == type_i)[0]
                sele_j = np.where(alloc == type_j)[0]
                sele   = np.append(sele_i, sele_j)
                k_values_ij = np.zeros_like(sele, dtype=int)+type_i
                score  = scorer(
                    parameter_manager,
                    None,
                    type_i,
                    sele,
                    k_values_ij
                )
                if score > 0.:
                    found_new_split = True
                    alloc[sele] = k_values_ij
                    valids = np.where(alloc > type_j)
                    alloc[valids] -= 1
                    break
                if found_new_split:
                    break
            if found_new_split:
                break
        if not found_new_split:
            break

    if apply_to_parameter_manager and alloc.size > 0:
        pvec = ForceFieldParameterVector(parameter_manager)
        while pvec.force_group_count < (alloc.max()+1):
            pvec.duplicate(0)
        pvec[:] += np.random.normal(0., 1.e-2, pvec.size)
        pvec.allocations[:] = alloc
        pvec.apply_changes()

    return alloc


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

        self._lower = [-10.0 * _FORCE_CONSTANT_TORSION]
        self._upper = [+10.0 * _FORCE_CONSTANT_TORSION]

    def get_bounds(self, atom_list, rdmol):

        return self._lower, self._upper


class DoubleTorsionBounds(object):

    def __init__(self, parameter_name_list=list()):

        self._lower = [-10.0 * _FORCE_CONSTANT_TORSION, -10.0 * _FORCE_CONSTANT_TORSION]
        self._upper = [+10.0 * _FORCE_CONSTANT_TORSION, +10.0 * _FORCE_CONSTANT_TORSION]

    def get_bounds(self, atom_list, rdmol):

        return self._lower, self._upper


class MultiTorsionBounds(object):

    def __init__(self, parameter_name_list=list()):

        N_parms = len(parameter_name_list)

        self._lower = [-10.0 * _FORCE_CONSTANT_TORSION for _ in range(N_parms)]
        self._upper = [+10.0 * _FORCE_CONSTANT_TORSION for _ in range(N_parms)]

    def get_bounds(self, atom_list, rdmol):

        return self._lower, self._upper


class FingerprintAllocationScorer(object):

    def __init__(
        self, 
        chemistry_constraint_group_list=list()
        ):

        from rdkit import Chem

        self.chemistry_constraint_group_list = chemistry_constraint_group_list

        self.hybridization_type_dict = {
            Chem.HybridizationType.SP3 : 0,
            Chem.HybridizationType.SP2 : 1,
            Chem.HybridizationType.SP  : 2,
            Chem.HybridizationType.S   : 3,
        }


    def __call__(
        self,
        parameter_manager,
        type_i,
        type_j,
        selection_ij,
        k_values_ij,
        ):

        selection_ij = np.array(selection_ij)
        k_values_ij  = np.array(k_values_ij)

        system_list = parameter_manager.system_list

        from rdkit.Chem import AllChem as Chem
        from rdkit import DataStructs
        maxLength = 2

        fingerprint_dict = dict()
        chemistry_constraint_group_dict = dict()
        chemistry_dict = dict()
        for constraint_group_key in np.unique(self.chemistry_constraint_group_list):
            valids = np.where(
                self.chemistry_constraint_group_list == constraint_group_key
                )[0]
            chemistry_constraint_group_dict[constraint_group_key] = valids

        subselection_i  = np.where(np.array(k_values_ij) == type_i)[0]
        subselection_j  = np.where(np.array(k_values_ij) == type_j)[0]
        subselection_ij = np.append(subselection_i, subselection_j)
        for sele in selection_ij[subselection_ij]:
            if not sele in fingerprint_dict:
                valid_ranks = np.where(sele == parameter_manager.force_ranks)[0]
                sys_idx = parameter_manager.system_idx_list[valid_ranks][0]
                atom_list = copy.deepcopy(parameter_manager.atom_list[valid_ranks[0]])
                atom_list = np.array(atom_list)
                rdmol = system_list[sys_idx].rdmol
                fp = Chem.GetHashedAtomPairFingerprintAsBitVect(
                    rdmol,
                    maxLength = maxLength,
                    fromAtoms = atom_list.tolist()
                    )
                fingerprint_dict[sele] = fp

                assert len(atom_list) == len(self.chemistry_constraint_group_list)

                chemistry_dict[sele] = dict()
                for constraint_group_key in chemistry_constraint_group_dict:
                    chemistry_dict[sele][constraint_group_key] = dict()
                    chemistry_dict[sele][constraint_group_key]["hybridization"] = list()
                    chemistry_dict[sele][constraint_group_key]["atoms"] = list()
                    chemistry_dict[sele][constraint_group_key]["bonds"] = list()
                    list_idx_list = chemistry_constraint_group_dict[constraint_group_key]
                    N_atom_idxs   = len(list_idx_list)
                    for atom_idx_i in range(N_atom_idxs):
                        atom_idx1 = atom_list[list_idx_list[atom_idx_i]]
                        atom = rdmol.GetAtomWithIdx(
                            int(atom_idx1)
                            )
                        chemistry_dict[sele][constraint_group_key]["atoms"].append(
                            atom.GetAtomicNum()
                            )
                        chemistry_dict[sele][constraint_group_key]["hybridization"].append(
                            atom.GetHybridization()
                            )
                        for atom_idx_j in range(N_atom_idxs):
                            if atom_idx_j == atom_idx_i:
                                continue
                            atom_idx2 = atom_list[list_idx_list[atom_idx_j]]
                            bond = rdmol.GetBondBetweenAtoms(
                                int(atom_idx1),
                                int(atom_idx2),
                                )
                            if bond == None:
                                chemistry_dict[sele][constraint_group_key]["bonds"].append(
                                    0.
                                    )
                            else:
                                chemistry_dict[sele][constraint_group_key]["bonds"].append(
                                    bond.GetBondTypeAsDouble()
                                    )
                    ### Within a group the order of atoms does not matter.
                    ### Sort them by element such that any group is selected deterministically.
                    chemistry_dict[sele][constraint_group_key]["atoms"] = sorted(
                        chemistry_dict[sele][constraint_group_key]["atoms"]
                        )
                    chemistry_dict[sele][constraint_group_key]["hybridization"] = sorted(
                        chemistry_dict[sele][constraint_group_key]["hybridization"]
                        )
                    chemistry_dict[sele][constraint_group_key]["bonds"] = sorted(
                        chemistry_dict[sele][constraint_group_key]["bonds"]
                        )

        def get_sim(
            subselection_1, 
            subselection_2, 
            with_constraints):

            sim_12_list = list()

            for sele_1 in subselection_1:
                all_fingerprints_subselection = list()
                for sele_2 in subselection_2:
                    is_match_list = list()
                    ### This case trivial. Don't include it.
                    if with_constraints:
                        if sele_1 == sele_2:
                            continue
                        else:
                            for constraint_group_key in chemistry_constraint_group_dict:
                                ### The wildcard (i.e. == 0) is always True
                                if constraint_group_key == 0:
                                    is_match_list.append(True)
                                else:
                                    constraint_values_2 = chemistry_dict[sele_1][constraint_group_key]
                                    constraint_values   = chemistry_dict[sele_2][constraint_group_key]
                                    if constraint_values_2["atoms"] == constraint_values["atoms"] and \
                                    constraint_values_2["hybridization"] == constraint_values["hybridization"] and \
                                    constraint_values_2["bonds"] == constraint_values["bonds"]:
                                        is_match_list.append(True)
                                    else:
                                        is_match_list.append(False)
                    else:
                        is_match_list.append(True)
                    if all(is_match_list):
                        all_fingerprints_subselection.append(
                            fingerprint_dict[sele_2]
                            )
                    else:
                        ### Only possible with `with_constraints=True`.
                        return -1.
                sim_12 = DataStructs.BulkTanimotoSimilarity(
                    fingerprint_dict[sele_1],
                    all_fingerprints_subselection,
                )

                sim_12_list.extend(sim_12)

            if sim_12_list:
                value = np.mean(sim_12_list)
            else:
                value = 0.

            return value

        dissim_ji = 1. - get_sim(
            selection_ij[subselection_j],
            selection_ij[subselection_i],
            False
            )
        sim_jj = get_sim(
            selection_ij[subselection_j],
            selection_ij[subselection_j],
            True
            )
        if sim_jj < 0.:
            value = 0.
        else:
            value = sim_jj + dissim_ji

        return value


@ray.remote
def get_gradient_scores(
    ff_parameter_vector,
    targetcomputer,
    type_i, # for split/merge
    type_j = None, # for merge/move
    selection_i = None, # for split *only*
    k_values_ij = None, # for split *only*
    split=True,
    grad_diff = 1.e-2,
    ):

    ff_parameter_vector_cp = ff_parameter_vector.copy(
        include_systems = True,
        rebuild_to_old_systems = False
        )

    grad_score      = list()
    grad_norm       = list()
    allocation_list = list()
    selection_list  = list()
    type_list       = list()

    N_types    = ff_parameter_vector_cp.force_group_count

    N_parms    = ff_parameter_vector_cp.parameters_per_force_group
    N_typables = ff_parameter_vector_cp.allocations.size

    ### =================== ###
    ### CARRY OUT SPLITTING ###
    ### =================== ###

    if split:

        if type_j == None:
            type_j = N_types
            ### Create dummy type for gradient assessment later
            ff_parameter_vector_cp.duplicate(type_i)
            ff_parameter_vector_cp.apply_changes()

        first_parm_i = type_i * N_parms
        last_parm_i  = first_parm_i + N_parms

        first_parm_j = type_j * N_parms
        last_parm_j  = first_parm_j + N_parms

        N_comb = k_values_ij.shape[0]

        likelihood_func = LikelihoodVectorized(
            [ff_parameter_vector_cp],
            targetcomputer
            )

        for comb_idx in range(N_comb):
            ff_parameter_vector_cp.allocations[selection_i] = k_values_ij[comb_idx]
            ff_parameter_vector_cp.apply_changes()

            parm_idx_list = list()
            parm_idx_list.extend(range(first_parm_i, last_parm_i))
            parm_idx_list.extend(range(first_parm_j, last_parm_j))
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
            allocation_list.append(tuple(k_values_ij[comb_idx]))
            selection_list.append(tuple(selection_i))
            type_list.append(tuple([type_i, type_j]))

    ### ================= ###
    ### CARRY OUT MERGING ###
    ### ================= ###

    else:

        selection_i = ff_parameter_vector_cp.allocations.index([type_i])[0]

        first_parm_i = type_i * N_parms
        last_parm_i  = first_parm_i + N_parms

        value_list_i   = ff_parameter_vector_cp[first_parm_i:last_parm_i]
        value_list_i_0 = ff_parameter_vector_cp.vector_0[first_parm_i:last_parm_i]
        scaling_i      = ff_parameter_vector_cp.scaling_vector[first_parm_i:last_parm_i]

        if isinstance(type_j, type(None)):
            type_j_list = list(range(N_types))
        elif isinstance(type_j, int):
            type_j_list = [type_j]
        elif isinstance(type_j, list):
            type_j_list = type_j
        elif isinstance(type_j, np.ndarray):
            type_j_list = type_j
        else:
            raise ValueError(
                "type of type_j not understood."
                )

        likelihood_func = LikelihoodVectorized(
            [ff_parameter_vector_cp],
            targetcomputer
            )

        for type_j in type_j_list:
            if type_i == type_j:
                continue

            first_parm_j = type_j * N_parms
            last_parm_j  = first_parm_j + N_parms

            selection_j   = ff_parameter_vector_cp.allocations.index([type_j])[0]
            selection_ij  = np.append(selection_i, selection_j)
            allocation_ij = ff_parameter_vector_cp.allocations[selection_ij]

            ff_parameter_vector_cp.set_parameters_by_force_group(
                type_j,
                value_list_i,
                value_list_i_0
            )
            ff_parameter_vector_cp.apply_changes()

            grad = likelihood_func.grad(
                ff_parameter_vector_cp[:],
                parm_idx_list=[type_i, type_j],
                grad_diff=grad_diff,
                use_jac=False
                )
            grad_i = grad[first_parm_i:last_parm_i]
            grad_j = grad[first_parm_j:last_parm_j]

            norm_i = np.linalg.norm(grad_i)
            norm_j = np.linalg.norm(grad_j)

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
                grad_score.append(grad_ij_diff)
            else:
                grad_score.append(grad_ij_dot)
            grad_norm.append([norm_i, norm_j])
            allocation_list.append(tuple(allocation_ij))
            selection_list.append(tuple(selection_ij))
            type_list.append(tuple([type_i, type_j]))

    return grad_score, grad_norm, allocation_list, selection_list, type_list


@ray.remote
def get_switchon_scores(
    ff_parameter_vector,
    targetcomputer,
    switch_values,
    selection_i,
    k_values_ij,
    grad_diff = 1.e-2,
    ):

    ff_parameter_vector_cp = ff_parameter_vector.copy(
        include_systems = True,
        rebuild_to_old_systems = False
        )

    N_types = ff_parameter_vector_cp.force_group_count
    type_j  = N_types
    N_typables = ff_parameter_vector_cp.allocations.size
    N_parms    = ff_parameter_vector_cp.parameters_per_force_group

    grad_score_list = list()
    grad_norm_list  = list()
    allocation_list = list()
    selection_list  = list()
    type_list       = list()

    N_comb = k_values_ij.shape[0]

    ### Create dummy type for gradient assessment
    ff_parameter_vector_cp.add_force_group(switch_values)
    likelihood_func = LikelihoodVectorized(
        [ff_parameter_vector_cp],
        targetcomputer
    )
    first_parm = type_j * N_parms
    last_parm  = first_parm + N_parms
    parm_idx_list = list(range(first_parm,last_parm))
    for comb_idx in range(N_comb):
        ff_parameter_vector_cp.allocations[selection_i] = k_values_ij[comb_idx]
        ff_parameter_vector_cp.apply_changes()

        ### I think these should always be multiplied by -1
        ### Since we want to minimize -log(L)
        grad_j = -1. * likelihood_func.grad(
            ff_parameter_vector_cp[:],
            parm_idx_list = parm_idx_list,
            use_jac = False,
            )

        norm_j = np.linalg.norm(grad_j)

        grad_score_list.append(tuple([norm_j]))
        grad_norm_list.append(norm_j)
        allocation_list.append(tuple(k_values_ij[comb_idx]))
        selection_list.append(tuple(selection_i))
        type_list.append(type_j)

    return grad_score_list, grad_norm_list, allocation_list, selection_list, type_list


@ray.remote
def minimize_FF(
    system_list,
    targetcomputer,
    pvec_list,
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
        if isinstance(pvec, SmartsForceFieldParameterVector):
            pvec.apply_changes(True, True)
        else:
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
            return fun(x0), pvec_list_cp, time.time() - time_start
        else:
            return fun(x0), pvec_list_cp

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
            return best_f, pvec_list_cp, time.time() - time_start
        else:
            return best_f, pvec_list_cp

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
        return best_f, pvec_list_cp, time.time() - time_start
    else:
        return best_f, pvec_list_cp


@ray.remote
def generate_split_candidates(
    pvec, 
    type_i, 
    type_j, 
    positions, 
    selection_allocation_ij, 
    max_alloc_states=20):

    import itertools
    import numpy as np
    import copy

    state_dict = dict()

    positions = np.array(positions)
    np.random.shuffle(positions)
    positions = positions.tolist()
    N_actives = len(positions)
    size_ij = selection_allocation_ij.size

    ff_parameter_vector = pvec.copy()
    ff_parameter_vector.set_inactive(type_j)
    found_max_for_pos = [False for _ in range(N_actives)]
    pos_i = 0
    while not all(found_max_for_pos):
        pos       = positions[pos_i]
        max_alloc = ff_parameter_vector.get_max_allocation(pos)
        val       = ff_parameter_vector.smarts_allocations[pos]
        if max_alloc == 0:
            val = _INACTIVE_GROUP_IDX
            found_max_for_pos[pos_i] = True
        elif max_alloc > 0 and val == _INACTIVE_GROUP_IDX:
            val = 0
            found_max_for_pos[pos_i] = False
        elif max_alloc > 0 and val == max_alloc-1:
            val = _INACTIVE_GROUP_IDX
            found_max_for_pos[pos_i] = True
        elif max_alloc > 0 and val < max_alloc:
            val += 1
            found_max_for_pos[pos_i] = False
        ff_parameter_vector.set_allocation(
            pos,
            val
        )

        ff_parameter_vector.apply_changes(
            flush_to_systems=False,
            flush_to_allocations=True
            )
        size_i = ff_parameter_vector.allocations.index([type_i])[0].size
        size_j = ff_parameter_vector.allocations.index([type_j])[0].size

        if size_i > 0 and size_j > 0:
            if (size_i + size_j) == size_ij:
                alloc = ff_parameter_vector.allocations[selection_allocation_ij]

                sele_i = np.where(alloc == type_i)
                sele_j = np.where(alloc == type_j)

                alloc_swap = np.copy(alloc)

                alloc_swap[sele_i] = type_j
                alloc_swap[sele_j] = type_i

                alloc_swap = tuple(alloc_swap.tolist())
                alloc = tuple(alloc.tolist())
                if alloc in state_dict:
                    type_j = ff_parameter_vector.smarts_manager_allocations[pos]
                    smarts_manager = ff_parameter_vector.smarts_manager_list[type_j]
                    key = (
                        type_j,
                        copy.deepcopy(tuple(smarts_manager.allocation_state.items())),
                        copy.deepcopy(tuple(smarts_manager.allocations)),
                        )
                    state_dict[alloc].add(key)
                elif alloc_swap in state_dict:
                    alloc = alloc_swap
                    type_j = ff_parameter_vector.smarts_manager_allocations[pos]
                    smarts_manager = ff_parameter_vector.smarts_manager_list[type_j]
                    key = (
                        type_j,
                        copy.deepcopy(tuple(smarts_manager.allocation_state.items())),
                        copy.deepcopy(tuple(smarts_manager.allocations)),
                        )
                    state_dict[alloc].add(key)
                else:
                    type_j = ff_parameter_vector.smarts_manager_allocations[pos]
                    smarts_manager = ff_parameter_vector.smarts_manager_list[type_j]
                    key = (
                        type_j,
                        copy.deepcopy(tuple(smarts_manager.allocation_state.items())),
                        copy.deepcopy(tuple(smarts_manager.allocations)),
                        )
                    state_dict[alloc] = {key}
                while len(state_dict[alloc]) > (10. * max_alloc_states):
                    do_delete = np.random.randint(0, len(state_dict[alloc]))
                    del state_dict[alloc][do_delete]

        if len(state_dict) > max_alloc_states:
            break

        if not found_max_for_pos[pos_i]:
            if pos_i != N_actives-1:
                pos_i += 1
        else:
            if pos_i > 0:
                pos_i -= 1
            else:
                break

    #print(
    #    f"Number of candidate states {len(state_dict)} for",
    #    f"positions {positions} and selection {selection_allocation_ij} ..."
    #    )

    del ff_parameter_vector
    del positions

    return state_dict


@ray.remote
def set_parameters_remote(
    old_pvec_list_init,
    targetcomputer,
    smarts_typing_list,
    worker_id_dict=dict(),
    parm_penalty=1.,
    typing_constraints_list=list(),
    typing_temperature_list=list(),
    verbose=False,
    ):

    import copy
    import numpy as np
    from .molgraphs import get_smarts_score
    import ray

    MAX_AIC = 9999999999999999.
    found_improvement = True
    N_mngr = len(old_pvec_list_init)

    pvec_list = [pvec.copy(include_systems=True) for pvec in old_pvec_list_init]
    old_pvec_list = [pvec.copy() for pvec in old_pvec_list_init]

    system_list = pvec_list[0].parameter_manager.system_list
    for mngr_idx in range(N_mngr):
        ### IMPORANT: We must rebuild this with list of systems
        ###           that is common to all parameter managers.
        pvec_list[mngr_idx].rebuild_from_systems(
            lazy=True, 
            system_list=system_list
            )
        pvec_list[mngr_idx].apply_changes(True, True)

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

    old_AIC = _calculate_AIC(
        pvec_list,
        0,
        [pvec_list[0].allocations[:].tolist()]
        )[0]

    if verbose:
        print(
            "Current best AIC:", old_AIC
            )
        print(
            f"Checking {len(worker_id_dict)} solutions..."
            )
    best_pvec_list = [pvec.copy() for pvec in pvec_list]
    best_ast       = None
    best_AIC       = MAX_AIC
    best_smarts_score = 0.

    ### For each system, find the best solution
    for ast in worker_id_dict:
        result = worker_id_dict[ast]
        if len(result) == 3:
            worker_id, state_list, mngr_idx_main = result
            state_list = tuple(state_list)
            is_smarts_typing_result = True
        else:
            worker_id = result
            is_smarts_typing_result = False
        _, _pvec_list = ray.get(worker_id)
        
        ### `full_reset=True` means we will set all parameter
        ### managers to their optimized values
        ### `full_reset=False` means we will set only the main
        ### parameter manager to its optimized values and all other
        ### are set to the best value.
        for full_reset in [True, False]:
            if (not is_smarts_typing_result) and full_reset:
                continue

            ### First set the parameter values to
            ### their optimized values.
            for mngr_idx in range(N_mngr):
                smarts_typing = smarts_typing_list[mngr_idx]
                if smarts_typing:
                    if full_reset:
                        pvec_list[mngr_idx].reset(
                            _pvec_list[mngr_idx]
                            )
                    else:
                        if mngr_idx == mngr_idx_main:
                            pvec_list[mngr_idx].reset(
                                _pvec_list[mngr_idx]
                                )
                        else:
                            pvec_list[mngr_idx].reset(
                                old_pvec_list[mngr_idx]
                                )
                    pvec_list[mngr_idx].apply_changes(True, True)
                else:
                    pvec_list[mngr_idx].reset(
                        _pvec_list[mngr_idx]
                        )
                    pvec_list[mngr_idx].apply_changes()
            if is_smarts_typing_result:
                typing_dict = dict()
                N_states = len(state_list)
                for state_idx in range(N_states):
                    key = state_list[state_idx]
                    type_j, allocation_state, allocations = key
                    smarts_manager = pvec_list[mngr_idx_main].smarts_manager_list[type_j]
                    for k,v in allocation_state:
                        smarts_manager.allocation_state[k] =v
                    smarts_manager.allocations[:] = allocations[:]
                    
                    valids = pvec_list[mngr_idx_main].smarts_manager_allocations.index([type_j])[0]
                    pvec_list[mngr_idx_main].smarts_allocations[valids] = allocations[:]
                    
                    pvec_list[mngr_idx_main].apply_changes(False, True)
                    typing_list = tuple(pvec_list[mngr_idx_main].allocations[:])
                    if typing_list in typing_dict:
                        typing_dict[typing_list] += (state_idx,)
                    else:
                        typing_dict[typing_list] = (state_idx,)

                AIC_list = _calculate_AIC(
                    pvec_list,
                    mngr_idx_main,
                    list(typing_dict.keys())
                    )

                counts = 0
                for typing_list in typing_dict:
                    new_AIC = AIC_list[counts]
                    accept = False
                    if new_AIC < best_AIC:
                        accept = True
                    counts += 1
                    for state_idx in typing_dict[typing_list]:
                        key = state_list[state_idx]
                        type_j, allocation_state, allocations = key
                        smarts_manager = pvec_list[mngr_idx_main].smarts_manager_list[type_j]
                        for k,v in allocation_state:
                            smarts_manager.allocation_state[k] =v
                        smarts_manager.allocations[:] = allocations[:]
                        
                        valids = pvec_list[mngr_idx_main].smarts_manager_allocations.index([type_j])[0]
                        pvec_list[mngr_idx_main].smarts_allocations[valids] = allocations[:]

                        pvec_list[mngr_idx_main].apply_changes(True, True)

                        if accept:
                            smarts_score = get_smarts_score(smarts_manager)
                            if smarts_score > best_smarts_score:
                                best_smarts_score = smarts_score
                        else:
                            diff = np.abs(new_AIC - best_AIC)
                            if diff < 1.e-4:
                                smarts_score = get_smarts_score(smarts_manager)
                                if smarts_score > best_smarts_score:
                                    best_smarts_score = smarts_score
                                    accept = True

                        if accept:
                            best_AIC       = new_AIC
                            best_pvec_list = [pvec.copy() for pvec in pvec_list]
                            best_ast       = ast
                            
                        #if verbose:
                        #    if accept:
                        #        print("Accepted solution.")
                        #    else:
                        #        print("Rejected solution.")
                        #    for mngr_idx in range(N_mngr):
                        #        smarts_typing = smarts_typing_list[mngr_idx]
                        #        if smarts_typing:
                        #            print(
                        #                pvec_list[mngr_idx].vector_k,
                        #                pvec_list[mngr_idx].allocations,
                        #                state.get_smarts(),
                        #                )


            else:
                AIC_old = _calculate_AIC(
                    pvec_list,
                    0,
                    [pvec_list[0].allocations[:].tolist()]
                    )[0]

                accept = False
                if new_AIC < best_AIC:
                    accept = True

                if accept:
                    best_AIC       = new_AIC
                    best_pvec_list = [pvec.copy() for pvec in pvec_list]
                    best_ast       = ast

    if best_AIC < old_AIC:
        for mngr_idx in range(N_mngr):
            typing_constraints = typing_constraints_list[mngr_idx]
            if not isinstance(typing_constraints, TypingConstraints):
                result = True
            else:
                result = typing_constraints.is_pvec_valid_hierarchy(
                    best_pvec_list[mngr_idx],
                    verbose
                    )
                if not result:
                    diff = best_AIC - old_AIC
                    typing_temperature = typing_temperature_list[mngr_idx]
                    p = np.exp(-diff/typing_temperature)
                    u = np.random.random()
                    if u < p:
                        result = True

            if result:
                found_improvement = True
            else:
                if verbose:
                    print(
                        "Typing violation."
                        )
                found_improvement = False
                break

    else:
        found_improvement = False

    if found_improvement:
        for mngr_idx in range(N_mngr):
            pvec_list[mngr_idx].reset(
                best_pvec_list[mngr_idx]
                )
            smarts_typing = smarts_typing_list[mngr_idx]
            if smarts_typing:
                pvec_list[mngr_idx].apply_changes(True,True)
            else:
                pvec_list[mngr_idx].apply_changes()
    else:
        for mngr_idx in range(N_mngr):
            pvec_list[mngr_idx].reset(
                old_pvec_list[mngr_idx]
                )
            smarts_typing = smarts_typing_list[mngr_idx]
            if smarts_typing:
                pvec_list[mngr_idx].apply_changes(True,True)
            else:
                pvec_list[mngr_idx].apply_changes()

    if verbose:
        if best_ast != None:
            print("best_AIC:", best_AIC)
            if best_AIC < old_AIC:
                print("Solution accepted.")
            else:
                print("Solution not accepted.")
            for pvec in best_pvec_list:
                if isinstance(pvec, SmartsForceFieldParameterVector):
                    print(
                        pvec.vector_k,
                        pvec.allocations,
                        pvec.get_smarts()
                    )
                else:
                    print(
                        pvec.vector_k,
                        pvec.allocations
                    )
        else:
            print(
                f"No move attempted. Best optimized: {best_AIC}"
                )

    return found_improvement, pvec_list, best_AIC


class ForceFieldHopper(object):

    def __init__(
        self, 
        system_list, 
        parm_penalty_split = 1.,
        parm_penalty_merge = 1.,
        name="ForceFieldHopper",
        verbose=False):

        from .targets import TargetComputer

        self.parm_penalty_split = parm_penalty_split
        self.parm_penalty_merge = parm_penalty_merge

        self.bounds_list = list()
        self.allocation_scorer_list = list()
        self.switching = list()

        self.parameter_manager_list = list()
        self.smarts_typing_list = list()
        self.smarts_manager_list = list()
        self.exclude_others = list()
        self.parameter_name_list = list()
        self.scaling_factor_list = list()
        self.allocation_prior_constraints_list = list()
        self.typing_constraints_list = list()
        self.typing_temperature_list = list()
        self.best_pvec_list = list()

        self.N_parms_traj = list()
        self.pvec_traj    = list()
        self.like_traj    = list()
        self.aic_traj     = list()
        
        self._N_steps   = 0
        self._N_mngr    = 0

        self.system_list     = system_list
        self._N_systems      = len(system_list)

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

        self._allocation_score_cacher = dict()

        self.merge_cutoff = list()
        
        self.pvec_cache_dict = dict()

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
            smarts_typing = self.smarts_typing_list[mngr_idx]
            if smarts_typing:
                N_parms += self.best_pvec_list[mngr_idx].size
            else:
                pvec = self.generate_parameter_vectors(
                    [mngr_idx],
                    _system_idx_list,
                    as_copy=True
                    )[0]
                N_parms += pvec.size

        return N_parms

    
    def generate_parameter_vectors(
        self, 
        mngr_idx_list = list(),
        system_idx_list = list(),
        as_copy=False
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

        pvec_list = list()
        for mngr_idx in _mngr_idx_list:
            key = mngr_idx, tuple(_system_idx_list)
            if key in self.pvec_cache_dict:
                pvec = self.pvec_cache_dict[key]
            else:
                parm_mngr = copy.deepcopy(
                    self.parameter_manager_list[mngr_idx]
                    )
                for sys_idx in _system_idx_list:
                    parm_mngr.add_system(
                        self.system_list[sys_idx]
                        )
                smarts_typing = self.smarts_typing_list[mngr_idx]
                if smarts_typing:
                    pvec = SmartsForceFieldParameterVector(
                        parm_mngr,
                        [self.smarts_manager_list[mngr_idx][0]],
                        self.parameter_name_list[mngr_idx],
                        self.scaling_factor_list[mngr_idx],
                        exclude_others = self.exclude_others[mngr_idx]
                    )
                    if self.best_pvec_list[mngr_idx] != None:
                        pvec.reset(
                            self.best_pvec_list[mngr_idx]
                            )
                        pvec.apply_changes(True, True)

                else:
                    pvec = ForceFieldParameterVector(
                        parm_mngr,
                        self.parameter_name_list[mngr_idx],
                        self.scaling_factor_list[mngr_idx],
                        exclude_others = self.exclude_others[mngr_idx]
                        )
                self.pvec_cache_dict[key] = pvec
            if as_copy:
                pvec_list.append(
                    self.pvec_cache_dict[key].copy()
                    )
            else:
                pvec_list.append(
                    self.pvec_cache_dict[key]
                    )

        return pvec_list


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

    
    def reset_parameters(self):

        self.pvec_cache_dict = dict()
        pvec_list = self.generate_parameter_vectors()
        for mngr_idx in range(self.N_mngr):
            pvec = pvec_list[mngr_idx]
            pvec.reset(
                self.best_pvec_list[mngr_idx]
                )
            smarts_typing = self.smarts_typing_list[mngr_idx]
            if smarts_typing:
                pvec.apply_changes(True, True)
            else:
                pvec.apply_changes()

    
    def update_best(
        self,         
        pvec_list = None,
        mngr_idx_list = list(),
        ):

        import copy

        self.pvec_cache_dict = dict()
        if pvec_list == None:
            mngr_idx_list = list()
            for mngr_idx in range(self.N_mngr):
                smarts_typing = self.smarts_typing_list[mngr_idx]
                if not smarts_typing:
                    mngr_idx_list.append(mngr_idx)

            pvec_list = self.generate_parameter_vectors(
                mngr_idx_list,
                as_copy=True
                )

        else:
            assert len(mngr_idx_list) > 0
            assert len(mngr_idx_list) == len(pvec_list)

        for idx, mngr_idx in enumerate(mngr_idx_list):
            self.best_pvec_list[mngr_idx] = pvec_list[idx].copy()
            smarts_typing = self.smarts_typing_list[mngr_idx]
            if smarts_typing:
                self.smarts_manager_list[mngr_idx] = copy.deepcopy(
                    pvec_list[idx].smarts_manager_list
                    )


    def add_parameters(
        self,
        parameter_manager,
        smarts_manager_list = None,
        smarts_typing = False,
        parameter_name_list = None,
        exclude_others = False,
        scale_list = None,
        allocation_scorer = None,
        allocation_prior_constraints = None,
        typing_constraints = None,
        typing_temperature = 1.,
        switching = False,
        ### Any parameters that are closer then `merge_cutoff`
        ### to each other will be brute force merged.
        merge_cutoff= 0.2,
        bounds = None
        ):

        if parameter_manager.N_systems != 0:
            raise ValueError(
                f"parameter_manager must be empty, but found {parameter_manager.N_systems} systems."
                )

        self.parameter_manager_list.append(parameter_manager)
        self.smarts_typing_list.append(smarts_typing)
        self.smarts_manager_list.append(smarts_manager_list)
        self.parameter_name_list.append(parameter_name_list)
        self.exclude_others.append(exclude_others)
        self.scaling_factor_list.append(scale_list)
        self.allocation_scorer_list.append(allocation_scorer)
        self.allocation_prior_constraints_list.append(allocation_prior_constraints)
        self.typing_constraints_list.append(typing_constraints)
        self.typing_temperature_list.append(typing_temperature)

        self.bounds_list.append(bounds)

        self.switching.append(switching)

        self.merge_cutoff.append(merge_cutoff)

        ### Must increment at the end, not before.
        self._N_mngr += 1

        self.best_pvec_list.append(None)
        pvec = self.generate_parameter_vectors(
            mngr_idx_list=[self.N_mngr-1]
            )[0]
        self.best_pvec_list[-1] = pvec.copy()

    
    def move(
        self,
        ff_parameter_vector,
        alloc_scorer,
        ):

        N_comb_max = 10

        N_types = ff_parameter_vector.force_group_count
        type_j  = N_types

        ff_parameter_vector_id = ray.put(ff_parameter_vector)
        worker_id_list = list()
        for type_i in range(N_types):
            for type_j in range(type_i+1, N_types):
                selection_i = ff_parameter_vector.allocations.index([type_i])[0]
                selection_j = ff_parameter_vector.allocations.index([type_j])[0]
                selection_ij = np.append(selection_i, selection_j)
                N_allocs = selection_ij.size
                if N_allocs < 2:
                    continue
                ### Generate all possible splitting solutions
                ### The `[1:-1]` removes the two solutions that
                ### would lead to empty types.
                k_values_ij = np.array(
                    np.meshgrid(
                        *[[type_i, type_j] for _ in range(N_allocs)]
                        )
                    ).T.reshape(-1,N_allocs)[1:-1]

                ### We only need first part of array.
                ### Rest is mirror typing
                k_values_ij,_ = np.array_split(k_values_ij, 2)
                N_comb = k_values_ij.shape[0]
                scores = np.zeros(N_comb, dtype=float)
                if not isinstance(alloc_scorer, type(None)):
                    for comb_idx in range(N_comb):
                        score_i = alloc_scorer(
                            parameter_manager = ff_parameter_vector.parameter_manager, 
                            type_i = type_j,
                            type_j = type_i, # Constraints only in this group
                            selection_ij = selection_ij, 
                            k_values_ij = k_values_ij[comb_idx]
                            )
                        score_j = alloc_scorer(
                            parameter_manager = ff_parameter_vector.parameter_manager, 
                            type_i = type_i,
                            type_j = type_j, # Constraints only in this group
                            selection_ij = selection_ij, 
                            k_values_ij = k_values_ij[comb_idx]
                            )
                        if score_i > 0. and score_j > 0.:
                            scores[comb_idx] = score_i + score_j

                    ### Only keep highest scores (currently 50)
                    if N_comb > N_comb_max:
                        score_sort = np.argsort(scores)[::-1][:N_comb_max]
                    else:
                        score_sort = np.argsort(scores)[::-1]

                    _k_values_ij = list()
                    for score_idx in score_sort:
                        if scores[score_idx] > 0.:
                            _k_values_ij.append(
                                k_values_ij[score_idx]
                                )
                    k_values_ij  = np.array(_k_values_ij, dtype=int)

                worker_id = get_gradient_scores.remote(
                    ff_parameter_vector = ff_parameter_vector_id,
                    targetcomputer = self.targetcomputer_id,
                    type_i = type_i,
                    type_j = type_j,
                    selection_i = selection_ij,
                    k_values_ij = k_values_ij,
                    split = True,
                    grad_diff = self.grad_diff,
                    )
                worker_id_list.append(worker_id)

        return worker_id_list


    def split_smarts(
        self,
        ff_parameter_vector,
        split_all=False,
        N_max_positions=500
        ):
    
        import ray
        import itertools
        import numpy as np
        import copy

        N_types = ff_parameter_vector.force_group_count
        if split_all:
            type_query_list = range(N_types)
        else:
            type_query_list = [N_types-1]

        worker_id_dict = dict()
        for type_i in type_query_list:

            N_types = ff_parameter_vector.force_group_count

            ff_parameter_vector.apply_changes(
                flush_to_systems=False,
                flush_to_allocations=True
                )
            selection_allocation_ij = ff_parameter_vector.allocations.index([type_i])[0]
            size_ij = selection_allocation_ij.size

            type_j = type_i+1

            ### Generate new type and put behind type
            ### it was derived from
            ff_parameter_vector.duplicate(type_i)
            if type_j != N_types:
                ff_parameter_vector.swap_smarts(N_types, type_j)

            selection_i = ff_parameter_vector.get_smarts_indices(type_i)
            selection_j = ff_parameter_vector.get_smarts_indices(type_j)

            ### Get number of entries in the smarts allocations vector
            ### that are active
            N_actives = np.where(
                ff_parameter_vector.smarts_allocations[selection_i] != _INACTIVE_GROUP_IDX
                )[0].size

            positions_set = set()
            positions_set.update(
                set(itertools.permutations(selection_j, N_actives))
            )
            positions_set.update(
                set(itertools.permutations(selection_j, N_actives+1))
            )

            N_positions = len(positions_set)
            if N_positions < N_max_positions:
                N_max_positions = N_positions
            sele_positions = np.arange(N_positions, dtype=int)
            np.random.shuffle(sele_positions)
            positions_set = tuple(positions_set)
            if self.verbose:
                print(
                    f"Querying {len(positions_set)} SMARTS positions for splitting of type {type_i}"
                    )

            ff_parameter_vector_id = ray.put(ff_parameter_vector.copy())
            key = (type_i, type_j, tuple(selection_allocation_ij.tolist()))
            worker_id_dict[key] = list()
            for positions_idx in sele_positions[:N_max_positions]:
                positions = positions_set[positions_idx]
                N_actives = len(positions)
                if N_actives == 0:
                    continue
                worker_id = generate_split_candidates.remote(
                    ff_parameter_vector_id,
                    type_i,
                    type_j,
                    positions,
                    selection_allocation_ij
                    )
                worker_id_dict[key].append(worker_id)

            ff_parameter_vector.set_inactive(type_j)
            if type_j != N_types:
                ff_parameter_vector.swap_smarts(type_j, N_types)
            ff_parameter_vector.remove(N_types)
            ### The apply changes call is necessary here,
            ### so that the allocations vector gets reset properly.
            ff_parameter_vector.apply_changes(
                flush_to_systems=False,
                flush_to_allocations=True
            )

        state_all_dict = dict()
        for key in worker_id_dict:
            type_i, type_j, selection_allocation_ij = key
            for worker_id in worker_id_dict[key]:
                state_dict = ray.get(worker_id)
                if not (key in state_all_dict):
                    state_all_dict[key] = state_dict
                else:
                    for alloc in state_dict:
                        if alloc in state_all_dict[key]:
                            ### `state_all_dict[key][alloc]` is type dict
                            ### `state_dict[alloc]` too
                            state_all_dict[key][alloc].update(state_dict[alloc])
                        else:
                            state_all_dict[key][alloc] = state_dict[alloc]

        ff_parameter_vector_id = ray.put(ff_parameter_vector)
        worker_id_list = list()
        state_dict_return = dict()
        for key in state_all_dict:
            type_i, type_j, selection_allocation_ij = key
            k_values_ij = list(state_all_dict[key].keys())
            k_values_ij = np.array(k_values_ij)
            selection_allocation_ij = np.array(selection_allocation_ij)
            sele = np.where(k_values_ij == type_j)
            k_values_ij[sele] = ff_parameter_vector.force_group_count
            N_comb = k_values_ij.shape[0]
            N_array_splits = 2
            if N_comb > 10:
                N_array_splits = int(N_comb/10)
            for split_idxs in np.array_split(np.arange(N_comb), N_array_splits): 
                worker_id = get_gradient_scores.remote(
                    ff_parameter_vector = ff_parameter_vector_id,
                    targetcomputer = self.targetcomputer_id,
                    type_i = type_i,
                    type_j = None,
                    selection_i = selection_allocation_ij,
                    k_values_ij = k_values_ij[split_idxs],
                    split = True,
                    grad_diff = self.grad_diff,
                    )
                worker_id_list.append(worker_id)

            state_dict_return[type_i] = state_all_dict[key]

        return worker_id_list, state_dict_return


    def split(
        self,
        ff_parameter_vector,
        alloc_scorer,
        max_N_queries=300,
        max_group_size=2,
        ):

        __doc__ = """
        `max_N_queries` is the maxium number of attempts to switch types. This number
        will be used to limit the total number of queries.
        The `max_group_size` is the maximum number of groups that are attempt to split/switch-on.

        Examples:
            N_g: Number of switchable groups
            N_a: Maximum number of groups beeing switched at a time (see `max_group_size`)
            N_k: Total number possible realizations. This is the number we care about.

            N_g=10, N_a=10
                --> N_k=1022

            N_g=20, N_a=10
                --> N_k=431909

            N_g=30, N_a=10
                --> N_k=22964086

            It can be verified that N_k = \\sum_i^{N_a-1} \\binom N_g i
        """

        from scipy import special
        import itertools

        N_comb_max = 10

        N_types = ff_parameter_vector.force_group_count
        type_j  = N_types

        ff_parameter_vector_id = ray.put(ff_parameter_vector)
        worker_id_list = list()
        for type_i in range(N_types):
            selection_i = ff_parameter_vector.allocations.index([type_i])[0]
            N_allocs = selection_i.size
            if N_allocs < 2:
                continue

            relocation_count = N_allocs
            effective_max_group_size = 1
            ### We start at `2`, since `special.comb(N_allocs,1)` equals `N_allocs`
            for i in range(2, max_group_size):
                count = int(special.comb(N_allocs, i))
                if (relocation_count+count) > max_N_queries:
                    break
                else:
                    effective_max_group_size = i
                    relocation_count += count

            k_values_ij = np.ones(
                (relocation_count, selection_i.size), 
                dtype=int
                )
            c = 0
            selection_i_selection = np.arange(
                selection_i.size,
                dtype=int
            )
            for i in range(1, effective_max_group_size+1):
                k_values_N = list(
                    set(
                        itertools.combinations(
                            selection_i_selection, 
                            r=i
                        )
                    )
                )
                for k in k_values_N:
                    k_values_ij[c][:] = type_i
                    k_values_ij[c][list(k)] = type_j
                    c += 1

            ### We only need first part of array.
            ### Rest is mirror typing
            N_comb = k_values_ij.shape[0]
            scores = np.zeros(N_comb, dtype=float)
            if not isinstance(alloc_scorer, type(None)):
                for comb_idx in range(N_comb):
                    scores[comb_idx] = alloc_scorer(
                        parameter_manager = ff_parameter_vector.parameter_manager, 
                        type_i = type_i,
                        type_j = type_j,
                        selection_ij = selection_i,
                        k_values_ij = k_values_ij[comb_idx]
                        )
                ### Only keep highest scores (currently 50)
                if N_comb > N_comb_max:
                    score_sort = np.argsort(scores)[::-1][:N_comb_max]
                else:
                    score_sort = np.argsort(scores)[::-1]

                _k_values_ij = list()
                for score_idx in score_sort:
                    if scores[score_idx] > 0.:
                        _k_values_ij.append(
                            k_values_ij[score_idx]
                            )
                k_values_ij  = np.array(_k_values_ij, dtype=int)

            N_comb = k_values_ij.shape[0]
            N_array_splits = 2
            if N_comb > 10:
                N_array_splits = int(N_comb/10)
            for split_k_value_ij in np.array_split(k_values_ij, N_array_splits):
                worker_id = get_gradient_scores.remote(
                    ff_parameter_vector = ff_parameter_vector_id,
                    targetcomputer = self.targetcomputer_id,
                    type_i = type_i,
                    selection_i = selection_i,
                    k_values_ij = split_k_value_ij,
                    split = True,
                    grad_diff = self.grad_diff,
                    )
                worker_id_list.append(worker_id)

        return worker_id_list

    
    def merge(
        self,
        ff_parameter_vector,
        alloc_scorer,
        ):

        N_comb_max = 10

        N_types = ff_parameter_vector.force_group_count

        ff_parameter_vector_id = ray.put(ff_parameter_vector)
        worker_id_list = list()
        for type_i in range(N_types):
            selection_i = ff_parameter_vector.allocations.index([type_i])[0]
            score_list  = np.zeros(N_types, dtype=float)
            type_j_list = np.zeros(N_types, dtype=int)
            if not isinstance(alloc_scorer, type(None)):
                for type_j in range(N_types):
                    if type_i == type_j:
                        continue
                    selection_j   = ff_parameter_vector.allocations.index([type_j])[0]
                    selection_ij  = np.append(selection_i, selection_j)
                    type_j_list[type_j] = type_j
                    score_list[type_j] = alloc_scorer(
                        parameter_manager = ff_parameter_vector.parameter_manager, 
                        type_i = None,
                        type_j = type_i, # Looks weird. Is correct!
                        selection_ij = selection_ij, 
                        k_values_ij = np.zeros_like(selection_ij, dtype=int) + type_i
                        )

                ### `alloc_scorer` reports dissimilarity. Therefore
                ### only keep the lowest scores. We want high within-group similarity here!
                ### Only keep highest scores (currently 50)
                if N_types > N_comb_max:
                    score_sort = np.argsort(score_list)[:N_comb_max]
                else:
                    score_sort = np.argsort(score_list)

                _type_j_list = list()
                for score_idx in score_sort:
                    if score_list[score_idx] > 0.:
                        _type_j_list.append(
                            type_j_list[score_idx]
                            )
                type_j_list  = _type_j_list
            if len(type_j_list) > 0:
                worker_id = get_gradient_scores.remote(
                    ff_parameter_vector = ff_parameter_vector_id,
                    targetcomputer = self.targetcomputer_id,
                    type_i = type_i,
                    type_j = type_j_list,
                    split = False,
                    grad_diff = self.grad_diff,
                    )
                worker_id_list.append(worker_id)

        return worker_id_list

    
    def switch_on(
        self,
        ff_parameter_vector,
        alloc_scorer,
        max_N_queries=50,
        max_group_size=2,
        ):

        __doc__ = """
        `max_N_queries` is the maxium number of attempts to switch types. This number
        will be used to limit the total number of queries.
        The `max_group_size` is the maximum number of groups that are attempt to split/switch-on
        per attempt.

        Examples:
            N_g: Number of switchable groups
            N_a: Maximum number of groups beeing switched at a time (see `max_group_size`)
            N_k: Total number possible realizations. This is the number we care about.

            N_g=10, N_a=10
                --> N_k=1022

            N_g=20, N_a=10
                --> N_k=431909

            N_g=30, N_a=10
                --> N_k=22964086

            It can be verified that N_k = \\sum_i^{N_a-1} \\binom N_g i

        """

        from scipy import special
        import numpy as np
        import itertools

        N_comb_max = 10

        N_types = ff_parameter_vector.force_group_count
        ### This is the new parameter type
        type_i = N_types

        ff_parameter_vector_id = ray.put(ff_parameter_vector)
        worker_id_list = list()

        selection_i = np.where(ff_parameter_vector.allocations[:] == _INACTIVE_GROUP_IDX)[0]
        N_allocs    = selection_i.size
        if N_allocs == 0:
            return list()

        relocation_count = N_allocs
        effective_max_group_size = 1
        ### We start at `2`, since `special.comb(N_allocs,1)` equals `N_allocs`
        for i in range(2, max_group_size):
            count = int(special.comb(N_allocs, i))
            if (relocation_count+count) > max_N_queries:
                break
            else:
                effective_max_group_size = i
                relocation_count += count

        k_values_ij = np.ones(
            (relocation_count, selection_i.size), 
            dtype=int
            )
        c = 0
        selection_i_selection = np.arange(
            selection_i.size,
            dtype=int
        )
        for i in range(1, effective_max_group_size+1):
            k_values_N = list(
                set(
                    itertools.combinations(
                        selection_i_selection, 
                        r=i
                    )
                )
            )
            for k in k_values_N:
                k_values_ij[c][:] = _INACTIVE_GROUP_IDX
                k_values_ij[c][list(k)] = type_i
                c += 1

        N_comb = k_values_ij.shape[0]
        scores = np.zeros(N_comb, dtype=float)
        if not isinstance(alloc_scorer, type(None)):
            for comb_idx in range(N_comb):
                scores[comb_idx] = alloc_scorer(
                    parameter_manager = ff_parameter_vector.parameter_manager, 
                    type_i = None,
                    type_j = type_i, # This is right.
                    selection_ij = selection_i, 
                    k_values_ij = k_values_ij[comb_idx]
                    )

            ### Only keep highest scores (currently 50)
            if N_comb > N_comb_max:
                score_sort = np.argsort(scores)[::-1][:N_comb_max]
            else:
                score_sort = np.argsort(scores)[::-1]

            _k_values_ij = list()
            for score_idx in score_sort:
                if scores[score_idx] > 0.:
                    _k_values_ij.append(
                        k_values_ij[score_idx]
                        )
            k_values_ij  = np.array(_k_values_ij, dtype=int)

        N_comb = k_values_ij.shape[0]
        N_array_splits = 2
        if N_comb > 10:
            N_array_splits = int(N_comb/10)

        worker_id_list = list()
        for split_k_value_ij in np.array_split(k_values_ij, N_array_splits):
            switch_values = [0. * _FORCE_CONSTANT_TORSION for _ in range(ff_parameter_vector.parameters_per_force_group)]
            worker_id = get_switchon_scores.remote(
                ff_parameter_vector = ff_parameter_vector_id,
                targetcomputer = self.targetcomputer_id,
                switch_values = switch_values,
                selection_i = selection_i,
                k_values_ij = split_k_value_ij,
                grad_diff = self.grad_diff,
                )

            worker_id_list.append(worker_id)

        return worker_id_list


    def reduce_parameters(
        self, 
        switch_min_val, 
        merge_cutoff_scale,
        use_pre_smirksify,
        mngr_idx_list=list()
        ):

        if len(mngr_idx_list) == 0:
            _mngr_idx_list = list(range(self.N_mngr))
        else:
            _mngr_idx_list = mngr_idx_list

        for mngr_idx in _mngr_idx_list:

            smarts_typing = self.smarts_typing_list[mngr_idx]
            if smarts_typing:
                ### Figure out which types can be brute force removed.
                ### Note this only works with `SmartsForceFieldParameterVector`
                ### objects.
                pvec = self.generate_parameter_vectors(
                    mngr_idx_list=[mngr_idx]
                    )[0]
                ### First remove empty types
                to_delete = list()
                N_types = pvec.force_group_count
                for i in range(N_types):
                    N_allocations = pvec.allocations.index([i])[0].size
                    if N_allocations == 0:
                        to_delete.append(i)
                to_delete = sorted(to_delete, reverse=True)
                for idx in to_delete:
                    pvec.remove(idx)
                pvec.apply_changes(True, True)
                self.update_best(
                    [pvec],
                    [mngr_idx]
                    )

                if self.verbose:
                    if len(to_delete) > 0:
                        print(
                            f"Removed empty types {to_delete} from mngr {mngr_idx}."
                            )
                    else:
                        print(
                            f"Did not remove any empty types from mngr {mngr_idx}."
                            )

                state_old = pvec.copy()
                N_types = pvec.force_group_count
                AIC_old = self.calculate_AIC()
                best_type_removal = None
                for i in range(N_types):
                    pvec.remove(i)
                    pvec.apply_changes(True, True)
                    self.update_best(
                        [pvec],
                        [mngr_idx]
                        )
                    AIC_new = self.calculate_AIC()
                    if AIC_new < AIC_old:
                        best_type_removal = i
                        AIC_old = AIC_new

                    pvec.reset(state_old)
                    pvec.apply_changes(True, True)
                    self.update_best(
                        [pvec],
                        [mngr_idx]
                        )

                if best_type_removal != None:
                    pvec.remove(best_type_removal)
                    pvec.apply_changes(True, True)
                    self.update_best(
                        [pvec],
                        [mngr_idx]
                        )

                if self.verbose:
                    if best_type_removal != None:
                        print(
                            f"Removed bad performing type {best_type_removal} from mngr {mngr_idx}"
                            )
                    else:
                        print(
                            f"Did not remove any bad perfomring type from mngr {mngr_idx}"
                            )


                continue                    

            if use_pre_smirksify:
                ### Brute force merge everything that
                ### won't be processed by chemper.SMIRKSifier
                pvec = self.generate_parameter_vectors([mngr_idx])[0]
                force_group_idx_list = np.array(pvec.parameter_manager.force_group_idx_list)
                sys_idx_list  = np.array(pvec.parameter_manager.system_idx_list)
                all_atom_list = np.array(pvec.parameter_manager.atom_list)
                atom_index_list = list()
                for p_type in range(pvec.force_group_count):
                    atom_list = list()
                    for sys_idx in range(pvec.parameter_manager.N_systems):
                        valids = np.where((force_group_idx_list == p_type) * (sys_idx_list == sys_idx))[0]
                        atom_list.append([tuple(a.tolist()) for a in all_atom_list[valids]])
                    atom_index_list.append((f'c{p_type}',atom_list))

                add_list, _ = pre_smirksify(
                    molecules=[sys.rdmol for sys in self.system_list],
                    cluster_definitions=atom_index_list,
                    max_layers=3,
                    )
                types_to_be_deleted = list()
                force_ranks         = list()
                for type_list in add_list:
                    if len(type_list) > 1:
                        value_list = list()
                        N_types = len(type_list)
                        for idx in range(N_types):
                            value = pvec.get_parameters_by_force_group(
                                type_list[idx], 
                                get_all_parms=False
                                )
                            value_list.append(
                                value
                                )
                            if idx > 0:
                                ### Set everything to type in `type_list[0]`
                                valids = pvec.allocations.index([type_list[idx]])[0]
                                pvec.allocations[valids] = type_list[0]
                                for v in valids:
                                    if not v in force_ranks:
                                        force_ranks.append(v)
                        pvec.set_parameters_by_force_group(
                            force_group_idx = type_list[0],
                            values = np.zeros(pvec.parameters_per_force_group),
                            values_0 = np.mean(value_list, axis=0),
                            )
                        types_to_be_deleted.extend(
                            type_list[1:]
                            )
                pvec.apply_changes()
                for type_i in sorted(types_to_be_deleted, reverse=True):
                    pvec.remove(type_i)
                    pvec.apply_changes()
                force_group_idxs = [[] for _ in range(self.N_mngr)]
                for r in force_ranks:
                    force_group = pvec.allocations[r]
                    ### Figure out which force_groups must be optimized again
                    if not force_group in force_group_idxs[mngr_idx]:
                        force_group_idxs[mngr_idx].append(
                            force_group
                            )

                self.minimize_mngr(
                    [pvec],
                    self.system_list,
                    bounds_list=[self.bounds_list[mngr_idx]],
                    force_group_idxs=force_group_idxs,
                    parallel_targets=True
                    )

            pvec = self.generate_parameter_vectors([mngr_idx])[0]
            ### See if we can merge some parameters
            if pvec.force_group_count > 1:
                ### `merge_cutoff` is the effetive radius
                ### used for clustering
                merge_cutoff = merge_cutoff_scale * self.merge_cutoff[mngr_idx]
                pvec.cluster_merge(merge_cutoff)
            pvec.apply_changes()
            if self.switching[mngr_idx]:
                ### See if we can switch off some parameters
                for type_i in range(pvec.force_group_count)[::-1]:
                    value_list = pvec.get_parameters_by_force_group(
                        type_i, 
                        get_all_parms=False
                        )
                    isoff = True
                    for v in value_list:
                        v = v.value_in_unit(_FORCE_CONSTANT_TORSION)
                        isoff *= v < switch_min_val
                    if isoff:
                        valids = pvec.allocations.index([type_i])[0]
                        pvec.allocations[valids] = _INACTIVE_GROUP_IDX
                        pvec.apply_changes()
                        pvec.remove(type_i)
                        pvec.apply_changes()


    def save_traj(
        self, 
        parm_penalty=1.
        ):

        self.like_traj.append(self.calc_log_likelihood())
        self.aic_traj.append(self.calculate_AIC(parm_penalty=parm_penalty))

        pvec_list_cp = self.generate_parameter_vectors(as_copy=True)
        N_parms_all  = self.get_number_of_parameters()
        self.pvec_traj.append(pvec_list_cp)
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
            tuple(sys_idx_pair) for sys_idx_pair in system_idx_list_batch
            )

        return system_idx_list_batch


    def get_grad_scores(
        self,
        pvec,
        max_group_size=2,
        N_trials_gradient=5,
        alloc_scorer=None,
        split=True,
        move=True,
        switch_on=True,
        merge=True,
        smarts_typing=False,
        ):

        split_worker_id_dict = dict()
        switchon_worker_id_dict = dict()
        move_worker_id_dict = dict()
        merge_worker_id_dict = dict()

        pvec_start = copy.deepcopy(pvec[:])
        for trial_idx in range(N_trials_gradient):
            pvec[:] += np.random.normal(0, self.perturbation/2.)
            pvec.apply_changes()

            if smarts_typing:
                if split:
                    split_worker_id_dict[trial_idx] = self.split_smarts(
                        pvec,
                        split_all=True
                        )
            else:
                if split:
                    split_worker_id_dict[trial_idx] = self.split(
                        pvec,
                        alloc_scorer,
                        max_group_size=max_group_size,
                        )
                if move:
                    move_worker_id_dict[trial_idx] = self.move(
                        pvec,
                        alloc_scorer,
                        )
                if switch_on:
                    switchon_worker_id_dict[trial_idx] = self.switch_on(
                        pvec,
                        alloc_scorer,
                        max_group_size=max_group_size,
                        )
                if merge:
                    merge_worker_id_dict[trial_idx] = self.merge(
                        pvec,
                        alloc_scorer,
                        )

            ### Reset parameter vector
            pvec[:] = copy.deepcopy(pvec_start[:])
            pvec.apply_changes()

        return split_worker_id_dict, move_worker_id_dict, switchon_worker_id_dict, merge_worker_id_dict


    def get_votes(
        self,
        worker_id_dict = dict(),
        low_to_high=True,
        abs_grad_score=False,
        norm_cutoff=0.,
        keep_N_best=10,
        smarts_typing=False,
        ):

        import numpy as np

        if len(worker_id_dict) == 0:
            return list()

        votes_dict = dict()
        state_dict = dict()
        for trial_idx in worker_id_dict:
            score_dict = dict()
            _state_dict = dict()

            if smarts_typing:
                worker_id_list, state_all_dict = worker_id_dict[trial_idx]
            else:
                worker_id_list = worker_id_dict[trial_idx]
            for worker_id in worker_id_list:

                grad_score, \
                grad_norm, \
                allocation_list, \
                selection_list, \
                type_list = ray.get(worker_id)

                for g, n, a, s, t in zip(grad_score, grad_norm, allocation_list, selection_list, type_list):
                    ### Only use this split, when the norm
                    ### is larger than norm_cutoff
                    #print(
                    #    f"Gradient scores {g}",
                    #    f"Gradient norms {n}",
                    #    )
                    if np.all(np.abs(n) < norm_cutoff):
                        pass
                    else:
                        if smarts_typing:
                            _a = np.array(a)
                            sele = np.where(_a == t[1])
                            _a[sele] = t[0] + 1
                            a = tuple(_a.tolist())

                            _t = list(t)
                            _t[1] = _t[0] + 1
                            t = tuple(_t)
                        ### Only consider the absolute value of the gradient score, not the sign
                        if abs_grad_score:
                            _g = [abs(gg) for gg in g]
                            g  = _g
                        if (a,s,t) in score_dict:
                            print(
                                "ast alert", (a,s,t)
                                )
                        score_dict[(a,s,t)] = g
                        if smarts_typing:
                            _state_dict[(a,s,t)] = state_all_dict[t[0]][a]

                #from .debugging_tools import total_size
                #print(
                #    "State size:", total_size(_state_dict),
                #    "Score size:", total_size(score_dict),
                #    )

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
                    state_dict[ast] = _state_dict[ast]

        ### Only keep the final best 10 ast
        votes_list = [ast for ast, _ in sorted(votes_dict.items(), key=lambda items: items[1], reverse=True)]
        if len(votes_list) > keep_N_best:
            votes_list = votes_list[:keep_N_best]

        if smarts_typing:
            state_list = [state_dict[ast] for ast in votes_list]
            return votes_list, state_list
        else:        
            return votes_list


    def minimize_mngr(
        self,
        pvec_all,
        system_list,
        bounds_list=None,
        force_group_idxs=None,
        parallel_targets=True,
        bounds_penalty=1000.,
        ):

        if isinstance(bounds_list, type(None)):
            bounds_list = self.bounds_list

        worker_id = minimize_FF.remote(
                system_list = system_list,
                targetcomputer = self.targetcomputer_id,
                pvec_list = pvec_all,
                bounds_list = bounds_list,
                parm_penalty = self.parm_penalty_merge,
                pvec_idx_min = None,
                force_group_idxs = force_group_idxs,
                parallel_targets = parallel_targets,
                bounds_penalty=bounds_penalty,
                verbose = self.verbose,
                )

        _, pvec_list_new = ray.get(worker_id)

        N_pvec = len(pvec_all)
        for i in range(N_pvec):
            pvec_all[i].reset(
                pvec_list_new[i],
                pvec_list_new[i].allocations
                )
            if isinstance(pvec, SmartsForceFieldParameterVector):
                pvec_all[i].apply_changes(True, True)
            else:
                pvec_all[i].apply_changes()


    def get_min_scores(
        self,
        mngr_idx,
        system_idx_list = list(),
        votes_split_list = list(),
        votes_move_list = list(),
        votes_switch_on_list = list(),
        votes_merge_list = list(),
        parallel_targets = False,
        bounds_penalty=1000.,
        smarts_typing=False,
        votes_state_list=None,
        ):

        use_scipy = True

        if len(system_idx_list) == 0:
            system_idx_list = list(range(self.N_systems))

        system_list_id = ray.put([self.system_list[sys_idx] for sys_idx in system_idx_list])
        worker_id_dict = dict()

        ### Query split candidates
        ### ======================
        for counts, ast in enumerate(votes_split_list):
            allocation, selection, type_ = ast
            pvec_all = self.generate_parameter_vectors(
                [],
                system_idx_list,
                )
            pvec = pvec_all[mngr_idx]

            if smarts_typing:
                votes_state_list[counts] = tuple(votes_state_list[counts])

                key = votes_state_list[counts][0]
                type_j, allocation_state, allocations = key
                pvec.duplicate(type_j-1)

                smarts_manager = pvec.smarts_manager_list[type_j]
                for k,v in allocation_state:
                    smarts_manager.allocation_state[k] =v
                smarts_manager.allocations[:] = allocations[:]
                
                valids = pvec.smarts_manager_allocations.index([type_j])[0]
                pvec.smarts_allocations[valids] = allocations[:]
                
                pvec.apply_changes(True, True)

                mngr_idx_list = list()
                for _mngr_idx in range(self.N_mngr):
                    smarts_typing = self.smarts_typing_list[_mngr_idx]
                    if smarts_typing:
                        mngr_idx_list.append(_mngr_idx)
            else:
                pvec.duplicate(type_[0])
                pvec.allocations[list(selection)] = list(allocation)
                pvec.apply_changes()
                mngr_idx_list = [mngr_idx]

            worker_id = minimize_FF.remote(
                    system_list = system_list_id,
                    targetcomputer = self.targetcomputer_id,
                    pvec_list = pvec_all,
                    bounds_list = self.bounds_list,
                    parm_penalty = self.parm_penalty_split,
                    pvec_idx_min = mngr_idx_list,
                    #pvec_idx_min = None,
                    parallel_targets = parallel_targets,
                    bounds_penalty=bounds_penalty,
                    use_scipy = use_scipy,
                    verbose = self.verbose,
                    )

            if smarts_typing:
                worker_id_dict[ast] = worker_id, votes_state_list[counts], mngr_idx
            else:
                worker_id_dict[ast] = worker_id
            self.reset_parameters()

        ### Query move candidates
        ### =====================
        for ast in votes_move_list:
            allocation, selection, type_ = ast
            pvec_all = self.generate_parameter_vectors(
                [],
                system_idx_list,
                )
            pvec = pvec_all[mngr_idx]
            pvec.allocations[list(selection)] = list(allocation)
            pvec.apply_changes()

            worker_id = minimize_FF.remote(
                    system_list = system_list_id,
                    targetcomputer = self.targetcomputer_id,
                    pvec_list = pvec_all,
                    bounds_list = self.bounds_list,
                    parm_penalty = self.parm_penalty_split,
                    pvec_idx_min = mngr_idx,
                    #pvec_idx_min = None,
                    parallel_targets = parallel_targets,
                    bounds_penalty=bounds_penalty,
                    use_scipy = use_scipy,
                    verbose = self.verbose,
                    )

            worker_id_dict[ast] = worker_id
            self.reset_parameters()

        ### Query switch-on candidates
        ### ==========================
        for ast in votes_switch_on_list:
            allocation, selection, type_ = ast
            pvec_all = self.generate_parameter_vectors(
                [],
                system_idx_list,
                )
            pvec = pvec_all[mngr_idx]
            pvec.add_force_group(
                [0. * _FORCE_CONSTANT_TORSION for _ in range(pvec.parameters_per_force_group)]
                )
            pvec.allocations[list(selection)] = list(allocation)
            pvec.apply_changes()

            worker_id = minimize_FF.remote(
                    system_list = system_list_id,
                    targetcomputer = self.targetcomputer_id,
                    pvec_list = pvec_all,
                    bounds_list = self.bounds_list,
                    parm_penalty = self.parm_penalty_split,
                    pvec_idx_min = mngr_idx,
                    #pvec_idx_min = None,
                    parallel_targets = parallel_targets,
                    bounds_penalty=bounds_penalty,
                    use_scipy = use_scipy,
                    verbose = self.verbose,
                    )

            worker_id_dict[ast] = worker_id
            self.reset_parameters()

        ### Query merge candidates
        ### ======================
        for ast in votes_merge_list:
            allocation, selection, type_ = ast
            ### `type_[0]` is the type that everything is merged 'in' (non-empty type)
            ### `type_[1]` is the type that everything is merged 'from' (empty type)
            pvec_all = self.generate_parameter_vectors(
                [],
                system_idx_list,
                )
            pvec = pvec_all[mngr_idx]
            pvec.allocations[list(selection)] = type_[0]
            pvec.apply_changes()
            pvec.remove(type_[1])
            pvec.apply_changes()

            force_group_idxs = [[] for _ in range(self.N_mngr)]
            force_group_idxs[mngr_idx] = [pvec.allocations[list(selection)][0]]

            worker_id = minimize_FF.remote(
                    system_list = system_list_id,
                    targetcomputer = self.targetcomputer_id,
                    pvec_list = pvec_all,
                    bounds_list = self.bounds_list,
                    parm_penalty = self.parm_penalty_merge,
                    pvec_idx_min = mngr_idx,
                    #pvec_idx_min = None,
                    force_group_idxs = force_group_idxs,
                    parallel_targets = parallel_targets,
                    bounds_penalty=bounds_penalty,
                    use_scipy = use_scipy,
                    verbose = self.verbose,
                    )

            worker_id_dict[ast] = worker_id
            self.reset_parameters()

        return worker_id_dict
    

    def set_parameters(
        self,
        system_idx_list=list(),
        worker_id_dict=dict(),
        parm_penalty=1.,
        ):

        import copy
        import numpy as np
        from .molgraphs import get_smarts_score

        MAX_AIC = 9999999999999999.
        found_improvement = True

        if len(system_idx_list) == 0:
            system_idx_list = list(range(self.N_systems))

        old_AIC = self.calculate_AIC(
            [],
            system_idx_list,
            parm_penalty=parm_penalty
            )
        if self.verbose:
            print(
                "Current best AIC:", old_AIC
                )
        old_pvec_list = self.generate_parameter_vectors(
            system_idx_list=system_idx_list,
            as_copy=True
            )
        best_pvec_list = copy.deepcopy(old_pvec_list)
        best_ast       = None
        best_AIC       = MAX_AIC

        pvec_list = self.generate_parameter_vectors(
            system_idx_list=system_idx_list
            )

        ### For each system, find the best solution
        selection_worker_id_dict = dict()
        for ast in worker_id_dict:
            result = worker_id_dict[ast]
            if len(result) == 3:
                worker_id, state_list, mngr_idx_main = result
                is_smarts_typing_result = True
            else:
                worker_id = result
                is_smarts_typing_result = False
            _, _pvec_list = ray.get(worker_id)
            for full_reset in [True, False]:
                if (not is_smarts_typing_result) and full_reset:
                    continue

                for mngr_idx in range(self.N_mngr):
                    pvec_list[mngr_idx].reset(
                        old_pvec_list[mngr_idx]
                        )
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    if smarts_typing:
                        pvec_list[mngr_idx].apply_changes(True,True)
                    else:
                        pvec_list[mngr_idx].apply_changes()

                ### First set the parameter values to
                ### their optimized values.
                for mngr_idx in range(self.N_mngr):
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    if smarts_typing:
                        if full_reset:
                            pass
                        else:
                            ### If we don't do full reset,
                            ### then we only reset the main
                            ### pvec and everything else will be
                            ### at its best found state (so far).
                            if mngr_idx != mngr_idx_main:
                                continue
                        pvec_list[mngr_idx].reset(
                            _pvec_list[mngr_idx]
                            )
                        pvec_list[mngr_idx].apply_changes(True, True)
                    else:
                        pvec_list[mngr_idx].reset(
                            _pvec_list[mngr_idx]
                            )
                        pvec_list[mngr_idx].apply_changes()
                if is_smarts_typing_result:
                    best_smarts_score = 0.
                    typing_list = list()
                    for state in state_list:
                        pvec_list[mngr_idx_main].reset(
                            state,
                            ignore_values=True
                            )
                        pvec_list[mngr_idx_main].apply_changes(True, True)
                        typing_list.append(
                            pvec_list[mngr_idx_main].allocations[:].tolist()
                            )

                    worker_id = batch_likelihood_typing_remote.remote(
                        pvec_list[mngr_idx_main],
                        self.targetcomputer,
                        typing_list,
                        rebuild_from_systems=True
                        )
                    selection_worker_id_dict[worker_id] = self.generate_parameter_vectors(as_copy=True)

                    N_parms_all = 0.
                    for pvec in pvec_list:
                        N_parms_all += pvec.size

                    for idx, state in enumerate(state_list):
                        new_AIC  = 2. * N_parms_all * parm_penalty
                        new_AIC -= 2. * logL_list[idx]

                        accept = False
                        if new_AIC < best_AIC:
                            accept = True
                        else:
                            diff = np.abs(new_AIC - best_AIC)
                            if diff < 1.e-4:
                                smarts_score = 0.
                                for smarts_manager in state.smarts_manager_list:
                                    smarts_score += get_smarts_score(smarts_manager)
                                if smarts_score > best_smarts_score:
                                    best_smarts_score = smarts_score
                                    accept = True

                        if accept:
                            pvec_list[mngr_idx_main].reset(
                                state,
                                ignore_values=True
                                )
                            pvec_list[mngr_idx_main].apply_changes(True, True)

                            best_AIC       = new_AIC
                            best_pvec_list = copy.deepcopy(pvec_list)
                            best_ast       = ast
                else:
                    new_AIC = self.calculate_AIC(
                        [],
                        system_idx_list,
                        parm_penalty=parm_penalty
                        )
                    accept = False
                    if new_AIC < best_AIC:
                        accept = True

                    if accept:
                        best_AIC       = new_AIC
                        best_pvec_list = copy.deepcopy(pvec_list)
                        best_ast       = ast

        if best_AIC < old_AIC:
            for mngr_idx in range(self.N_mngr):
                typing_constraints = self.typing_constraints_list[mngr_idx]
                if not isinstance(typing_constraints, TypingConstraints):
                    result = True
                else:
                    result = typing_constraints.is_pvec_valid_hierarchy(
                        pvec_list[mngr_idx],
                        self.verbose
                        )
                    if not result:
                        diff = best_AIC - old_AIC
                        typing_temperature = self.typing_temperature_list[mngr_idx]
                        p = np.exp(-diff/typing_temperature)
                        u = np.random.random()
                        if u < p:
                            result = True

                if result:
                    found_improvement = True
                    pvec_list[mngr_idx].reset(
                        best_pvec_list[mngr_idx]
                        )
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    if smarts_typing:
                        pvec_list[mngr_idx].apply_changes(True, True)
                        self.update_best(
                            [pvec_list[mngr_idx]],
                            [mngr_idx]
                            )
                    else:
                        pvec_list[mngr_idx].apply_changes()

                else:
                    if self.verbose:
                        print(
                            "Typing violation."
                            )
                    found_improvement = False
                    break

        else:
            found_improvement = False

        if not found_improvement:
            pvec_list = self.generate_parameter_vectors(
                [],
                system_idx_list,
                )
            for mngr_idx in range(self.N_mngr):
                pvec_list[mngr_idx].reset(
                    old_pvec_list[mngr_idx],
                    )
                smarts_typing = self.smarts_typing_list[mngr_idx]
                if smarts_typing:
                    pvec_list[mngr_idx].apply_changes(True, True)
                    self.update_best(
                        [pvec_list[mngr_idx]],
                        [mngr_idx]
                        )                    
                else:
                    pvec_list[mngr_idx].apply_changes()

        if self.verbose:
            if best_ast != None:
                print("best_AIC:", best_AIC)
                if best_AIC < old_AIC:
                    print("Solution accepted.")
                else:
                    print("Solution not accepted.")
                for pvec in best_pvec_list:
                    if isinstance(pvec, SmartsForceFieldParameterVector):
                        print(
                            pvec.vector_k,
                            pvec.allocations,
                            pvec.get_smarts()
                        )
                    else:
                        print(
                            pvec.vector_k,
                            pvec.allocations
                        )
            else:
                print(
                    f"No move attempted. Best optimized: {best_AIC}"
                    )

        return found_improvement, pvec_list


    def run(
        self,
        ### Number of iterations in the outer loop.
        iterations=5,
        ### Maximum number of splitting attempts
        max_splitting_attempts=10,
        ### Maximum number of merging attempts
        max_merging_attempts=0,
        ### Maximum number of cluster merge attempts
        max_clustermerge_attempts = 10,
        ### Number of Gibbs Parameter Type relaxation attempts
        max_gibbs_attempts=0,
        ### Number of Gibbs Parameter Type relaxation attempts
        ### to spread out parameters after splitting
        max_gibbs_post_split_attempts=0,
        ### Number of gradient trials per parameter vector
        N_trials_gradient = 5,
        ### Any torsion parameter (i.e. its absolute value) 
        ### below this threshold, will be brute force switched 
        ### off (i.e. set to zero).
        ### The larger, the more parameters will be switched off
        switch_min_val = 1.e-2,
        ### Multiply merge cutoff radius in each manager by this number
        ### The larger `merge_cutoff_scale`, the more merging occurs.
        merge_cutoff_scale = 1.,
        ### Maximum number of type scrambling attempts after
        ### failed type merging attempt.
        max_scramble_iterations = 1,
        ### Per iteration Type-Mixing-Iteration, the clustering radius
        ### is increased by this fraction of `merge_cutoff_scale`
        merge_radius_grow_factor = 0.05,
        ### Use `pre_smirksify` to merge some types so that
        ### later-on they can be processed by chemper.SMIRKSifier
        use_pre_smirksify = True,
        ### Every `pair_incr` iterations of the outer looop, 
        ### the number of systems per batch `N_sys_per_batch`
        ### is incremented by +1.
        pair_incr_split = 10,
        pair_incr_merge = 10,
        ### Initial number of systems per batch
        N_sys_per_batch_split = 1,
        N_sys_per_batch_merge = 1,
        ### Maximum size of split/switch-on. For a given parameter type,
        ### only a maximum of `max_group_size` groups will be
        ### split/switched on.
        max_group_size=2,
        ### penalty constant for quadratic penalty function
        ### during parameter optimization through MLE.
        bounds_penalty=1000.,
        ### Optimize ordering in which systems are computed
        optimize_system_ordering=True,
        ### Prefix used for saving checkpoints
        prefix="ff_hopper",
        ):

        self.update_best()
        self.save_traj(parm_penalty=1.)

        draw_typing_vector_remote = ray.remote(draw_typing_vector)

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
                if (iteration_idx%pair_incr_merge) == 0:
                    N_sys_per_batch_merge += 1

            raute_fill = ''.ljust(len(str(iteration_idx))-1,"#")
            print("ITERATION", iteration_idx)
            print(f"###########{raute_fill}")

            ### ============== ###
            ### PROCESS SPLITS ###
            ### ============== ###
            split_iteration_idx = 0
            found_improvement   = True
            while found_improvement and split_iteration_idx < max_splitting_attempts:
                print(f"ATTEMPTING SPLIT/SWITCH-ON {iteration_idx}/{split_iteration_idx}")
                found_improvement       = False
                system_idx_list_batch   = self.get_random_system_idx_list(N_sys_per_batch_split)
                split_worker_id_dict    = dict()
                switchon_worker_id_dict = dict()
                move_worker_id_dict     = dict()

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
                        pvec = self.generate_parameter_vectors([0], sys_idx_pair)[0]
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
                for mngr_idx in range(self.N_mngr):
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    for sys_idx_pair in system_idx_list_batch:
                        pvec = self.generate_parameter_vectors(
                            [mngr_idx],
                            sys_idx_pair
                            )[0]
                        split_worker_id_dict[mngr_idx,sys_idx_pair],\
                        move_worker_id_dict[mngr_idx,sys_idx_pair],\
                        switchon_worker_id_dict[mngr_idx,sys_idx_pair],\
                        _,\
                        =  self.get_grad_scores(
                            pvec, 
                            max_group_size,
                            N_trials_gradient, 
                            self.allocation_scorer_list[mngr_idx],
                            split=True,
                            move=allow_move,
                            switch_on=True,
                            merge=False,
                            smarts_typing=smarts_typing
                            )

                if self.verbose:
                    print(
                        "Obtaining votes and submitting optimizations."
                        )
                worker_id_dict = dict()
                for mngr_idx in range(self.N_mngr):
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    for sys_idx_pair in system_idx_list_batch:
                        keep_N_best=3
                        ### Split
                        if smarts_typing:
                            votes_split_list, votes_state_list = self.get_votes(
                                worker_id_dict=split_worker_id_dict[mngr_idx,sys_idx_pair], 
                                low_to_high=True,
                                abs_grad_score=False,
                                norm_cutoff=1.e-2,
                                keep_N_best=keep_N_best,
                                smarts_typing=smarts_typing,
                                )
                        else:
                            votes_split_list = self.get_votes(
                                worker_id_dict=split_worker_id_dict[mngr_idx,sys_idx_pair], 
                                low_to_high=True,
                                abs_grad_score=False,
                                norm_cutoff=1.e-2,
                                keep_N_best=keep_N_best,
                                smarts_typing=smarts_typing,
                                )

                        ### Move
                        votes_move_list = self.get_votes(
                            worker_id_dict=move_worker_id_dict[mngr_idx,sys_idx_pair], 
                            low_to_high=True,
                            abs_grad_score=False,
                            norm_cutoff=1.e-2,
                            keep_N_best=keep_N_best,
                            )
                        ### Switch-on
                        votes_switch_on_list = self.get_votes(
                            worker_id_dict=switchon_worker_id_dict[mngr_idx,sys_idx_pair], 
                            low_to_high=False,
                            abs_grad_score=True,
                            norm_cutoff=1.e-2,
                            keep_N_best=keep_N_best,
                            )

                        worker_id_dict[mngr_idx,sys_idx_pair] = self.get_min_scores(
                            mngr_idx=mngr_idx,
                            system_idx_list=sys_idx_pair,
                            votes_split_list=votes_split_list,
                            votes_move_list=votes_move_list,
                            votes_switch_on_list=votes_switch_on_list,
                            parallel_targets=False,
                            bounds_penalty=bounds_penalty,
                            smarts_typing=smarts_typing,
                            votes_state_list=votes_state_list,
                            )

                        if self.verbose:
                            print(
                                f"For mngr {mngr_idx} and systems {sys_idx_pair}:\n"
                                f"Found {len(votes_split_list)} candidate split solutions ...\n",
                                f"Found {len(votes_move_list)} candidate move solutions ...\n",
                                f"Found {len(votes_switch_on_list)} candidate switch solutions ..."
                                )

                if self.verbose:
                    print(
                        "Selecting best parameters."
                        )
                system_idx_list_batch = system_idx_list_batch[::-1]
                gibbs_dict = dict()
                gibbs_count_dict = dict()
                accepted_counter = 0
                for mngr_idx_main in range(self.N_mngr):
                    smarts_typing_main = self.smarts_typing_list[mngr_idx_main]
                    selection_worker_id_dict = dict()
                    AIC_best_dict = dict()
                    old_pvec_list_dict = dict()
                    for sys_idx_pair in system_idx_list_batch:
                        ### If we are using smarts typing
                        ### we want to compute AIC for the whole dataset.
                        if smarts_typing_main:
                            _sys_idx_pair = tuple()
                        else:
                            _sys_idx_pair = sys_idx_pair
                        if _sys_idx_pair in old_pvec_list_dict:
                            old_pvec_list = old_pvec_list_dict[_sys_idx_pair]
                        else:
                            old_pvec_list = self.generate_parameter_vectors(
                                system_idx_list = _sys_idx_pair
                                )
                            old_pvec_list_dict[_sys_idx_pair] = old_pvec_list
                        worker_id = set_parameters_remote.remote(
                            old_pvec_list_init = old_pvec_list,
                            targetcomputer = self.targetcomputer_id,
                            smarts_typing_list = self.smarts_typing_list,
                            worker_id_dict = worker_id_dict[mngr_idx_main,sys_idx_pair],
                            parm_penalty = self.parm_penalty_split,
                            typing_constraints_list = self.typing_constraints_list,
                            typing_temperature_list = self.typing_temperature_list,
                            verbose = self.verbose,
                            )
                        if not _sys_idx_pair in AIC_best_dict:
                            AIC_best_dict[_sys_idx_pair] = self.calculate_AIC(
                                system_idx_list = _sys_idx_pair,
                                parm_penalty = self.parm_penalty_split,
                                )

                        selection_worker_id_dict[sys_idx_pair] = worker_id, _sys_idx_pair

                    for sys_idx_pair in selection_worker_id_dict:
                        if self.verbose:
                            print("mngr_idx/sys_idx_pair", mngr_idx_main, "/", sys_idx_pair)

                        worker_id, _sys_idx_pair = selection_worker_id_dict[sys_idx_pair]
                        _, pvec_list, best_AIC = ray.get(worker_id)
                        old_AIC = AIC_best_dict[_sys_idx_pair]

                        if self.verbose:
                            print(
                                "Current best AIC:", old_AIC,
                                "New AIC:", best_AIC,
                                )

                        found_improvement_mngr = False
                        if best_AIC < old_AIC:
                            AIC_best_dict[_sys_idx_pair] = best_AIC
                            found_improvement_mngr = True

                        if found_improvement_mngr:
                            found_improvement = True
                            if self.verbose:
                                print("Solution globally accepted.")
                            old_pvec_list = old_pvec_list_dict[_sys_idx_pair]
                            for mngr_idx in range(self.N_mngr):
                                old_pvec_list[mngr_idx].reset(
                                    pvec_list[mngr_idx]
                                    )
                                smarts_typing = self.smarts_typing_list[mngr_idx]
                                if smarts_typing:
                                    old_pvec_list[mngr_idx].apply_changes(True, True)
                                    self.update_best(
                                        [old_pvec_list[mngr_idx]],
                                        [mngr_idx]
                                        )
                                    if self.verbose:
                                        print(
                                            old_pvec_list[mngr_idx].vector_k,
                                            old_pvec_list[mngr_idx].allocations,
                                            old_pvec_list[mngr_idx].get_smarts()
                                        )
                                else:
                                    old_pvec_list[mngr_idx].apply_changes()
                                    if self.verbose:
                                        print(
                                            old_pvec_list[mngr_idx].vector_k,
                                            old_pvec_list[mngr_idx].allocations
                                        )
                            old_pvec_list_dict[_sys_idx_pair] = old_pvec_list

                            if not smarts_typing_main:
                                if sys_idx_pair in gibbs_dict:
                                    gibbs_dict[sys_idx_pair].append(mngr_idx_main)
                                else:
                                    gibbs_dict[sys_idx_pair] = [mngr_idx_main]
                                    gibbs_count_dict[sys_idx_pair] = 0

                            import pickle
                            with open(f"{prefix}-MAIN-{iteration_idx}-SPLIT-{split_iteration_idx}-ACCEPTED-{accepted_counter}.pickle", "wb") as fopen:
                                pickle.dump(
                                    self,
                                    fopen
                                )
                            accepted_counter += 1

                        else:
                            if self.verbose:
                                print("Solution globally rejected.")

                    if found_improvement:
                        self.update_best()
                        self.reset_parameters()
                
                if self.verbose:
                    print(
                        "Gibbs type refinment."
                        )
                while len(gibbs_dict) > 0:

                    found_improvement = False
                    worker_id_dict = dict()
                    for sys_idx_pair in gibbs_dict:
                        mngr_idx_list = gibbs_dict[sys_idx_pair]
                        gibbs_count_dict[sys_idx_pair] += 1
                        pvec_list = self.generate_parameter_vectors(
                            mngr_idx_list = mngr_idx_list,
                            system_idx_list = sys_idx_pair,
                            )
                        switching = list()
                        allocation_scorer_list = list()
                        typing_constraints_list = list()
                        smarts_typing_list = list()
                        for idx, mngr_idx in enumerate(mngr_idx_list):
                            smarts_typing_list.append(
                                self.smarts_typing_list[mngr_idx]
                                )
                            switching.append(
                                self.switching[mngr_idx]
                                )
                            allocation_scorer_list.append(
                                AllocationScorePrior(
                                    self.allocation_prior_constraints_list[mngr_idx],
                                    pvec_list[idx].parameter_manager
                                    ),
                                )
                            typing_constraints_list.append(
                                self.typing_constraints_list[mngr_idx]
                                )
                        worker_id = draw_typing_vector_remote.remote(
                            pvec_list,
                            self.targetcomputer_id,
                            typing_constraints_list,
                            switching = switching,
                            N_attempts = 10,
                            alpha_list = None,
                            weight_list = None,
                            #sigma_list = [
                            #    np.ones(pvec.force_group_count, dtype=float) for pvec in pvec_list
                            #],
                            #typing_prior = "jeffreys",
                            allocation_scorer_list = allocation_scorer_list,
                            typing_prior = "allocation-score",
                            parallel_targets = False,
                            draw_maximum=True,
                            smarts_typing=smarts_typing_list,
                            )
                        worker_id_dict[sys_idx_pair] = worker_id

                    for sys_idx_pair in worker_id_dict:
                        worker_id = worker_id_dict[sys_idx_pair]
                        allocations_list_new, state_all_list_new = ray.get(worker_id)
                        mngr_idx_list = gibbs_dict[sys_idx_pair]
                        pvec_list = self.generate_parameter_vectors(
                            mngr_idx_list,
                            )
                        AIC_old = self.calculate_AIC(
                            mngr_idx_list,
                            parm_penalty=self.parm_penalty_split
                        )
                        state_all_list_old = self.generate_parameter_vectors(
                            mngr_idx_list,
                            as_copy=True
                            )
                        for idx, mngr_idx in enumerate(mngr_idx_list):
                            pvec_list[idx].reset(
                                state_all_list_new[idx]
                                )
                            smarts_typing = self.smarts_typing_list[mngr_idx]
                            if smarts_typing:
                                pvec_list[idx].apply_changes(True, True)
                            else:
                                pvec_list[idx].apply_changes()
                        AIC_new = self.calculate_AIC(
                            mngr_idx_list,
                            parm_penalty=self.parm_penalty_split
                        )
                        ### Reject
                        if AIC_new < AIC_old:
                            found_improvement_sys_idx = True
                        else:
                            found_improvement_sys_idx = False

                        if found_improvement_sys_idx:
                            found_improvement = True

                        if found_improvement_sys_idx:
                            for idx, mngr_idx in enumerate(mngr_idx_list):
                                pvec_list[idx].reset(
                                    state_all_list_old[idx]
                                    )
                                smarts_typing = self.smarts_typing_list[mngr_idx]
                                if smarts_typing:
                                    pvec_list[idx].apply_changes(True, True)
                                    self.update_best(
                                        [pvec_list[idx]],
                                        [mngr_idx]
                                        )
                                else:
                                    pvec_list[idx].apply_changes()
                                if sys_idx_pair in gibbs_dict:
                                    del gibbs_dict[sys_idx_pair]

                        if self.verbose:
                            print(
                                "Cluster merge",
                                "AIC_old:", AIC_old,
                                "AIC_new:", AIC_new,
                                )
                            if found_improvement_sys_idx:
                                print(
                                    "Found improvement."
                                    )
                            else:
                                print(
                                    "Did not find improvement."
                                    )
                            for idx, mngr_idx in enumerate(mngr_idx_list):
                                pvec = pvec_list[idx]
                                if self.verbose:
                                    print(
                                        "mngr_idx/sys_idx_pair", 
                                        mngr_idx, 
                                        "/", 
                                        sys_idx_pair
                                        )
                                    smarts_typing = self.smarts_typing_list[mngr_idx]
                                    if smarts_typing:
                                        print(
                                            pvec.vector_k,
                                            pvec.allocations,
                                            pvec.get_smarts(),
                                            )
                                    else:
                                        print(
                                            pvec.vector_k,
                                            pvec.allocations,
                                            )

                        if gibbs_count_dict[sys_idx_pair] > max_gibbs_post_split_attempts:
                            if sys_idx_pair in gibbs_dict:
                                del gibbs_dict[sys_idx_pair]

                    if found_improvement:
                        self.update_best()
                        self.reset_parameters()

                import pickle
                with open(f"{prefix}-MAIN-{iteration_idx}-SPLIT-{split_iteration_idx}.pickle", "wb") as fopen:
                    pickle.dump(
                        self,
                        fopen
                    )

                split_iteration_idx += 1

            self.reset_parameters()

            ### ============== ###
            ### PROCESS MERGES ###
            ### ============== ###
            merge_iteration_idx = 0
            found_improvement   = True
            while found_improvement and merge_iteration_idx < max_merging_attempts:
                print(f"ATTEMPTING MERGE {iteration_idx}/{merge_iteration_idx}")
                found_improvement = False
                system_idx_list_batch = self.get_random_system_idx_list(N_sys_per_batch_merge)
                merge_worker_id_dict  = dict()
                for mngr_idx in range(self.N_mngr):
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    if smarts_typing:
                        continue
                    for sys_idx_pair in system_idx_list_batch:
                        pvec = self.generate_parameter_vectors(
                            [mngr_idx],
                            sys_idx_pair,
                            )[0]

                        _,\
                        _,\
                        _,\
                        merge_worker_id_dict[mngr_idx,sys_idx_pair],\
                        =  self.get_grad_scores(
                            pvec, 
                            max_group_size,
                            N_trials_gradient, 
                            self.allocation_scorer_list[mngr_idx],
                            split=False,
                            move=False,
                            switch_on=False,
                            merge=True
                            )

                worker_id_dict = dict()
                for mngr_idx in range(self.N_mngr):
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    if smarts_typing:
                        continue
                    for sys_idx_pair in system_idx_list_batch:
                        votes_merge_list = self.get_votes(
                            worker_id_dict=merge_worker_id_dict[mngr_idx,sys_idx_pair], 
                            low_to_high=False,
                            abs_grad_score=False,
                            norm_cutoff=1.e-2
                            )
                        parallel_targets = True
                        if len(sys_idx_pair) > 3:
                            parallel_targets = True
                        else:
                            parallel_targets = False

                        worker_id_dict[mngr_idx,sys_idx_pair] = self.get_min_scores(
                            mngr_idx=mngr_idx,
                            system_idx_list=sys_idx_pair,
                            votes_merge_list=votes_merge_list,
                            parallel_targets=parallel_targets,
                            bounds_penalty=bounds_penalty,
                            )

                for mngr_idx in range(self.N_mngr):
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    if smarts_typing:
                        continue
                    for sys_idx_pair in system_idx_list_batch:
                        if self.verbose:
                            print("mngr_idx/sys_idx_pair", mngr_idx, "/", sys_idx_pair)
                        found_improvement_mngr, pvec_list = self.set_parameters(
                            system_idx_list = sys_idx_pair,
                            worker_id_dict = worker_id_dict[mngr_idx,sys_idx_pair],
                            parm_penalty = self.parm_penalty_merge,
                            )
                        if found_improvement_mngr:
                            found_improvement = True
                            self.update_best(
                                [pvec_list[mngr_idx]],
                                [mngr_idx]
                                )

                if found_improvement:
                    self.update_best()
                    self.reset_parameters()

                import pickle
                with open(f"{prefix}-MAIN-{iteration_idx}-MERGE-{merge_iteration_idx}.pickle", "wb") as fopen:
                    pickle.dump(
                        self,
                        fopen
                    )

                merge_iteration_idx += 1

            self.reset_parameters()

            parallel_targets = True
            if self.N_systems > 3:
                parallel_targets = True
            else:
                parallel_targets = False
            clustermerge_iteration_idx = 0
            found_improvement_global   = True
            _merge_cutoff_scale = merge_cutoff_scale
            while found_improvement_global and (clustermerge_iteration_idx < max_clustermerge_attempts):
                print(
                    f"ATTEMPTING GIBBS-CLUSTERMERGE {iteration_idx}/{clustermerge_iteration_idx}"
                    )
                found_improvement_global = False
                old_AIC_global = self.calculate_AIC(
                    parm_penalty=self.parm_penalty_merge,
                    )
                old_pvec_list_global = self.generate_parameter_vectors(as_copy=True)
                if self.verbose:
                    print(
                        f"AIC global: {old_AIC_global}"
                        )
                ### Cluster parameters
                self.reduce_parameters(
                    switch_min_val,
                    _merge_cutoff_scale,
                    use_pre_smirksify
                    )
                ### Minimize parameters
                pvec_list_all = self.generate_parameter_vectors()
                worker_id = minimize_FF.remote(
                        system_list = self.system_list,
                        targetcomputer = self.targetcomputer_id,
                        pvec_list = pvec_list_all,
                        bounds_list = self.bounds_list,
                        parm_penalty = self.parm_penalty_merge,
                        pvec_idx_min = None,
                        force_group_idxs = None,
                        parallel_targets = parallel_targets,
                        bounds_penalty = bounds_penalty,
                        use_scipy = True,
                        verbose = self.verbose,
                        )
                _, best_pvec_list = ray.get(worker_id)
                for mngr_idx in range(self.N_mngr):
                    pvec_list_all[mngr_idx].reset(
                        best_pvec_list[mngr_idx],
                    )
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    if smarts_typing:
                        pvec_list_all[mngr_idx].apply_changes(True, True)
                        self.update_best(
                            [pvec_list_all[mngr_idx]],
                            [mngr_idx]
                            )
                    else:
                        pvec_list_all[mngr_idx].apply_changes(True, True)

                old_AIC = self.calculate_AIC(
                    parm_penalty=self.parm_penalty_merge,
                    )
                old_pvec_list = self.generate_parameter_vectors(as_copy=True)
                ### Mix and Optimize in alternating order until
                ### we don't improve anymore.
                scramble_iterations = 0
                found_improvement = True
                while found_improvement or scramble_iterations < max_scramble_iterations:
                    found_improvement = False
                    if self.verbose:
                        print("Before Type Minimization Cycle:")
                        _pvec_list = self.generate_parameter_vectors()
                        for mngr_idx in range(self.N_mngr):
                            smarts_typing = self.smarts_typing_list[mngr_idx]
                            if smarts_typing:
                                print(
                                    mngr_idx,
                                    _pvec_list[mngr_idx].vector_k,
                                    _pvec_list[mngr_idx].allocations,
                                    _pvec_list[mngr_idx].get_smarts(),
                                )
                            else:
                                print(
                                    mngr_idx,
                                    _pvec_list[mngr_idx].vector_k,
                                    _pvec_list[mngr_idx].allocations,
                                )
                        _AIC = self.calculate_AIC(
                            parm_penalty=self.parm_penalty_merge,
                            )
                        print(
                            f"AIC Local before Type Minimization Cycle: {_AIC}"
                            )
                    ### Do the minimization...
                    allocation_scorer_list = list()
                    typing_constraints_list = list()
                    switching = list()
                    pvec_list = list()
                    mngr_idx_list = list()
                    for mngr_idx in range(self.N_mngr):
                        smarts_typing = self.smarts_typing_list[mngr_idx]
                        if not smarts_typing:
                            switching.append(
                                self.switching[mngr_idx]
                                )
                            allocation_scorer_list.append(
                                AllocationScorePrior(
                                    self.allocation_prior_constraints_list[mngr_idx],
                                    pvec_list_all[mngr_idx].parameter_manager
                                    ),
                                )
                            typing_constraints_list.append(
                                self.typing_constraints_list[mngr_idx]
                                )
                            pvec_list_all.append(
                                self.generate_parameter_vectors([mngr_idx])[0]
                                )
                            mngr_idx_list.append(mngr_idx)

                    if len(mngr_idx_list) == 0:
                        if self.verbose:
                            print(
                                "Did not find any mngr for type optimization."
                                "No type optimization attempted."
                                )
                        break

                    allocations_list, state_list = draw_typing_vector(
                        pvec_list,
                        self.targetcomputer_id,
                        typing_constraints_list,
                        switching = switching,
                        N_attempts = 5,
                        alpha_list = None,
                        weight_list = None,
                        sigma_list = None,
                        allocation_scorer_list = allocation_scorer_list,
                        typing_prior = "allocation-score",
                        parallel_targets = True,
                        draw_maximum=True
                        )

                    pvec_list = self.generate_parameter_vectors()
                    for idx, mngr_idx in enumerate(mngr_idx_list):
                        pvec_list[mngr_idx].reset(
                            state_list[idx]
                            )
                        smarts_typing = self.smarts_typing_list[mngr_idx]
                        if smarts_typing:
                            pvec_list[mngr_idx].apply_changes(True, True)
                            self.update_best(
                                [pvec_list[mngr_idx]],
                                [mngr_idx]
                                )
                        else:
                            pvec_list[mngr_idx].apply_changes()

                    if self.verbose:
                        _pvec_list = self.generate_parameter_vectors()
                        print("After Type Minimization Cycle:")
                        for idx, mngr_idx in enumerate(mngr_idx_list):
                            smarts_typing = self.smarts_typing_list[mngr_idx]
                            if smarts_typing:
                                print(
                                    mngr_idx,
                                    _pvec_list[idx].vector_k,
                                    _pvec_list[idx].allocations,
                                    _pvec_list[idx].get_smarts()
                                )
                            else:
                                print(
                                    mngr_idx,
                                    _pvec_list[idx].vector_k,
                                    _pvec_list[idx].allocations,
                                )
                        _AIC = self.calculate_AIC(
                            parm_penalty=self.parm_penalty_merge,
                            )
                        print(
                            f"AIC Local after Type Minimization Cycle: {_AIC}"
                            )

                    ### Do the minimization...
                    pvec_list_all = self.generate_parameter_vectors()
                    worker_id = minimize_FF.remote(
                            system_list = self.system_list,
                            targetcomputer = self.targetcomputer_id,
                            pvec_list = pvec_list_all,
                            bounds_list = self.bounds_list,
                            parm_penalty = self.parm_penalty_merge,
                            pvec_idx_min = None,
                            force_group_idxs = None,
                            parallel_targets = parallel_targets,
                            bounds_penalty = bounds_penalty,
                            use_scipy = True,
                            verbose = self.verbose,
                            )
                    best_AIC, best_pvec_list = ray.get(worker_id)

                    if self.verbose:
                        print(
                            f"Old AIC: {old_AIC}",
                            f"New AIC: {best_AIC}",
                            )

                    if best_AIC < old_AIC:
                        found_improvement = True

                    if found_improvement:
                        scramble_iterations = 0
                        ### Update parameters.
                        for mngr_idx in range(self.N_mngr):
                            pvec_list_all[mngr_idx].reset(
                                best_pvec_list[mngr_idx]
                                )
                            smarts_typing = self.smarts_typing_list[mngr_idx]
                            if smarts_typing:
                                pvec_list_all[mngr_idx].apply_changes(True, True)
                                self.update_best(
                                    [pvec_list_all[mngr_idx]],
                                    [mngr_idx]
                                    )
                            else:
                                pvec_list_all[mngr_idx].apply_changes(True, True)

                        old_AIC = best_AIC
                        old_pvec_list = best_pvec_list

                        if self.verbose:
                            print(
                                "Local Accepted."
                                )
                    elif scramble_iterations < max_scramble_iterations:
                        if self.verbose:
                            print(
                                "Local rejected.",
                                "Attempting type scrambling."
                                )
                        
                        allocation_scorer_list = list()
                        typing_constraints_list = list()
                        switching = list()
                        pvec_list_all = list()
                        mngr_idx_list = list()
                        for mngr_idx in range(self.N_mngr):
                            smarts_typing = self.smarts_typing_list[mngr_idx]
                            if not smarts_typing:
                                switching.append(
                                    self.switching[mngr_idx]
                                    )
                                allocation_scorer_list.append(
                                    AllocationScorePrior(
                                        self.allocation_prior_constraints_list[mngr_idx],
                                        pvec_list_all[mngr_idx].parameter_manager
                                        ),
                                    )
                                typing_constraints_list.append(
                                    self.typing_constraints_list[mngr_idx]
                                    )
                                pvec_list_all.append(
                                    self.generate_parameter_vectors([mngr_idx])[0]
                                    )
                                mngr_idx_list.append(mngr_idx)
                        
                        if len(mngr_idx_list) == 0:
                            if self.verbose:
                                print(
                                    "Did not find any mngr for type optimization.",
                                    "No type scrambling attempted."
                                    )
                            break

                        allocations_list, state_list = draw_typing_vector(
                            pvec_list_all,
                            self.targetcomputer_id,
                            typing_constraints_list,
                            switching = switching,
                            N_attempts = 5,
                            alpha_list = None,
                            weight_list = None,
                            sigma_list = None,
                            allocation_scorer_list = allocation_scorer_list,
                            typing_prior = "allocation-score",
                            parallel_targets = True,
                            ### IMPORTANT: Here we don't want to
                            ###            draw maximum but rather
                            ###            do some sampling.
                            draw_maximum=False
                            )
                        pvec_list = self.generate_parameter_vectors()
                        for idx, mngr_idx in enumerate(mngr_idx_list):
                            pvec_list[mngr_idx].reset(
                                state_list[idx]
                                )
                            smarts_typing = self.smarts_typing_list[mngr_idx]
                            if smarts_typing:
                                pvec_list[mngr_idx].apply_changes(True, True)
                                self.update_best(
                                    [pvec_list[mngr_idx]],
                                    [mngr_idx]
                                    )
                            else:
                                pvec_list[mngr_idx].apply_changes()

                        scramble_iterations += 1

                ### `old_pvec_list` will not be altered if don't actually
                ### improve. So it is safe to just blindly use it here to
                ### set the global parameters.
                for mngr_idx in range(self.N_mngr):
                    pvec_list_all[mngr_idx].reset(
                        old_pvec_list[mngr_idx],
                    )
                    smarts_typing = self.smarts_typing_list[mngr_idx]
                    if smarts_typing:
                        pvec_list_all[mngr_idx].apply_changes(True, True)
                        self.update_best(
                            [pvec_list_all[mngr_idx]],
                            [mngr_idx]
                            )
                    else:
                        pvec_list_all[mngr_idx].apply_changes()

                if old_AIC < old_AIC_global:
                    if self.verbose:
                        print(
                            "Global Accepted."
                            )
                    found_improvement_global = True
                else:
                    if self.verbose:
                        print(
                            "Global Rejected."
                            )
                    found_improvement_global = False

                if not found_improvement_global:
                    for mngr_idx in range(self.N_mngr):
                        pvec_list_all[mngr_idx].reset(
                            old_pvec_list_global[mngr_idx],
                        )
                        smarts_typing = self.smarts_typing_list[mngr_idx]
                        if smarts_typing:
                            pvec_list_all[mngr_idx].apply_changes(True, True)
                            self.update_best(
                                [pvec_list_all[mngr_idx]],
                                [mngr_idx]
                                )
                        else:
                            pvec_list_all[mngr_idx].apply_changes()

                self.update_best()
                self.reset_parameters()

                import pickle
                with open(f"{prefix}-MAIN-{iteration_idx}-CLUSTERMERGE-{clustermerge_iteration_idx}.pickle", "wb") as fopen:
                    pickle.dump(
                        self,
                        fopen
                    )

                clustermerge_iteration_idx += 1

                _merge_cutoff_scale += merge_radius_grow_factor * merge_cutoff_scale

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


class ForceFieldSampler(object):

    def __init__(
        self, 
        system_list, 
        verbose=False):

        import ray

        self.bounds_list = list()
        self.switching = list()

        self.parameter_manager_list = list()
        self.parameter_name_list = list()
        self.scaling_factor_list = list()

        self.targetcomputer_id = ray.put(
            TargetComputer(
                self.system_list
            )
            )
        self.targetcomputer = ray.get(
            self.targetcomputer_id
            )

        self.pvec_traj    = list()

        self._N_steps   = 0
        self._N_mngr    = 0

        self.system_list     = system_list
        self._N_systems      = len(system_list)

        self.grad_diff = 1.e-2

        ### Note, don't make this too small.
        ### Too small perturbations will lead to
        ### not finding the correct splits
        self.perturbation = 1.e-1

        self.verbose = verbose

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

    @property
    def N_systems(self):
        return self._N_systems

    @property
    def N_mngr(self):
        return self._N_mngr

    def add_parameters(
        self,
        parameter_manager,
        parameter_name_list = None,
        scale_list = None,
        switching = False,
        bounds = None,
        ):

        if parameter_manager.N_systems != 0:
            raise ValueError(
                f"parameter_manager must be empty, but found {parameter_manager.N_systems} systems."
                )

        self.parameter_manager_list.append(parameter_manager)
        self.parameter_name_list.append(parameter_name_list)
        self.scaling_factor_list.append(scale_list)

        if bounds == None:
            bounds = [
                [-np.inf, np.inf],
                [-np.inf, np.inf],
            ]
        self.bounds_list.append(bounds)

        self.switching.append(switching)

        ### Must increment at the end, not before.
        self._N_mngr += 1


    def generate_parameter_vectors(
        self, 
        mngr_idx_list = list(),
        system_idx_list = list()
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

        pvec_list = list()
        for mngr_idx in _mngr_idx_list:
            parm_mngr = copy.deepcopy(
                self.parameter_manager_list[mngr_idx]
                )
            for sys_idx in _system_idx_list:
                parm_mngr.add_system(
                    self.system_list[sys_idx]
                    )
            pvec = ForceFieldParameterVector(
                parm_mngr,
                self.parameter_name_list[mngr_idx],
                self.scaling_factor_list[mngr_idx]
                )
            pvec_list.append(pvec)
        return pvec_list


    def calc_log_likelihood(
        self, 
        system_idx_list = list()
        ):

        if len(system_idx_list) == 0:
            _system_idx_list = list(range(self.N_systems))
        else:
            _system_idx_list = system_idx_list

        worker_id_list = list()
        for sys_idx in _system_idx_list:
            sys = self.system_list[sys_idx]
            worker_id = self.targetcomputer(
                {
                    sys.name : [sys.openmm_system]
                },
                False
                )
            worker_id_list.append(worker_id)

        logP_likelihood = 0.
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(worker_id_list)
            _logP_likelihood = ray.get(worker_id[0])
            logP_likelihood += _logP_likelihood

        return logP_likelihood

    def calc_log_prior(
        self,
        mngr_idx_list = list()
        ):

        if len(mngr_idx_list) == 0:
            _mngr_idx_list = list(range(self.N_systems))
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

        log_prior_val, log_prior_alloc = calc_log_prior(
            pvec_list,
            bounds_list = bonds_list,
            alpha_list = alpha_list,
            weight_list = weight_list,
            sigma_parameter_list = None,
            sigma_list = None,
            parameter_prior = "gaussian",
            typing_prior = "multinomial-dirichlet",
            )

        return log_prior_val, log_prior_alloc


    def draw_typing_vector(
        self,
        ):

        alpha_list = [
            1.e+3 * np.ones(pvec.force_group_count, dtype=float) for pvec in self.pvec_list_all
            ]
        weight_list = [
            1.e+3 * np.ones(pvec.force_group_count, dtype=float) for pvec in self.pvec_list_all
            ]

        draw_typing_vector(
            self.pvec_list_all,
            self.targetcomputer_id,
            switching = self.switching,
            N_attempts = 1,
            alpha_list = alpha_list,
            weight_list = weight_list,
            typing_prior = "multinomial-dirichlet-conditional",
            parallel_targets=True
            )

        log_prior = calc_typing_log_prior(
                [pvec.allocations[:] for pvec in self.pvec_list_all],
                [pvec.force_group_count for pvec in pvec_list],
                alpha_list = alpha_list,
                weight_list = alpha_list,
                typing_prior = "multinomial-dirichlet-conditional",
                )


    def draw_parameter_vector(
        self,
        ):

        """
        RJMC with lifting, then proposing. We use
        metropolis-adjusted Langevin sampling to propagate
        the parameters.
        """

        from .constants import _INACTIVE_GROUP_IDX, _FORCE_CONSTANT_TORSION

        normalize_gradients = False

        sig_langevin  = 5.e-4
        sig_symm = 1.e-1

        sig_langevin2 = sig_langevin**2
        sig_symm2 = sig_symm**2

        from scipy import stats

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

        alloc_omit_list = [list() for _ in range(self.N_mngr)]

        if split:
            print("split ...")
            ### First do the lifting move.
            ### Randomly select one parameter type
            pvec_idx = np.random.choice(
                np.arange(self.N_mngr)
                )
            pvec = self.pvec_list_all[pvec_idx]
            if self.switching[pvec_idx]:
                ### Only do birth moves, if there is actually
                ### a group with _INACTIVE_GROUP_IDX type
                inactives = pvec.allocations.index([_INACTIVE_GROUP_IDX])[0]
                if len(inactives) > 0:
                    birth_move = True
                else:
                    birth_move = False
            else:
                if pvec.force_group_count == 0:
                    print("Rejected.")
                    return alloc_omit_list
                else:
                    birth_move = False

            hist = np.where(pvec.force_group_histogram > 0)[0]
            if birth_move:
                _hist = np.append(hist, _INACTIVE_GROUP_IDX)
                hist  = _hist
            ### Now find the type we want to duplicate from
            type_i = np.random.choice(
                hist,
                replace=True
                )
            if type_i == _INACTIVE_GROUP_IDX:
                pvec.add_force_group(
                    [0. * _FORCE_CONSTANT_TORSION for _ in range(pvec.parameters_per_force_group)]
                    )
            else:
                pvec.duplicate(type_i)
                pvec.apply_changes()

            ### Split the parameter type randomly
            N_types   = pvec.force_group_count
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
                self.grad_diff,
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
                self.grad_diff,
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
            if self.switching[pvec_idx]:
                ### Only do death moves if there is actually
                ### something that is ready to die.
                if pvec.force_group_count > 0:
                    death_move = True
                else:
                    print("Rejected.")
                    return alloc_omit_list
            else:
                if pvec.force_group_count < 2:
                    print("Rejected.")
                    return alloc_omit_list
                else:
                    death_move = False

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
                self.grad_diff,
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
                self.grad_diff,
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
                self.grad_diff,
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
                self.grad_diff,
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
                return alloc_omit_list
            pvec = self.pvec_list_all[pvec_idx]
            ### This is the type we want to duplicate from
            valids_empty = np.where(pvec.force_group_histogram == 0)[0]
            N_all   = pvec.force_group_count
            N_empty = valids_empty.size
            if N_empty == 0:
                print("Rejected.")
                return alloc_omit_list
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
