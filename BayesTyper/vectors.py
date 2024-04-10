#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

from collections import OrderedDict

import numpy as np
from scipy.special import gamma as gamma_function
import copy
import math

from .tools import (is_type_or_raise,
                    is_one_of_types_or_raise,
                    )

from scipy.stats import multivariate_normal

import networkx as nx
from networkx.algorithms import isomorphism

from .parameters import ParameterManager

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================
from .constants import (
    _UNIT_QUANTITY,
    _SEED,
    _INACTIVE_GROUP_IDX
    )

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================


def get_allocation_vector(parameter_manager):

    __doc__ = """
    Generate an allocation vector of type `BaseVector` based
    on the current state of parameter manager `parameter_manager`.
    """

    allocations  = BaseVector(dtype=int)
    unique_ranks = np.unique(parameter_manager.force_ranks)
    allocations.append(
        np.zeros(
            unique_ranks.size, dtype=int
            )
        )

    force_group_idxs = parameter_manager.force_group_idx_list

    for idx, force_group_idx in enumerate(force_group_idxs):
        rank = parameter_manager.force_ranks[idx]
        allocations[rank] = force_group_idx

    return allocations


class BaseVector(object):

    def SelfReturner(func):

        def Returner(self, *args):
            func(self, *args)
            return self

        return Returner

    def VectorUpdater(func):

        def Updater(self, *args):
            func(self, *args)
            p = self._vector_values * self._vector_units
            self._vector = p.astype(self._dtype_np)

        return Updater

    def __init__(self, dtype=np.float64):

        if dtype in [float, np.float64]:
            dtype = [np.float64, float]
        elif dtype in [int, np.int64]:
            dtype = [np.int64, int]
        elif dtype in [bool, np.bool_]:
            dtype = [np.bool, bool]
        else:
            raise ValueError(f"dtype {dtype} not understood.")

        ### Python data type
        self._dtype_py      = dtype[0]
        ### Analog Numpy data type
        self._dtype_np      = dtype[1]
        ### holds only the *values* of the vector elements
        self._vector_values = np.array([],dtype=self._dtype_np)
        ### holds only the *units* of the vector elements
        self._vector_units  = np.array([],dtype=self._dtype_np)
        ### Holds both, *units and values* of the vector elements
        self._vector        = np.array([],dtype=self._dtype_np)
        self.call_back      = None

    @property
    def vector_units(self):
        return self._vector_units

    @property
    def vector_values(self):
        return self._vector_values

    @property
    def vector(self):
        return self._vector

    @vector_values.setter
    @VectorUpdater
    def vector_values(self, value):
        self._vector_values[:] = np.copy(value)

    @VectorUpdater
    def reset(self, base_vector):

        self._vector_values = copy.deepcopy(
            base_vector._vector_values
            )
        self._vector_units = copy.deepcopy(
            base_vector._vector_units
            )

    @VectorUpdater
    def append(self, value):
        if isinstance(value, list):
            value = np.array(value)

        if is_one_of_types_or_raise(
            "value",
             value,
              [_UNIT_QUANTITY, np.ndarray, self._dtype_np, self._dtype_py]
              ):
            if isinstance(value, np.ndarray):
                if not (value.dtype.type == self._dtype_np or\
                       value.dtype.type == self._dtype_py):
                    raise ValueError(
                        f"np.ndarray must be of type {self._dtype_np} or {self._dtype_py} but is {value.dtype}.")
            if hasattr(value, "size"):
                Ndim = value.size
            else:
                Ndim = 1

            ### First, make sure we have the right unit
            if isinstance(value,_UNIT_QUANTITY):
                _vector_units = np.append(
                    self._vector_units,  
                    [value.unit for _ in range(Ndim)],
                    )
                _vector_values = np.append(
                    self._vector_values, 
                    value._value
                    )
            elif isinstance(value, tuple([self._dtype_py, self._dtype_np, np.ndarray])):
                _vector_units = np.append(
                    self._vector_units, 
                    [1. for _ in range(Ndim)],
                    )
                _vector_values = np.append(
                    self._vector_values, 
                    value
                    )
            else:
                raise ValueError(f"Type of value {type(value)} is not understood.")

            self._vector_values = _vector_values
            self._vector_units  = _vector_units

    @VectorUpdater
    def remove(self, index):

        _vector_values = copy.copy(self._vector_values[:-1])
        _vector_units  = copy.copy(self._vector_units[:-1])

        if index == -1:
            index = self.size

        if index == 0:
            _vector_values[:] = self._vector_values[1:]
            _vector_units[:]  = self._vector_units[1:]
        elif index == self.size:
            _vector_values[:] = self._vector_values[:-1]
            _vector_units[:]  = self._vector_units[:-1]
        else:
            _vector_values[:index] = self._vector_values[:index]
            _vector_units[:index]  = self._vector_units[:index]

            _vector_values[index:] = self._vector_values[index+1:]
            _vector_units[index:]  = self._vector_units[index+1:]

        self._vector_values = _vector_values
        self._vector_units  = _vector_units

    @property
    def log_jacobian(self):
        ### log(1.)
        return 0.

    @property
    def L2norm(self):
        return np.sum(self._vector_values**2)

    @property
    def size(self):
        return self._vector_values.size

    @VectorUpdater
    def apply_changes(self):
        if self.call_back == None:
            pass
        else:
            self.call_back()

    def index(
        self, 
        key_list: list) -> list:

        __doc__ = """
        Retrieve indices of values in key_list. Returns list
        of length `len(key_list)`. Each element `i` in this list,
        is a list holding the indices of `i` in self.vector.
        """
        index_list = list()
        for key in key_list:
            index_list.append(np.where(self.vector_values == key)[0])
        return index_list

    def __str__(self):
        return str(self._vector_values)

    def __repr__(self):
        return str(self._vector_values)

    def __getitem__(self, key):
        return copy.copy(self._vector_values[key])

    @VectorUpdater
    def __setitem__(self, key, value):
        self._vector_values[key] = copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __add__(self, value):
        self._vector_values += copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __sub__(self, value):
        self._vector_values -= copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __mul__(self, value):
        self._vector_values *= copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __div__(self, value):
        self._vector_values /= copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __truediv__(self, value):
        self._vector_values /= copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __div__(self, value):
        self._vector_values /= copy.copy(value)


class ParameterVectorLinearTransformation(BaseVector):

    def SelfReturner(func):

        def Returner(self, *args):
            func(self, *args)
            return self

        return Returner

    def VectorUpdater(func):

        def Updater(self, *args):
            func(self, *args)
            self._vector_k  = self._vector_0.vector_values * self._vector_units
            self._vector_k += self._scaling_vector.vector_values * self._vector_values * self._vector_units

            self._vector_k_vec[:]  = self._vector_0.vector_values
            self._vector_k_vec[:] += self._scaling_vector.vector_values * self._vector_values

        return Updater

    def __init__(self):

        super().__init__(dtype=float)

        self._vector_0       = BaseVector(dtype=float)
        self._vector_k       = np.ndarray([], dtype=float)
        self._vector_k_vec   = BaseVector(dtype=float)
        self._scaling_vector = BaseVector(dtype=float)
        self.call_back       = None

    @property
    def vector_k_vec(self):
        return self._vector_k_vec

    @property
    def vector_values(self):
        return self._vector_values

    @property
    def vector(self):
        return self._vector_k

    @property
    def vector_0(self):
        return self._vector_0

    @property
    def vector_units(self):
        return self._vector_units

    @property
    def scaling_vector(self):
        return self._scaling_vector

    @property
    def vector_k(self):
        return self._vector_k

    @vector_0.setter
    @VectorUpdater
    def vector_0(self, value):

        self._vector_0.vector_values[:] = copy.deepcopy(value[:])
        k  = self.vector_k[:]
        v0 = self._vector_0.vector_values * self._vector_units
        s  = self._scaling_vector.vector_values * self._vector_units
        self._vector_values[:] = (k-v0)/s
    
    @VectorUpdater
    def apply_changes(self):
        if self.call_back == None:
            pass
        else:
            self.call_back()
    
    @VectorUpdater
    def reset(self, vector):

        self._vector_values = copy.deepcopy(
            vector._vector_values
            )
        self._vector_units = copy.deepcopy(
            vector._vector_units
            )
        self._vector_0 = copy.deepcopy(
            vector._vector_0
            )
        self._vector_k_vec = copy.deepcopy(
            vector._vector_k_vec
            )
        self._scaling_vector = copy.deepcopy(
            vector._scaling_vector
            )

    @VectorUpdater
    def append(
        self, 
        value, 
        scaling = None,
        value_0 = None):

        if is_one_of_types_or_raise("value", value, [_UNIT_QUANTITY, float, int, np.ndarray]):
            if hasattr(value, "size"):
                Ndim = value.size
            else:
                Ndim = 1

        _vector_values = np.append(
            self._vector_values,
            [0. for _ in range(Ndim)],
            )

        if value_0 != None:
            if type(value) == _UNIT_QUANTITY and\
               type(value_0) != _UNIT_QUANTITY:
                raise ValueError(
                    f"value_0 must be type {_UNIT_QUANTITY} but is {type(value_0)}")

        ### First, make sure we have the right unit
        if type(value) == _UNIT_QUANTITY:
            _vector_units = np.append(
                self._vector_units,
                [value.unit for _ in range(Ndim)],
                )
            if value_0 == None:
                self._vector_0.append(value._value)
                self._vector_k_vec.append(value._value)
            else:
                self._vector_0.append(
                    value_0.value_in_unit(
                        value.unit
                        )
                    )
                self._vector_k_vec.append(
                    value_0.value_in_unit(
                        value.unit
                        )
                    )

        elif type(value) in [float, int, np.ndarray]:
            _vector_units = np.append(
                self._vector_units, 
                [1. for _ in range(Ndim)],
                )
            if value_0 == None:
                self._vector_0.append(value)
                self._vector_k_vec.append(value)
            else:
                self._vector_0.append(value_0)
                self._vector_k_vec.append(value_0)
        else:
            raise ValueError(f"Type of value {type(value)} is not understood.")

        ### Second, take care of the scaling
        if scaling == None:
            if type(value) == _UNIT_QUANTITY:
                _value = copy.copy(value._value)
            else:
                _value = copy.copy(value)
            _abs_value = abs(_value)
            
            if type(_abs_value) in [list, np.ndarray]:
                _abs_value = np.array(_abs_value)
                scaling    = np.ones_like(_abs_value)
                zeros      = np.where(_abs_value == 0.)
                nonzeros   = np.where(_abs_value > 0.)
                power      = np.floor(
                    np.log10(
                        _abs_value[nonzeros]
                        )
                    )
                power     += 1
                scaling[zeros]    = 1.
                scaling[nonzeros] = 10.**power.astype(int)
                self._scaling_vector.append(scaling)
            else:
                self._scaling_vector.append(
                    [1. for _ in range(Ndim)],
                    )
        elif type(scaling) in [int, float]:
            scaling = float(scaling)
            self._scaling_vector.append(
                [scaling for _ in range(Ndim)],
                )
        elif type(scaling) in [list, np.ndarray]:
            self._scaling_vector.append(scaling)
        else:
            is_one_of_types_or_raise(
                "scaling", 
                scaling, 
                [list, float, int, np.ndarray]
                )
            self._scaling_vector.append(scaling)

        self._vector_units  = _vector_units
        self._vector_values = _vector_values

    @VectorUpdater
    def remove(self, index):

        _vector_values  = copy.copy(self._vector_values[:-1])
        _vector_units   = copy.copy(self._vector_units[:-1])

        if index == -1:
            index = self.size

        self._vector_0.remove(index)
        self._vector_k_vec.remove(index)
        self._scaling_vector.remove(index)

        if index == 0:
            _vector_values[:] = self._vector_values[1:]
            _vector_units[:]  = self._vector_units[1:]
        elif index == self.size:
            _vector_values[:] = self._vector_values[:-1]
            _vector_units[:]  = self._vector_units[:-1]
        else:
            _vector_values[:index] = self._vector_values[:index]
            _vector_units[:index]  = self._vector_units[:index]

            _vector_values[index:] = self._vector_values[index+1:]
            _vector_units[index:]  = self._vector_units[index+1:]

        self._vector_values  = _vector_values
        self._vector_units   = _vector_units

    def get_real(self, parameter):
        real  = self.vector_0.vector_values * self.vector_units
        real += self.scaling_vector.vector_values * parameter * self.vector_units
        return real

    def get_transform(self, real_parameter):

        trans  = real_parameter - self.vector_0.vector_values * self.vector_units
        trans /= (self.scaling_vector.vector_values * self.vector_units)
        return trans

    @property
    def jacobian(self):

        __doc__ = """
            Get the Jacobian matrix as vector.
            See log_det_jacobian for more info.
        """

        return self._scaling_vector.vector_values

    @property
    def log_det_jacobian(self):

        __doc__ = """

        Jacobian log determinant for the transformation:

        g(p^) = f(v(p^)) J
        p     = v(p^)

        here: v(p^) = p^ * scaling + p0

        p0: original vector
        p^: transformed vector
        v : transformation operation (this class)

        J : Jacobian:
                        | d v  |
                        | ---- |
                        | d p^ |

            here:

            for i==j:
            J_ij = scaling

            for i!=j:
            J_ij = 0

        """

        log_J  = np.log(self._scaling_vector.vector_values)
        return np.sum(log_J)

    @property
    def L2norm(self):
        return np.sum(self._vector_values.vector_values**2)

    @property
    def size(self):
        return self._vector_values.size

    def __str__(self):
        return str(self._vector_values)

    def __repr__(self):
        return str(self._vector_values)

    def __getitem__(self, key):
        return copy.copy(self._vector_values[key])

    @VectorUpdater
    def __setitem__(self, key, value):
        self._vector_values[key] = copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __add__(self, value):
        self._vector_values += copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __sub__(self, value):
        self._vector_values -= copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __mul__(self, value):
        self._vector_values *= copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __truediv__(self, value):
        self._vector_values /= copy.copy(value)

    @SelfReturner
    @VectorUpdater
    def __neg__(self, value):
        self._vector_values = -copy.copy(value)


class ForceFieldParameterVector(ParameterVectorLinearTransformation):

    def __init__(
        self, 
        parameter_manager, 
        parameter_name_list = None, 
        scaling_list = None, 
        exclude_others = True, 
        ):

        super().__init__()

        if parameter_name_list == None:
            parameter_name_list = list(parameter_manager.default_parms.keys())

        if scaling_list != None:
            if len(scaling_list) != len(parameter_name_list):
                raise ValueError(
                    f"scaling_list and parameter_name_list must be of equal length, but is {len(scaling_list)} and {len(parameter_name_list)}.")
        else:
            scaling_list = [None for _ in parameter_name_list]

        for parameter_name in parameter_name_list:
            if not parameter_name in parameter_manager.parm_names:
                raise ValueError(
                    f"Parameter with name {parameter_name} not found in parameter_manager {parameter_manager.parm_names}."
                    )

        self.parameter_manager   = parameter_manager
        self.parameter_name_list = parameter_name_list
        self.scaling_list        = scaling_list
        self.exclude_list        = list()

        ### First, build the allocation list and load all parameter values and their
        ### scaling factors into the object.
        ### ========================================================================
        ###
        ### Note:
        ### -----
        ### self.allocations holds the force_group index of each *unique* force rank.
        ### For instance:
        ### parameter_manager.force_group_idx_list : [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        ### parameter_manager.force_ranks          : [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9]
        ### self.allocations                       : [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        self.allocations = get_allocation_vector(self.parameter_manager)

        for force_group_idx in range(self.force_group_count):
            for parameter_name, scaling in zip(
                self.parameter_name_list, 
                self.scaling_list):
                value = self.parameter_manager.get_parameter(
                    force_group_idx, 
                    parameter_name)
                super().append(value, scaling)
        
        # Do we really need this???
        #self.forcecontainer_system_idx = list()
        #for rank in range(self.allocations.size):
        #    valids = np.where(self.parameter_manager.force_ranks == rank)[0]
        #    idx    = valids[0]
        #    system_idx = self.parameter_manager.system_idx_list[idx]
        #    self.forcecontainer_system_idx.append(system_idx)
        #self.forcecontainer_system_idx = np.array(self.forcecontainer_system_idx)

        ### Second, build parameter exclusion list
        ### ======================================
        ### This may be useful for cases where one
        ### wants to, for instance, sample k and 
        ### bondlength seperatetly.
        if exclude_others:
            for parameter_name in self.parameter_manager.parm_names:
                if parameter_name not in self.parameter_name_list:
                    self.exclude_list.append(parameter_name)

        ### Make sure everything is flushed down to the 
        ### parameter manager.
        self.apply_changes()

    @property
    def parameters_per_force_group(self):

        __doc__ = """
        Number of parameters that are affected for each
        force group within this parameter vector.
        """

        return len(self.parameter_name_list)

    @property
    def force_group_histogram(self):

        __doc__ = """
        Return a "histogram" of the allocation vector.
        This computes the number of allocations for each
        force group.
        """

        hist = np.zeros(
            self.force_group_count, dtype=int
            )
        for i in range(self.force_group_count):
            alloc_pos = self.allocations.index([i])[0]
            hist[i]   = alloc_pos.size

        return hist

    @property
    def force_group_count(self):

        return self.parameter_manager.force_group_count

    @property
    def forcecontainer_list(self):

        return self.parameter_manager.forcecontainer_list

    @property
    def force_group_size(self):

        values, counts = np.unique(
            self.parameter_manager.force_group_idx_list,
            return_counts=True
            )
        return counts

    @property
    def N_systems(self):

        return self.parameter_manager.N_systems

    
    def swap_types(self, source_index, target_index):

        if source_index == -1:
            source_index = self.force_group_count - 1
        if target_index == -1:
            target_index = self.force_group_count - 1

        if source_index > self.force_group_count-1 or source_index < 0:
            raise ValueError("source_index must be >0 and <= force_group_count")
        if target_index > self.force_group_count-1 or target_index < 0:
            raise ValueError("target_index must be >0 and <= force_group_count")

        if source_index == target_index:
            return

        start = source_index * self.parameters_per_force_group
        stop  = start + self.parameters_per_force_group
        source_values_0 = self.vector_0[start:stop]
        source_mapped = self[start:stop]

        start = target_index * self.parameters_per_force_group
        stop  = start + self.parameters_per_force_group
        target_values_0 = self.vector_0[start:stop]
        target_mapped = self[start:stop]

        self.set_parameters_by_force_group(
            source_index,
            target_mapped,
            target_values_0
            )

        self.set_parameters_by_force_group(
            target_index,
            source_mapped,
            source_values_0
            )


    def copy(self, include_systems=False, rebuild_to_old_systems=False):

        import copy

        self_cp = copy.deepcopy(self)

        if not include_systems:
            if hasattr(self_cp.parameter_manager, "system_list"):
                del self_cp.parameter_manager.system_list
        else:
            if rebuild_to_old_systems:
                self_cp.rebuild_from_systems(
                    lazy = True,
                    system_list = self.parameter_manager.system_list
                    )
            else:
                self_cp.rebuild_from_systems(
                    lazy = True,
                    system_list = self_cp.parameter_manager.system_list
                    )

        return self_cp


    def remove_system(self, system_idx_list):

        __doc__ = """
        Remove system via idx. We just remove the entries from the
        allocation vector and then call `ParameterManager.remove_system`
        to remove the systems from the parameter manager object underneath.
        """

        to_delete = list()
        for system_idx in np.unique(system_idx_list):
            if isinstance(system_idx, str):
                system_name = system_idx
                system_idx  = None
                for _system_idx in range(self.N_systems):
                    system = self.parameter_manager.system_list[_system_idx]
                    if system.name == system_name:
                        system_idx = _system_idx
                        break
                if system_idx == None:
                    continue
            if system_idx < 0:
                system_idx = self.N_systems+system_idx
            idx_list = np.where(
                    self.parameter_manager.system_idx_list == system_idx
                    )[0]
            if idx_list.size == 0:
                continue
            _idx_list = np.unique(
                self.parameter_manager.force_ranks[idx_list]
                )
            to_delete.extend(_idx_list.tolist())
        to_delete = np.unique(to_delete).tolist()
        to_delete = sorted(to_delete, reverse=True)
        for idx in to_delete:
            self.allocations.remove(idx)

        self.parameter_manager.remove_system(system_idx_list)


    def rebuild_from_systems(self, lazy=False, system_list=None):
        
        import copy

        if system_list == None:
            system_list_cp = copy.deepcopy(self.parameter_manager.system_list)
        else:
            system_list_cp = system_list
        self.parameter_manager.rebuild_from_systems(
            system_list_cp,
            lazy=lazy
            )

    def reset(self, parameters, allocations=None):

        import copy

        if isinstance(allocations, type(None)):
            allocations = parameters.allocations

        assert (parameters.size-1) >= max(allocations)
        assert allocations.size == self.allocations.size

        ### This should work for both numpy and BaseVector
        ### arrays.
        self.allocations[:] = copy.deepcopy(allocations[:])

        ### We do a "try ... except ..." part here, since
        ### a parameter vector might have no parameter_manager.
        ### In that case, we can still use `self.reset`, but it is
        ### less safe.
        try:
            ### Check if we must create new force groups first
            diff = parameters.force_group_count - self.force_group_count
            if diff > 0:
                value_list = parameters.get_parameters_by_force_group(
                    0, 
                    get_all_parms=False
                    )
                for _ in range(diff):
                    ### This will update vector_k
                    self.add_force_group(
                        copy.deepcopy(
                            value_list
                            )
                        )
                    ### Note: `self.duplicate(0)` only works if there
                    ### is already a force group present.
                    ### Therefore we must use `self.add_force_group`.
        except:
            import warnings
            warnings.warn(
                "Attempting to reset parameter vector without parameter manager."
                )
            ### Check if we must create new force groups first
            diff = int(len(parameters.vector_k)/parameters.parameters_per_force_group) - self.force_group_count
            if diff > 0:
                value_list = parameters.vector_k[:parameters.parameters_per_force_group]
                for _ in range(diff):
                    ### This will update vector_k
                    self.add_force_group(value_list)
                    ### Note: `self.duplicate(0)` only works if there
                    ### is already a force group present.

        ### Make sure each force object is in the right forcegroup
        for k, z in enumerate(allocations):
            self.parameter_manager.relocate_by_rank(
                k, ### force_rank
                z, ### force_group_idx
                self.exclude_list
                )

        ### Check if we have more force groups than we
        ### should have.
        if diff < 0:
            diff = -1 * diff
            for _ in range(diff):
                ### This will update vector_k
                self.remove(-1)

        ### Set the parameter values
        vector_idx = 0
        for force_group_idx in range(self.force_group_count):
            for parameter_name in self.parameter_name_list:
                ### This will update vector_k
                self._vector_0[vector_idx] = copy.deepcopy(
                    parameters._vector_0[vector_idx]
                    )
                self._vector_k_vec[vector_idx] = copy.deepcopy(
                    parameters._vector_k_vec[vector_idx]
                    )
                self._scaling_vector[vector_idx] = copy.deepcopy(
                    parameters._scaling_vector[vector_idx]
                    )
                self[vector_idx] = copy.deepcopy(
                    parameters._vector_values[vector_idx]
                    )
                vector_idx += 1

        self.apply_changes()

    def apply_changes(self):

        ### Some sanity checking before we proceed.
        if self.allocations.size > 0:
            actual_max      = max(self.allocations)
            theoretical_max = self.force_group_count-1
            if actual_max == _INACTIVE_GROUP_IDX:
                actual_max = 0
            if self.force_group_count == 0:
                theoretical_max = 0
            if actual_max > theoretical_max:
                raise ValueError(
                    f"{self.force_group_count} force groups, but maximum in allocation vector is {actual_max}."
                    )

        ### First, make sure each force is in the right forcegroup
        for k, z in enumerate(self.allocations):
            self.parameter_manager.relocate_by_rank(
                k, z, self.exclude_list, update_forces=False)

        ### Second, set the parameters
        batch_mode = self.parameter_manager.N_systems > 50
        if batch_mode:
            force_group_idx_list = list()
            name_list = list()
            value_list = list()

        vector_idx = 0
        for force_group_idx in range(self.force_group_count):
            if batch_mode:
                force_group_idx_list.append(force_group_idx)
                name_list.append([])
                value_list.append([])
            for parameter_name in self.parameter_name_list:
                if batch_mode:
                    name_list[-1].append(parameter_name)
                    value_list[-1].append(self.vector_k[vector_idx])
                else:
                    self.parameter_manager.set_parameter(
                            force_group_idx, parameter_name, 
                            self.vector_k[vector_idx])
                vector_idx += 1
        if batch_mode:
            self.parameter_manager.set_parameter_batch(
                    force_group_idx_list,
                    name_list,
                    value_list)

        super().apply_changes()

    def append(self, value_list):

        raise NotImplementedError(
            "Cannot use append with ForceFieldParameterVector. Use `duplicate` instead.")

    def add_force_group(self, value_list):

        import copy

        if len(self.parameter_name_list) != len(value_list):
            raise ValueError(
                f"`value_list` must have length {len(self.parameter_name_list)}, but has {len(value_list)}.")

        parms    = dict()
        parm_idx = 0
        for parameter_name in self.parameter_name_list:
            parms[parameter_name] = copy.deepcopy(value_list[parm_idx])
            parm_idx += 1

        self.parameter_manager.add_force_group(parms)

        force_group_idx = self.force_group_count-1
        vector_idx      = force_group_idx * self.parameters_per_force_group
        for parameter_name, scaling, value in zip(
            self.parameter_name_list, 
            self.scaling_list,
            value_list):
            ### This will update vector_k
            super().append(
                copy.deepcopy(value),
                copy.deepcopy(scaling)
                )
            self._vector_values[-1]  = copy.deepcopy(self._vector_values[vector_idx])
            self._vector_0[-1]       = copy.deepcopy(self._vector_0[vector_idx])
            self._vector_k_vec[-1]   = copy.deepcopy(self._vector_k_vec[vector_idx])
            self._scaling_vector[-1] = copy.deepcopy(self._scaling_vector[vector_idx])
            self.parameter_manager.set_parameter(
                self.force_group_count-1,
                parameter_name,
                self.vector_k[-1]
                )
            vector_idx += 1

        return self.force_group_count


    def duplicate(self, force_group_idx):

        import copy

        if force_group_idx == -1:
            force_group_idx = self.force_group_count - 1

        v0 = self._vector_0.vector_values * self._vector_units
        parm_1 = force_group_idx * self.parameters_per_force_group
        parm_2 = parm_1 + self.parameters_per_force_group
        self.add_force_group(
            copy.deepcopy(v0[parm_1:parm_2]))

        force_group_idx = self.force_group_count - 1
        _parm_1 = force_group_idx * self.parameters_per_force_group
        _parm_2 = _parm_1 + self.parameters_per_force_group
        self[_parm_1:_parm_2] = self[parm_1:parm_2]


    def remove(self, force_group_idx):

        if force_group_idx == -1:
            force_group_idx = self.force_group_count - 1

        force_group_idx = int(force_group_idx)
        ### This will throw and Exception if the force group is not empty.
        ### So we don't need to check for force group emptiness here.
        self.parameter_manager.remove_force_group(force_group_idx)
        
        ### Change the allocation values
        valids_gt = np.where(
            self.allocations.vector_values > force_group_idx
            )[0]
        self.allocations[valids_gt] = self.allocations[valids_gt] - 1

        ### Make sure the FF parameter values are reordered properly
        remove_idx = force_group_idx * self.parameters_per_force_group
        for _ in range(self.parameters_per_force_group):
            ### This will update vector_k
            super().remove(remove_idx)


    def is_force_group_empty(self, force_group_idx):

        is_empty = self.parameter_manager.is_force_group_empty(force_group_idx)
        return is_empty


    def set_parameters_by_force_group(
        self, 
        force_group_idx,
        values,
        values_0 = None,
        ):

        N_parms    = self.parameters_per_force_group
        vector_idx = force_group_idx * N_parms
        if isinstance(values, float) or isinstance(values, int):
            if N_parms > 1:
                raise ValueError(
                    f"values must be list, not {type(values)}")
            values = [float(values)]
        else:
            if isinstance(values, list) or isinstance(values, np.ndarray):
                if N_parms == 0:
                    raise ValueError(
                        f"Value must be float, not {type(values)}")

        assert len(values) == N_parms

        if isinstance(values_0, type(None)):
            values_0 = self._vector_0[vector_idx:vector_idx+N_parms]
        elif isinstance(values_0, float) or isinstance(values_0, int):
            if N_parms > 1:
                raise ValueError(
                    f"values_0 must be list, not {type(values_0)}")
            values_0 = [float(values_0)]

        assert len(values_0) == N_parms

        name_list = list()
        value_list = list()
        for parm_idx in range(N_parms):
            v0 = values_0[parm_idx]
            if isinstance(v0, _UNIT_QUANTITY):
                unit = self.vector_units[vector_idx + parm_idx]
                v0   = v0.value_in_unit(unit)
            ### We must first update vector_0
            ### This will not update vector_k
            self._vector_0[vector_idx + parm_idx]     = v0
            self._vector_k_vec[vector_idx + parm_idx] = values[parm_idx]
            ### This will update vector_k
            self[vector_idx + parm_idx] = values[parm_idx]
            name_list.append(self.parameter_name_list[parm_idx])
            value_list.append(self.vector_k[vector_idx + parm_idx])
        self.parameter_manager.set_parameter_batch(
                [force_group_idx],
                name_list,
                value_list)

    def get_parameters_by_force_group(
        self, 
        force_group_idx, 
        get_all_parms = True):

        if get_all_parms:
            parm_name_list = self.parameter_manager.parm_names
        else:
            parm_name_list = self.parameter_name_list

        value_list     = list()
        for parameter_name in parm_name_list:
            value = self.parameter_manager.get_parameter(
                force_group_idx,
                parameter_name
                )
            value_list.append(copy.deepcopy(value))

        return value_list


class SmartsForceFieldParameterVector(ForceFieldParameterVector):

    def __init__(
        self,
        parameter_manager, 
        init_smarts_manager_list,
        parameter_name_list = None, 
        scaling_list = None, 
        exclude_others = True,
        remove_types = True,
        set_inactive = False
    ):
        from .tools import _remove_types

        if remove_types:
            _remove_types(
                parameter_manager,
                set_inactive = set_inactive
            )

        import copy
        self.smarts_manager_list = copy.deepcopy(init_smarts_manager_list)
        self.smarts_allocations  = BaseVector(dtype=int)
        self.smarts_manager_allocations = BaseVector(dtype=int)
        self.smarts_manager_allocations_mapping = BaseVector(dtype=int)

        idx = 0
        for smarts_manager in init_smarts_manager_list:
            counts = 0
            for a in smarts_manager.allocations:
                self.smarts_allocations.classappend(a)
                self.smarts_manager_allocations.append(idx)
                self.smarts_manager_allocations_mapping.append(counts)
                counts += 1
            idx += 1

        super().__init__(
            parameter_manager = parameter_manager, 
            parameter_name_list = parameter_name_list, 
            scaling_list = scaling_list, 
            exclude_others = exclude_others,
        )

        if not remove_types:
            assert len(init_smarts_manager_list) == self.force_group_count

        if remove_types:
            for idx in range(1, len(init_smarts_manager_list)):
                value_list = self.get_parameters_by_force_group(
                    0, 
                    get_all_parms=False
                    )
                super().add_force_group(value_list)

        self.apply_changes(True, True)


    def swap_smarts(self, source_index, target_index):

        self.swap_types(source_index, target_index)


    def swap_types(self, source_index, target_index):

        import copy
        import numpy as np

        if source_index == -1:
            source_index = self.force_group_count - 1
        if target_index == -1:
            target_index = self.force_group_count - 1

        if source_index > self.force_group_count-1 or source_index < 0:
            raise ValueError("source_index must be >0 and <= force_group_count")
        if target_index > self.force_group_count-1 or target_index < 0:
            raise ValueError("target_index must be >0 and <= force_group_count")

        if source_index == target_index:
            return

        source_indices = self.get_smarts_indices(source_index)
        target_indices = self.get_smarts_indices(target_index)

        source_manager = copy.deepcopy(self.smarts_manager_list[source_index])
        target_manager = copy.deepcopy(self.smarts_manager_list[target_index])

        source_smarts_allocations = copy.deepcopy(self.smarts_allocations[source_indices])
        target_smarts_allocations = copy.deepcopy(self.smarts_allocations[target_indices])

        source_smarts_manager_allocations = copy.deepcopy(self.smarts_manager_allocations[source_indices])
        target_smarts_manager_allocations = copy.deepcopy(self.smarts_manager_allocations[target_indices])

        source_smarts_manager_allocations_mapping = copy.deepcopy(self.smarts_manager_allocations_mapping[source_indices])
        target_smarts_manager_allocations_mapping = copy.deepcopy(self.smarts_manager_allocations_mapping[target_indices])

        self.smarts_manager_list[source_index] = target_manager
        self.smarts_manager_list[target_index] = source_manager

        self.smarts_allocations[source_indices] = target_smarts_allocations
        self.smarts_allocations[target_indices] = source_smarts_allocations

        self.smarts_manager_allocations_mapping[source_indices] = target_smarts_manager_allocations_mapping
        self.smarts_manager_allocations_mapping[target_indices] = source_smarts_manager_allocations_mapping

        super().swap_types(source_index, target_index)


    def set_inactive(self, force_group_idx):

        indices = self.get_smarts_indices(force_group_idx)
        for i in indices:
            self.set_allocation(i, _INACTIVE_GROUP_IDX)


    def get_smarts_indices(self, force_group_idx):

        import numpy as np

        if isinstance(force_group_idx, int):
            indices = self.smarts_manager_allocations.index([force_group_idx])[0]
        elif isinstance(force_group_idx, np.int):
            indices = self.smarts_manager_allocations.index([force_group_idx])[0]
        elif isinstance(force_group_idx, np.int64):
            indices = self.smarts_manager_allocations.index([force_group_idx])[0]
        elif isinstance(force_group_idx, list):
            indices = self.smarts_manager_allocations.index(force_group_idx)
        elif isinstance(force_group_idx, np.ndarray):
            indices = self.smarts_manager_allocations.index(force_group_idx.tolist())
        else:
            raise ValueError(
                f"Datatype {type(force_group_idx)} not understood."
                )

        return indices


    def reset(self, parameters, allocations=None, ignore_values=False):

        import copy

        ### We want to keep the allocations from the `self` instance
        ### and just update with whatever values and smarts definitions
        ### are stores in the arguments.
        allocations = copy.deepcopy(self.allocations)

        ### Check if we must create new force groups first
        diff = parameters.force_group_count - self.force_group_count
        if diff > 0:
            value_list = parameters.get_parameters_by_force_group(
                0, 
                get_all_parms=False
                )
            for _ in range(diff):
                ### This is just a dummy
                smarts_manager = copy.deepcopy(
                    parameters.smarts_manager_list[0]
                    )
                self.add_force_group(
                    copy.deepcopy(value_list), 
                    smarts_manager
                    )

        if ignore_values:
            super().reset(self.copy(), allocations)
        else:
            super().reset(parameters, allocations)

        self.smarts_manager_list                = copy.deepcopy(
            parameters.smarts_manager_list
            )
        self.smarts_allocations                 = copy.deepcopy(
            parameters.smarts_allocations
            )
        self.smarts_manager_allocations         = copy.deepcopy(
            parameters.smarts_manager_allocations
            )
        self.smarts_manager_allocations_mapping = copy.deepcopy(
            parameters.smarts_manager_allocations_mapping
            )

        self.apply_changes()


    def add_force_group(self, value_list, smarts_manager):

        import copy

        self.smarts_manager_list.append(
            smarts_manager
        )

        N = self.force_group_count
        counts = 0
        for a in smarts_manager.allocations:
            self.smarts_allocations.append(a)
            self.smarts_manager_allocations.append(N)
            self.smarts_manager_allocations_mapping.append(counts)
            counts += 1

        super().add_force_group(value_list)


    def duplicate(self, force_group_idx, smarts_manager=None):

        import copy

        if force_group_idx == -1:
            force_group_idx = self.force_group_count - 1

        value_list = self.get_parameters_by_force_group(
            force_group_idx, 
            get_all_parms=False
            )
        if smarts_manager == None:
            smarts_manager = copy.deepcopy(
                self.smarts_manager_list[force_group_idx]
            )

        self.add_force_group(
            copy.deepcopy(value_list), 
            smarts_manager
            )


    def remove(self, force_group_idx):

        ### The regular ForcefieldParameterVector won't
        ### let you remove a forcegroup that is still
        ### populated. However, with a smarts-based
        ### vector (as this class), we can actually do
        ### that.
        ### Therefore, first depopulate the allocations
        ### of that force group. It will be re-allocated
        ### ones the apply_changes is called.

        from BayesTyper.constants import _INACTIVE_GROUP_IDX

        if force_group_idx == -1:
            force_group_idx = self.force_group_count - 1

        index_list = self.allocations.index([force_group_idx])[0]
        self.allocations[index_list] = _INACTIVE_GROUP_IDX
        super().apply_changes()
        super().remove(force_group_idx)

        del self.smarts_manager_list[force_group_idx]
        to_delete = self.smarts_manager_allocations.index([force_group_idx])[0].tolist()
        to_delete = sorted(to_delete, reverse=True)
        for i in to_delete:
            self.smarts_allocations.remove(i)
            self.smarts_manager_allocations.remove(i)
            self.smarts_manager_allocations_mapping.remove(i)

        ### Finally make sure to account for the minus 1
        ### in number of smarts managers
        valids_gt = np.where(
            self.smarts_manager_allocations.vector_values > force_group_idx
            )[0]
        self.smarts_manager_allocations[valids_gt] = self.smarts_manager_allocations[valids_gt] - 1


    def apply_changes(self, flush_to_systems=True, flush_to_allocations=False):

        if flush_to_allocations:

            import numpy as np
            from rdkit import Chem

            force_group_idx_list = np.array(self.parameter_manager.force_group_idx_list)
            sys_idx_list  = np.array(self.parameter_manager.system_idx_list)
            all_atom_list = np.array(self.parameter_manager.atom_list)
            force_ranks   = np.array(self.parameter_manager.force_ranks)

            smarts_map_dict = dict()
            for force_group_idx in range(self.force_group_count):
                smarts_manager = self.smarts_manager_list[force_group_idx]
                smarts = smarts_manager.get_smarts()
                rdmol  = Chem.MolFromSmarts(smarts)
                map_dict = dict()
                for atom in rdmol.GetAtoms():
                    prop_dict = atom.GetPropsAsDict()
                    if "molAtomMapNumber" in prop_dict:
                        tag = prop_dict["molAtomMapNumber"]
                        map_dict[atom.GetIdx()] = tag
                map_dict = sorted(map_dict, key = lambda x: map_dict[x])
                smarts_map_dict[force_group_idx] = rdmol, map_dict

            for sys_idx in range(self.parameter_manager.N_systems):
                valids = np.where(sys_idx_list == sys_idx)[0]
                atom_dict = dict()
                for v in valids:
                    a = all_atom_list[v].tolist()
                    a = tuple(a)
                    r = force_ranks[v]
                    atom_dict[a] = r
                    atom_dict[a[::-1]] = r

                rdmol = self.parameter_manager.rdmol_list[sys_idx]
                for force_group_idx in range(self.force_group_count):
                    rdmol_smarts, map_dict = smarts_map_dict[force_group_idx]
                    matches = rdmol.GetSubstructMatches(rdmol_smarts)
                    for match in matches:
                        tagged_atoms = [match[idx] for idx in map_dict]
                        tagged_atoms = tuple(tagged_atoms)
                        if tagged_atoms in atom_dict:
                            r = atom_dict[tagged_atoms]
                            self.allocations[r] = force_group_idx

        if flush_to_systems:
            super().apply_changes()
    

    def set_allocation(self, index, value):

        _index = self.smarts_manager_allocations_mapping[index]
        smarts_manager = self.smarts_manager_list[
            self.smarts_manager_allocations[index]
        ]
        smarts_manager.set_allocation(_index, value)
        self.smarts_allocations[index] = value


    def get_max_allocation(self, index):

        _index = self.smarts_manager_allocations_mapping[index]
        smarts_manager = self.smarts_manager_list[
            self.smarts_manager_allocations[index]
        ]
        value = smarts_manager.get_max_allocation(_index)
        return value


    def get_smarts(self):

        smarts_list = list()
        for smarts_manager in self.smarts_manager_list:
            smarts_str = smarts_manager.get_smarts()
            smarts_list.append(smarts_str)
            
        return smarts_list


class ContinousGMMTypingVector(ParameterVectorLinearTransformation):

    def __init__(
        self, 
        parameter_manager,
        allocation_vecgmm,
        parameter_vector_values,
        parameter_name,
        scaling_value = None,
        ):

        super().__init__()

        if parameter_name not in parameter_manager.parm_names:
            raise ValueError(f"Parameter with name {parameter_name} not found in parameter_manager {parameter_manager.parm_names}.")

        if allocation_vecgmm.K != len(parameter_vector_values):
            raise ValueError(f"Number of components in allocation_vecgmm must match number of values in parameter_vector_values, but is {allocation_vecgmm.K} and {len(parameter_vector_values)}")

        self.parameter_manager  = parameter_manager
        self.parameter_name     = parameter_name
        self.allocation_vecgmm  = allocation_vecgmm
        self.vector_values      = parameter_vector_values
        self.scaling_value      = scaling_value

        ### Define the base types
        ### ============================
        ### Note this only makes really sense if calling ungroup(parameter_manager)
        ### before on the parameter_manager object.
        ### The only thing we really need to do here is to add each value.
        ### Note that we can only use one parameter type (e.g. k) per instance.

        for value in self.vector_values:
            self.append(value, scaling_value)

    @property
    def K(self):
        return self.allocation_vecgmm.K

    @property
    def N(self):
        return self.allocation_vecgmm.N

    def apply_changes(self):

        for force_group_idx in range(self.force_group_count):
            ### Note, here force_group_idx is equivalent to force_rank
            value = self.get_parm(force_group_idx)
            self.parameter_manager.set_parameter(force_group_idx,
                                                 self.parameter_name,
                                                 value)

    def get_parm(self, force_rank):

        rank_sele = self.allocation_vecgmm.rank_allocation_list[force_rank]
        L         = rank_sele.size
        rank_data = self.allocation_vecgmm.data[rank_sele].reshape(L,self.N)
        value     = 0. * self.vector_units[0]
        norm      = 0.
        for k in range(self.K):
            P_k   = np.sum(self.allocation_vecgmm.get_k(rank_data, k))
            if P_k > 1.0e-20:
                value += self.vector_k[k] * P_k
                norm  += P_k
        if norm > 1.0e-20:
            value /= norm

        return value

    def is_force_group_empty(self, force_group_idx):

        is_empty = self.parameter_manager.is_force_group_empty(force_group_idx)
        return is_empty

    def get_parameters_by_force_group(self, force_group_idx):

        value_list           = list()
        force_group_template = self.parameter_manager.force_group_forcecontainer_list[force_group_idx]
        value                = force_group_template.get_parameter(self.parameter_name)

        return value

    @property
    def force_group_count(self):

        return self.parameter_manager.force_group_count

    @property
    def force_group_forcecontainer_list(self):

        return self.parameter_manager.force_group_forcecontainer_list

    @property
    def force_group_size(self):

        values, counts = np.unique(
            self.parameter_manager.force_group_idx_list,
            return_counts=True
            )
        return counts

class GMM(object):

    __doc__ = """
    This is the original implementation of the Gaussian Mixture Model.
    It is now depracted.
    """

    def __init__(
        self, 
        K: int, 
        N: int) -> None:

        ### N dimensions
        self.N = int(N)

        ### weight vector
        self._w       = np.ones(0, dtype=float)
        ### Mean vector
        self._mu      = np.zeros((0,N), dtype=float)
        ### sigma matrix
        self._sig     = np.zeros((0, self.N, self.N), dtype=float)
        ### List of multivariate Gaussian objects
        self.gauss_list = list()

        for _ in range(K):
            self.add_component()

    @property
    def K(self) -> int:
        return self._w.size

    @property
    def mu(self) -> np.ndarray:
        return self._mu

    @property
    def sig(self) -> np.ndarray:
        return self._sig

    @property
    def w(self) -> np.ndarray:
        return self._w

    def __str__(self) -> str:

        return f"GMM with K={self.K} N={self.N}"

    def __repr__(self) -> str:

        return self.__str__()

    def add_component(
        self, 
        w: float,
        mu: np.ndarray, 
        sig: np.ndarray) -> None:

        _w   = np.append(self.w, w)
        _mu  = np.append(self.mu, np.expand_dims(mu, 0), axis=0)

        eye  = np.eye(self.N, dtype=float)
        np.fill_diagonal(eye, sig)
        _sig = np.append(self.sig, np.expand_dims(eye, 0), axis=0)

        ### Put back to class attributes
        self._mu  = _mu
        self._sig = _sig
        self._w   = _w

        self.update()

    def remove_component(
        self, 
        index: int) -> None:

        if self.K == 1:
            raise ValueError('Cannot remove components when K=1')
        if index > self.K-1:
            raise KeyError(f'Cannot remove component {index}. Only {self.K} components in GMM.')

        if index == -1:
            index = self.K

        if index == 0:
            _mu  = self.mu[1:]
            _sig = self.sig[1:]
            _w   = self.w[1:]
        elif index == self.K:
            _mu  = self.mu[:-1]
            _sig = self.sig[:-1]
            _w   = self.w[:-1]
        else:
            _mu  = np.zeros((self.K-1, self.N), dtype=float)
            _sig = np.zeros((self.K-1, self.N, self.N), dtype=float)
            _w   = np.zeros((self.K-1), dtype=float)

            _mu[:index]  = self.mu[:index]
            _sig[:index] = self.sig[:index]
            _w[:index]   = self.w[:index]

            _mu[index:]  = self.mu[index+1:]
            _sig[index:] = self.sig[index+1:]
            _w[index:]   = self.w[index+1:]

        ### Put back to class attributes
        self._mu  = _mu
        self._sig = _sig
        self._w   = _w

        self.update()

    def update(self) -> None:

        ### Perform some sanity checks
        assert self._mu.shape[0] == self.K
        assert self._sig.shape[0] == self.K

        assert self._mu.shape[1] == self.N
        assert self._sig.shape[1] == self.N
        assert self._sig.shape[2] == self.N

        ### Normalize weights
        self._w  = np.abs(self._w)
        self._w /= np.sum(self._w)

        self.gauss_list = list()
        for k in range(self.K):
            gaussian     = multivariate_normal(
                mean=self.mu[k], 
                cov=self.sig[k],
                seed=_SEED,
                allow_singular=True,
                )
            self.gauss_list.append(gaussian)

    def single_gauss(
        self, 
        x: np.ndarray, 
        k: int, 
        log: bool = False) -> np.ndarray:

        if x.shape[-1] != self.N:
            raise ValueError(
                f"Last dimension of input x must have length {self.N} but has length {x.shape[-1]}."
                )

        if log:
            return self.gauss_list[k].logpdf(x)
        else:
            return self.gauss_list[k].pdf(x)

    def get_k(
        self, 
        x: np.ndarray,         
        k: int,
        log: bool = False) -> np.ndarray:

        if x.shape[-1] != self.N:
            raise ValueError(
                f"Last dimension of input x must have length {self.N} but has length {x.shape[-1]}."
                )

        logP = self.single_gauss(x, k, log=True)
        if x.ndim == 1:
            result = np.array([
                math.fsum([logP + np.log(self.w[k])])
                ])
        else:
            result = np.apply_along_axis(
                lambda x: math.fsum([x + np.log(self.w[k])]),
                -1, 
                logP)
        if log:
            return result
        else:
            return np.exp(result)

    def __call__(
        self, 
        x: np.ndarray) -> np.ndarray:

        if x.shape[-1] != self.N:
            raise ValueError(f"Last dimension of input x must have length {self.N} but has length {x.shape[-1]}.")

        if x.ndim > 1:
            xshape = list(x.shape[:-1]) + [self.K]
            P      = np.zeros(xshape, dtype=float)
        else:
            P      = np.zeros(self.K, dtype=float)

        for k in range(self.K):
            if self.w[k] > 10.E-16:
                logP      = np.log(self.w[k]) + self.single_gauss(x, k, log=True)
                ### Select last axis
                ### no matter how many axis are in front
                P[...,-k] = np.exp(logP)
            else:
                P[...,-k] = 0.
        ### The fsum is needed for numerical stability
        ### when adding very small or very big numbers.
        if x.ndim > 1:
            ### The np.apply_along_axis might be really slow
            return np.apply_along_axis(math.fsum, -1, P)
        else:
            return np.array(math.fsum(P))


class VecGMM(object):

    def __init__(self, 
        K: int, 
        N: int, 
        w_vectype: BaseVector = BaseVector,
        mu_vectype: BaseVector = BaseVector,
        sig_vectype: BaseVector = BaseVector,
        covariance_type="diag"
        ) -> None:

        if covariance_type != "diag":
            NotImplementedError("Only diag covariances are implemented.")

        self.w_vec   = w_vectype()
        self.mu_vec  = mu_vectype()
        self.sig_vec = sig_vectype()

        self.covariance_type = covariance_type

        self.w_vec.call_back   = self.update
        self.mu_vec.call_back  = self.update
        self.sig_vec.call_back = self.update

        ### N dimensions
        self.N = int(N)

        ### weight vector
        self._w       = np.ones(0, dtype=float)
        ### Mean vector
        self._mu      = np.zeros((0,N), dtype=float)
        ### sigma matrix
        self._sig     = np.zeros((0, self.N, self.N), dtype=float)
        ### List of multivariate Gaussian objects
        self.gauss_list = list()

        for _ in range(K):
            self.add_component()

    @property
    def N_parms(self) -> int:

        N_parms = 0
        ### w
        N_parms += self.K
        ### mu
        N_parms += self.K * self.N
        ### sig
        N_parms += self.K * self.N

        return N_parms
    
    @property
    def size(self) -> int:

        return self.K * self.N

    def __str__(self) -> str:

        return f"GMM with K={self.K} N={self.N}"

    def __repr__(self) -> str:

        return self.__str__()

    @property
    def K(self) -> int:
        return self.w_vec.size

    @property
    def mu(self) -> np.ndarray:
        return self.mu_vec.vector_values.reshape(self.K, self.N)

    @property
    def sig(self) -> np.ndarray:
        sig_d = self.sig_vec.vector_values.reshape(self.K, self.N)
        sig_m = np.zeros((self.K, self.N, self.N), dtype=float)
        for k in range(self.K):
            np.fill_diagonal(sig_m[k], np.abs(sig_d[k]))
        return sig_m

    @property
    def w(self) -> np.ndarray:
        return self.w_vec.vector_values

    def update(self) -> None:

        ### Perform some sanity checks
        assert self.mu_vec.size == self.K * self.N
        assert self.sig_vec.size == self.K * self.N

        ### Normalize weights
        self.w_vec._vector_values  = np.abs(self.w_vec._vector_values)
        self.w_vec._vector_values /= np.sum(self.w_vec._vector_values)

        self.gauss_list = list()
        for k in range(self.K):
            gaussian     = multivariate_normal(mean=self.mu[k],
                                               cov=self.sig[k],
                                               seed=_SEED)
            self.gauss_list.append(gaussian)

    def add_component(
        self, 
        w: float= 0.,
        mu: float= 0.,
        sig: float= 0.) -> None:

        if not w:
            w = 1.

        if not mu:
            mu = np.zeros(self.N)
        else:
            mu = np.zeros(self.N) * mu

        if not sig:
            sig = np.ones(self.N)
        else:
            sig = np.ones(self.N) * sig

        self.w_vec.append(w)
        self.mu_vec.append(mu)
        self.sig_vec.append(sig)

        self.update()

    def remove_component(
        self, 
        index: int) -> None:

        if self.K == 1:
            raise ValueError('Cannot remove components when K=1')
        if index > self.K-1:
            raise KeyError(f'Cannot remove component {index}. Only {self.K} components in GMM.')

        if index == -1:
            index = self.K

        ### weights has length K
        self.w_vec.remove(index)
        ### mu has length K*N
        for _ in range(index*self.N, (index+1)*self.N):
            self.mu_vec.remove(index*self.N)
            self.sig_vec.remove(index*self.N)

        self.update()

    def apply_changes(self) -> None:

        self.update()

    def single_gauss(
        self, 
        x: np.ndarray, 
        k: int, 
        log: bool = False) -> float:

        if x.shape[-1] != self.N:
            raise ValueError(f"Last dimension of input x must have length {self.N} but has length {x.shape[-1]}.")

        if log:
            return self.gauss_list[k].logpdf(x)
        else:
            return self.gauss_list[k].pdf(x)

    def get_k(
        self, 
        x: np.ndarray, 
        k: int,
        log: bool = False) -> float:

        if x.shape[-1] != self.N:
            raise ValueError(f"Last dimension of input x must have length {self.N} but has length {x.shape[-1]}.")

        logP = self.single_gauss(x, k, log=True)
        if x.ndim == 1:
            result = np.array([
                math.fsum([logP + np.log(self.w[k])])
                ])

        else:
            result = np.apply_along_axis(
                lambda x: math.fsum([x + np.log(self.w[k])]),
                -1, 
                logP)
        if log:
            return result
        else:
            return np.exp(result)

    def __call__(
        self, 
        x: np.ndarray) -> float:

        if x.shape[-1] != self.N:
            raise ValueError(f"Last dimension of input x must have length {self.N} but has length {x.shape[-1]}.")

        if x.ndim > 1:
            xshape = list(x.shape[:-1]) + [self.K]
            P      = np.zeros(xshape, dtype=float)
        else:
            P      = np.zeros(self.K, dtype=float)

        for k in range(self.K):
            if self.w[k] > 10.E-16:
                logP      = np.log(self.w[k]) + self.single_gauss(x, k, log=True)
                ### Select last axis
                ### no matter how many axis are in front
                P[...,-k] = np.exp(logP)
            else:
                P[...,-k] = 0.
        ### The fsum is needed for numerical stability
        ### when adding very small or very big numbers.
        if x.ndim > 1:
            ### The np.apply_along_axis might be really slow
            return np.apply_along_axis(math.fsum, -1, P)
        else:
            return math.fsum(P)


class AllocationVecGMM(object):

    from typing import List

    def __init__(
        self, 
        nxmol_list: List[nx.Graph],
        parameter_manager: ParameterManager,
        grouping_graph: nx.Graph,
        grouping_nodes_dict: dict,
        grouping_edges_dict: dict,
        covariance_type: str = 'diag',
        w_vectype: BaseVector = BaseVector,
        mu_vectype: BaseVector = BaseVector,
        sig_vectype: BaseVector = BaseVector) -> None:

        ### ------------- ###
        ### Sanity checks ###
        ### ------------- ###
        assert max(parameter_manager.system_idx_list) + 1 == len(nxmol_list)
        for node_idx in grouping_graph.nodes():
            node = grouping_graph.nodes[node_idx]
            if "grouping" in node:
                h    = node["grouping"]
                assert h in grouping_nodes_dict
                assert "features" in grouping_nodes_dict[h]
                assert "N_components" in grouping_nodes_dict[h]
        for edge_idx in grouping_graph.edges():
            edge = grouping_graph.edges[edge_idx]
            if "grouping" in edge:
                h    = edge["grouping"]
                assert h in grouping_edges_dict
                assert "features" in grouping_edges_dict[h]
                assert "N_components" in grouping_edges_dict[h]

        ### --------------------------------------- ###
        ### Preorganize data according to groupings ###
        ### --------------------------------------- ###

        self.covariance_type = covariance_type

        self._grouping_nodes_dict = grouping_nodes_dict
        self._grouping_edges_dict = grouping_edges_dict

        max_force_rank = np.max(parameter_manager.force_ranks)+1

        self.node_data_dict = OrderedDict()
        self.edge_data_dict = OrderedDict()
        self.node_groupings_set  = set()
        self.edge_groupings_set  = set()
        for r in range(max_force_rank):
            self.node_data_dict[r]  = OrderedDict()
            self.edge_data_dict[r]  = OrderedDict()
            force_objects_of_rank_r = np.where(r == parameter_manager.force_ranks)[0]
            for force_object_idx in force_objects_of_rank_r:
                system_idx = parameter_manager.system_idx_list[force_object_idx]
                atom_list  = parameter_manager.atom_list[force_object_idx]
                atom_list  = sorted(atom_list)
                nxmol      = nxmol_list[system_idx]
                GM         = isomorphism.GraphMatcher(
                    nxmol,
                    grouping_graph
                    )
                ### Match grouping_graph to force objects based on 
                ### same atom indices. However, it is possible that
                ### there is no graph match for some nxmol (for instance,
                ### you cannot match (0)-(1)-(2)-(3) on methan). In that
                ### case, just skip.
                if not GM.subgraph_is_isomorphic():
                    continue
                found_force_object = False
                for match in GM.subgraph_monomorphisms_iter():
                    matched_atoms = list(match.keys())
                    matched_atoms = matched_atoms[:len(atom_list)]
                    ### The sorting is necessary to get a consistent ordering
                    ### of the atom indices. This will make sure that each 
                    ### atom set only is considered once in total.
                    if atom_list == sorted(matched_atoms):
                        found_force_object = True
                        match_reverse      = OrderedDict()
                        for nxmol_node_idx in matched_atoms:
                            grouping_node_idx = match[nxmol_node_idx]
                            match_reverse[grouping_node_idx] = nxmol_node_idx
                        break
                assert found_force_object

                ### ADD NODE DATA
                ### =============
                for grouping_node_idx in grouping_graph.nodes():
                    nxmol_node_idx = match_reverse[grouping_node_idx]
                    grouping_node  = grouping_graph.nodes[grouping_node_idx]
                    nxmol_node     = nxmol.nodes[nxmol_node_idx]
                    if not "grouping" in grouping_node:
                        continue
                    h = grouping_node["grouping"]
                    self.node_groupings_set.add(h)
                    features = self._grouping_nodes_dict[h]["features"]
                    N_feautures_nodes = len(features)
                    N_data_nodes      = nxmol_node["N_data"]
                    if not h in self.node_data_dict[r]:
                        self.node_data_dict[r][h] = list()
                    for data_idx in range(N_data_nodes):
                        data_nodes = list()
                        for feature_idx in range(N_feautures_nodes):
                            feature_name = features[feature_idx]
                            data_nodes.append(
                                nxmol_node[feature_name][data_idx]
                                )
                        self.node_data_dict[r][h].append(
                            data_nodes)

                ### ADD EDGE DATA
                ### =============
                for grouping_edge_idx in grouping_graph.edges():
                    grouping_node_1 = grouping_edge_idx[0]
                    grouping_node_2 = grouping_edge_idx[1]
                    nxmol_node_1    = match_reverse[grouping_node_1]
                    nxmol_node_2    = match_reverse[grouping_node_2]
                    nxmol_edge_idx  = (nxmol_node_1, nxmol_node_2)
                    grouping_edge   = grouping_graph.edges[grouping_edge_idx]
                    nxmol_edge      = nxmol.edges[nxmol_edge_idx]
                    if not "grouping" in grouping_edge:
                        continue
                    h = grouping_edge["grouping"]
                    self.edge_groupings_set.add(h)
                    features = self._grouping_edges_dict[h]["features"]
                    N_feautures_edges = len(features)
                    N_data_edges      = nxmol_edge["N_data"]
                    if not h in self.edge_data_dict[r]:
                        self.edge_data_dict[r][h] = list()
                    for data_idx in range(N_data_edges):
                        data_edges = list()
                        for feature_idx in range(N_feautures_edges):
                            feature_name = features[feature_idx]
                            data_edges.append(
                                nxmol_edge[feature_name][data_idx]
                                )
                        self.edge_data_dict[r][h].append(
                            data_edges)


    def build_gmm(self, allocation_vector):

        from sklearn.mixture import GaussianMixture

        unique_types = np.unique(allocation_vector.vector_values)

        self.gmm_nodes_dict = OrderedDict()
        self.gmm_edges_dict = OrderedDict()
        self.gaussian_allocations_node_dict = OrderedDict()
        self.gaussian_allocations_edge_dict = OrderedDict()
        self.node_group_weights = BaseVector(dtype=np.float64)
        self.edge_group_weights = BaseVector(dtype=np.float64)
        self.node_gmm_weights   = OrderedDict()
        self.edge_gmm_weights   = OrderedDict()
        for h in self.node_groupings_set:

            N_components_nodes = self._grouping_nodes_dict[h]["N_components"]

            for t in unique_types:
                if t == _INACTIVE_GROUP_IDX:
                    continue
                force_ranks = np.where(allocation_vector.vector_values == t)[0]
                data_nodes  = self.get_node_data(force_ranks, [h])

                if data_nodes.shape[0] < 2:
                    continue
                gm_nodes = GaussianMixture(
                    n_components=N_components_nodes,
                    covariance_type=self.covariance_type,
                    random_state=_SEED).fit(data_nodes)

                for k in range(N_components_nodes):
                    self.add_node_component(
                        grouping=h,
                        typing=t,
                        w=gm_nodes.weights_[k],
                        mu=gm_nodes.means_[k],
                        sig=gm_nodes.covariances_[k])
                    
        for h in self.edge_groupings_set:

            N_components_edges = self._grouping_edges_dict[h]["N_components"]

            for t in unique_types:
                if t == _INACTIVE_GROUP_IDX:
                    continue
                force_ranks = np.where(allocation_vector.vector_values == t)[0]
                data_edges  = self.get_edge_data(force_ranks, [h])

                if data_edges.shape[0] < 2:
                    continue
                gm_edges = GaussianMixture(
                    n_components=N_components_edges,
                    covariance_type=self.covariance_type,
                    random_state=_SEED).fit(data_edges)
                    
                for k in range(N_components_edges):
                    self.add_edge_component(
                        grouping=h,
                        typing=t,
                        w=gm_edges.weights_[k],
                        mu=gm_edges.means_[k],
                        sig=gm_edges.covariances_[k])

    @property
    def unique_types(self) -> np.ndarray:

        __doc__ = """
        Retrieve all available types over all node and edge GMM.
        """
        all_types = np.concatenate((
            self.unique_node_types,
            self.unique_node_types
            ))
        return np.unique(all_types)

    @property
    def unique_node_types(self) -> np.ndarray:

        __doc__ = """
        Retrieve all available types over all node GMM.
        """

        all_types = list()
        for h in self.node_groupings:
            node_allocations = self.gaussian_allocations_node_dict[h]
            all_types.extend(
                node_allocations.vector_values.tolist()
                )
        return np.unique(all_types)

    @property
    def unique_edge_types(self) -> np.ndarray:

        __doc__ = """
        Retrieve all available types over all edge GMM.
        """

        all_types = list()
        for h in self.node_groupings:
            edge_allocations = self.gaussian_allocations_edge_dict[h]
            all_types.extend(
                edge_allocations.vector_values.tolist()
                )
        return np.unique(all_types)

    @property
    def N_parms(self) -> int:

        __doc__="""
        Total number of parameters in all (node + edge) GMM
        """

        return self.N_node_parms + self.N_edge_parms

    @property
    def N_node_parms(self) -> int:

        __doc__ = """
        Total number of parameters in all node GMM.
        """

        N_node_parms = 0
        for h in self.gmm_nodes_dict.keys():
            N = self.gmm_nodes_dict[h].N
            K = self.gmm_nodes_dict[h].K
            ### w
            N_node_parms += K
            ### mu
            N_node_parms += (N * K)
            ### sig
            if self.covariance_type == 'diag':
                N_node_parms += (N * K)
            else:
                N_node_parms += (int(N*(N+1)/2) * K)
        return N_node_parms

    @property
    def N_edge_parms(self) -> int:

        __doc__ = """
        Total number of parameters in all edge GMM.
        """

        N_edge_parms = 0
        for h in self.gmm_edges_dict.keys():
            N = self.gmm_edges_dict[h].N
            K = self.gmm_edges_dict[h].K
            ### w
            N_edge_parms += K
            ### mu
            N_edge_parms += (N * K)
            ### sig
            if self.covariance_type == 'diag':
                N_edge_parms += (N * K)
            else:
                N_edge_parms += (int(N * (N + 1) / 2) * K)
        return N_edge_parms

    @property
    def K_nodes(self) -> int:

        __doc__ = """
        Total number of components in all nodes GMM.
        """

        N_nodes = 0
        for h in self.gaussian_allocations_node_dict.keys():
            N_nodes += self.gaussian_allocations_node_dict[h].size
        return N_nodes

    @property
    def K_edges(self) -> int:

        __doc__ = """
        Total number of components in all edges GMM.
        """

        N_edges = 0
        for h in self.gaussian_allocations_edge_dict.keys():
            N_edges += self.gaussian_allocations_edge_dict[h].size
        return N_edges

    @property
    def unique_force_ranks(self) -> np.ndarray:
        __doc__="""
        The list containing all ranks.
        """
        return np.array(list(self.node_data_dict.keys()))

    @property
    def max_force_rank(self) -> int:
        __doc__="""
        The maximum rank.
        """
        return max(self.unique_force_ranks)

    @property
    def H(self) -> int:
        __doc__ = """
        The total number of groups.
        """        
        return self.H_nodes + self.H_edges

    @property
    def H_nodes(self) -> int:
        __doc__ = """
        The number of node groups.
        """
        return len(self.gaussian_allocations_node_dict)

    @property
    def H_edges(self) -> int:
        __doc__ = """
        The number of edge groups.
        """
        return len(self.gaussian_allocations_edge_dict)

    @property
    def node_groupings(self) -> np.ndarray:
        __doc__="""
        List containing all node groupings
        """
        return np.array(list(self.gaussian_allocations_node_dict.keys()))

    @property
    def edge_groupings(self) -> np.ndarray:
        __doc__="""
        List containing all node groupings
        """
        return np.array(list(self.gaussian_allocations_edge_dict.keys()))

    def get_node_data(
        self, 
        force_ranks: list = list(),
        groupings: list = list()) -> np.ndarray:

        __doc__ = """
        Returns the node data for nodes with `ranks` and `groupings`.
        """

        if len(force_ranks) == 0:
            force_ranks = self.unique_force_ranks

        data_nodes = list()
        for r in force_ranks:
            if len(groupings) == 0:
                for h in self.node_data_dict[r].keys():
                    data_nodes.extend(
                        self.node_data_dict[r][h]
                        )                
            else:
                for h in groupings:
                    data_nodes.extend(
                        self.node_data_dict[r][h]
                        )
        return np.array(data_nodes)

    def get_edge_data(
        self, 
        force_ranks: list = list(),
        groupings: list = list()) -> np.ndarray:

        __doc__ = """
        Returns the edge data for nodes with `ranks` and `groupings`.
        """

        if len(force_ranks) == 0:
            force_ranks = self.unique_force_ranks

        data_edges = list()
        for r in force_ranks:
            if len(groupings) == 0:
                for h in self.edge_data_dict[r].keys():
                    data_edges.extend(
                        self.edge_data_dict[r][h]
                        )                
            else:
                for h in groupings:
                    data_edges.extend(
                        self.edge_data_dict[r][h]
                        )
        return np.array(data_edges)

    def update(self) -> None:

        self.node_group_weights /= np.sum(self.node_group_weights.vector_values)
        self.edge_group_weights /= np.sum(self.edge_group_weights.vector_values)

        for grouping in self.node_groupings:
            node_typing_vector = self.gaussian_allocations_node_dict[grouping]
            node_gmm_weights   = self.node_gmm_weights[grouping]
            unique_node_types  = np.unique(node_typing_vector.vector_values)
            for t in unique_node_types:
                gaussians_k = np.where(node_typing_vector.vector_values == t)[0]
                norm       = 0.
                for v in gaussians_k:
                    norm += node_gmm_weights[v]
                for v in gaussians_k:
                    node_gmm_weights[v] /= norm

        for grouping in self.edge_groupings:
            edge_typing_vector = self.gaussian_allocations_edge_dict[grouping]
            edge_gmm_weights   = self.edge_gmm_weights[grouping]
            unique_edge_types  = np.unique(edge_typing_vector.vector_values)
            for t in unique_edge_types:
                gaussians_k = np.where(edge_typing_vector.vector_values == t)[0]
                norm        = 0.
                for v in gaussians_k:
                    norm += edge_gmm_weights[v]
                for v in gaussians_k:
                    edge_gmm_weights[v] /= norm


    def add_node_component(
        self,
        grouping: int,
        typing: int,
        w: float,
        mu: np.ndarray,
        sig: np.ndarray) -> None:

        if sig.ndim == 1:
            assert mu.shape[0] == sig.shape[0]
        elif sig.ndim == 2:
            assert sig.shape[0] == sig.shape[1]
        else:
            raise ValueError(f"sig has {sig.ndim} dimensions, but should have only 1 or 2")

        if not grouping in self.gmm_nodes_dict:
            self.node_group_weights.append(1.)
            self.gmm_nodes_dict[grouping] = GMM(
                K=0,
                N=mu.shape[0]
                )
            self.gaussian_allocations_node_dict[grouping] = BaseVector(dtype=np.int64)
            self.node_gmm_weights[grouping] = BaseVector(dtype=np.float64)

        ### The weights in the GMM object are just mock weights.
        ### We don't actually use them.
        self.gmm_nodes_dict[grouping].add_component(1., mu, sig)
        self.gaussian_allocations_node_dict[grouping].append(typing)
        self.node_gmm_weights[grouping].append(w)


    def add_edge_component(
        self,
        grouping: int,
        typing: int,
        w: float,
        mu: np.ndarray,
        sig: np.ndarray) -> None:

        if sig.ndim == 1:
            assert mu.shape[0] == sig.shape[0]
        elif sig.ndim == 2:
            assert sig.shape[0] == sig.shape[1]
        else:
            raise ValueError(f"sig has {sig.ndim} dimensions, but should have only 1 or 2")

        ### For each grouping level (indexed by the key of the dict), 
        ### defines the allocation of each Gaussian component `k` to
        ### its GMM (a.k.a. FF parameter type).
        ### The value of each key must be of type BaseVector(dtype=np.int64)
        if not grouping in self.gmm_edges_dict:
            self.edge_group_weights.append(1.)
            self.gmm_edges_dict[grouping] = GMM(
                0,
                mu.shape[0]
                )
            self.gaussian_allocations_edge_dict[grouping] = BaseVector(dtype=np.int64)
            self.edge_gmm_weights[grouping] = BaseVector(dtype=np.float64)

        ### The weights in the GMM object are just mock weights.
        ### We don't actually use them.
        self.gmm_edges_dict[grouping].add_component(1., mu, sig)
        self.gaussian_allocations_edge_dict[grouping].append(typing)
        self.edge_gmm_weights[grouping].append(w)


    def remove_node_component(
        self,
        grouping: int,
        typing: int,
        k: int) -> None:

        __doc__ = """
        Remove node component with `grouping`, `typing` and `k`.
        `k` is the kth component in the GMM specified through 
        `grouping` and `typing`.
        """

        if not grouping in self.gaussian_allocations_node_dict:
            raise ValueError(f"grouping {grouping} not found.")
        if not typing in self.gaussian_allocations_node_dict[grouping]:
            raise ValueError(f"typing {typing} not found.")
        if k < 0:
            raise ValueError(f"k cannot be negative.")

        valids = self.gaussian_allocations_node_dict[grouping].index([typing])[0]
        if not k < valids.size:
            raise ValueError(f"k too small for {self.gaussian_allocations_node_dict[grouping]}")

        self.gmm_nodes_dict[grouping].remove_component(valids[k])
        self.gaussian_allocations_node_dict[grouping].remove(valids[k])
        self.node_gmm_weights[grouping].remove(valids[k])

        self.update()

    def remove_node_typing(
        self,
        grouping: int,
        typing: int) -> None:

        __doc__ = """
        Remove typing with `grouping` and `typing`.
        This removes all components of that specific GMM.
        """
        remove_list = list(
            range(
                self.count_node_components(
                    grouping, 
                    typing
                    )
                )
            )[::-1]        

        for k in np.sort(remove_list)[::-1]:
            self.remove_node_component(
                grouping,
                typing,
                k
                )

    def count_node_components(
        self,
        grouping: int,
        typing: int) -> int:

        __doc__ = """
        For a GMM specified through `grouping` and `typing`, return
        the number of components.
        """

        if not grouping in self.gaussian_allocations_node_dict:
            raise ValueError(f"grouping {grouping} not found.")
        if not typing in self.gaussian_allocations_node_dict[grouping]:
            raise ValueError(f"typing {typing} not found.")

        valids = self.gaussian_allocations_node_dict[grouping].index([typing])[0]
        return valids.size

    def get_node_gmm_parms(
        self,
        grouping: int,
        typing: int) -> tuple:

        if not grouping in self.gaussian_allocations_node_dict:
            raise ValueError(f"grouping {grouping} not found.")
        if not typing in self.gaussian_allocations_node_dict[grouping]:
            raise ValueError(f"typing {typing} not found.")

        valids = self.gaussian_allocations_node_dict[grouping].index([typing])[0]
        K   = valids.size
        N   = self.gmm_nodes_dict[grouping].N
        w   = np.zeros(K, dtype=float)
        mu  = np.zeros((K, N), dtype=float)
        if self.covariance_type == 'diag':
            sig = np.zeros((K, N), dtype=float)
        else:
            sig = np.zeros((K, N, N), dtype=float)

        for k in range(K):
            w[k]  = self.node_gmm_weights[grouping][valids[k]]
            mu[k] = self.gmm_nodes_dict[grouping].mu[valids[k]]
            if self.covariance_type == 'diag':
                sig[k] = np.diag(self.gmm_nodes_dict[grouping].sig[valids[k]])
            else:
                sig[k] = self.gmm_nodes_dict[grouping].sig[valids[k]]

        return (w, mu, sig)


    def remove_edge_component(
        self,
        grouping: int,
        typing: int,
        k: int) -> None:

        __doc__ = """
        Remove edge component with `grouping`, `typing` and `k`.
        `k` is the kth component in the GMM specified through 
        `grouping` and `typing`.
        """

        if not grouping in self.gaussian_allocations_edge_dict:
            raise ValueError(f"grouping {grouping} not found.")
        if not typing in self.gaussian_allocations_edge_dict[grouping]:
            raise ValueError(f"typing {typing} not found.")
        if k < 0:
            raise ValueError(f"k cannot be negative.")

        valids = self.gaussian_allocations_edge_dict[grouping].index([typing])[0]
        if not k < valids.size:
            raise ValueError(f"k too small for {self.gaussian_allocations_edge_dict[grouping]}")

        self.gmm_edges_dict[grouping].remove_component(valids[k])
        self.gaussian_allocations_edge_dict[grouping].remove(valids[k])
        self.edge_gmm_weights[grouping].remove(valids[k])

        self.update()

    def remove_edge_typing(
        self,
        grouping: int,
        typing: int) -> None:

        __doc__ = """
        Remove typing with `grouping` and `typing`.
        This removes all components of that specific GMM.
        """
        remove_list = list(
            range(
                self.count_edge_components(
                    grouping, 
                    typing
                    )
                )
            )[::-1]        

        for k in np.sort(remove_list)[::-1]:
            self.remove_edge_component(
                grouping,
                typing,
                k
                )

    def count_edge_components(
        self,
        grouping: int,
        typing: int) -> int:

        __doc__ = """
        For a GMM specified through `grouping` and `typing`, return
        the number of components.
        """

        if not grouping in self.gaussian_allocations_edge_dict:
            raise ValueError(f"grouping {grouping} not found.")
        if not typing in self.gaussian_allocations_edge_dict[grouping]:
            raise ValueError(f"typing {typing} not found.")

        valids = self.gaussian_allocations_edge_dict[grouping].index([typing])[0]
        return valids.size

    def get_k(
        self, 
        k,
        log: bool = False) -> np.ndarray:

        __doc__="""
        Compute the likelihood for each rank having type k.
        Likelihoods are not normalized. In order to compute norm
        for each rank, compute the sum over all k for each rank.
        """

        self.update()

        P = np.zeros(
            self.max_force_rank+1, 
            dtype=np.float64)

        ### For each rank compute:
        ### P( t(r) = k ) = Σ(h in H) Σ(g in G(h)) Σ(k in K(g)) w_h w_k 1/N(r,h) Σ(d in D(r,h)) N(d|...)
        ###
        ### H: List of groupings
        ### G(h): List of all GMM for h
        ### K(g): List of all Gaussians for g
        ### D(r,h): List of all data for (r,h)
        ### N(r,h): Number of datapoints for (r,h)

        for r in self.unique_force_ranks:
            prob_sum  = list()
            for node_grouping in self.node_groupings:
                node_gmm           = self.gmm_nodes_dict[node_grouping]
                node_gmm_weights   = self.node_gmm_weights[node_grouping]
                node_group_weights = self.node_group_weights[node_grouping]
                node_typing_vector = self.gaussian_allocations_node_dict[node_grouping]
                node_gaussians_k   = node_typing_vector.index([k])[0]
                if node_group_weights < 1.E-16:
                    continue
                node_data   = self.get_node_data([r], groupings=[node_grouping])
                N_node_data = node_data.shape[0]
                for g_k in node_gaussians_k:
                    if node_gmm_weights[g_k] < 1.E-16:
                        continue
                    ### Not sure what function to use here in order to
                    ### aggregate the data. For now let's just average out.
                    ### Here we could invoke boolean and/or logic as follows:
                    ### 
                    ### BOOL  idea1     idea2
                    ### ----  -----     -----
                    ### AND   min(A,B)  A*B
                    ### OR    max(A,B)  1-[(1-A)*(1-B)]
                    if N_node_data > 1:
                        result_k = math.fsum(
                            node_gmm.single_gauss(
                                node_data, 
                                g_k,
                                log=False
                                )
                            )
                    else:
                        result_k = node_gmm.single_gauss(
                            node_data, 
                            g_k,
                            log=False
                            )
                    if result_k < 1.E-16:
                        continue
                    prob_sum.append(0.)
                    prob_sum[-1]  = np.log(node_group_weights)
                    prob_sum[-1] += np.log(node_gmm_weights[g_k])
                    if N_node_data > 1:
                        prob_sum[-1] -= np.log(N_node_data)
                    prob_sum[-1] += np.log(result_k)
                    prob_sum[-1]  = np.exp(prob_sum[-1])

            for edge_grouping in self.edge_groupings:
                edge_gmm           = self.gmm_edges_dict[edge_grouping]
                edge_gmm_weights   = self.edge_gmm_weights[edge_grouping]
                edge_group_weights = self.edge_group_weights[edge_grouping]
                edge_typing_vector = self.gaussian_allocations_edge_dict[edge_grouping]
                edge_gaussians_k   = edge_typing_vector.index([k])[0]
                if edge_group_weights < 1.E-16:
                    continue
                edge_data   = self.get_edge_data([r], groupings=[edge_grouping])
                N_edge_data = edge_data.shape[0]
                for g_k in edge_gaussians_k:
                    if edge_gmm_weights[g_k] < 1.E-16:
                        continue
                    if N_edge_data > 1:
                        result_k = math.fsum(
                            edge_gmm.single_gauss(
                                edge_data,
                                g_k,
                                log=False
                                )
                            )
                    else:
                        result_k = edge_gmm.single_gauss(
                            edge_data, 
                            g_k,
                            log=False
                            )
                    if result_k < 1.E-16:
                        continue
                    #print(r, edge_grouping, g_k,
                    #    edge_gmm.single_gauss(
                    #        edge_data, 
                    #        g_k,
                    #        log=False
                    #        ),
                    #    edge_data,
                    #    edge_gmm.mu[g_k]
                    #    )
                    prob_sum.append(0.)
                    prob_sum[-1] += np.log(edge_group_weights)
                    prob_sum[-1] += np.log(edge_gmm_weights[g_k])
                    if N_edge_data > 1:
                        prob_sum[-1] -= np.log(N_edge_data)
                    prob_sum[-1] += np.log(result_k)
                    prob_sum[-1]  = np.exp(prob_sum[-1])

            P[r] = math.fsum(prob_sum)

        if log:
            ### Take care of P==0. situations
            valids = np.where(P > 0.)
            result = np.zeros_like(P)
            result[:] = -np.inf
            result[valids] = np.log(P[valids])
            return result
        else:
            return P
