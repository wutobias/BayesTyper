#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import numpy as np
import copy
from itertools import permutations
import openmm
from openmm import unit
from .constants import _UNIT_QUANTITY
from .tools import is_type_or_raise
from .system import System, SystemManager

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================
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
                        _INACTIVE_GROUP_IDX)

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================


class ForceTypeContainer(object):

    N_parms       = None
    N_atoms       = None
    parm_names    = None

    def __init__(
        self, 
        parms: dict,
        exclude_list: list=list(),
        ):

        self.is_active = True
        self.exclude_list = exclude_list

    def update(self, force, force_idx, atom_list, name_list):

        pass

    def activate(self):

        self.is_active = True

    def deactivate(self):

        self.is_active = False

    def set_parameter(
        self,
        name: str, 
        value: _UNIT_QUANTITY,
        force: openmm.Force,
        force_idx_list: list,
        atom_list_list: list):

        N_force_entity = len(force_idx_list)
        assert len(atom_list_list) == N_force_entity

        setattr(self, name, value)
        for i in range(N_force_entity):
            assert len(atom_list_list[i]) == self.N_atoms
            self.update(
                force, 
                force_idx_list[i], 
                atom_list_list[i],
                )

    def get_parameter(
        self, 
        name: str):

        return getattr(self, name)

    def __eq__(
        self, 
        other):

        return are_containers_same(self, other)


class LJC14ScaleContainer(ForceTypeContainer):

    N_parms    = 2
    N_atoms    = 2
    parm_names = ['coulomb14Scale', 'lj14Scale']

    def __init__(self, parms, exclude_list=list()):

        super().__init__(parms, exclude_list)
        
        self._coulomb14Scale = parms['coulomb14Scale']
        self._lj14Scale      = parms['lj14Scale']

    def update(self, force, force_idx, atom_list):

        ### geometric mean: sqrt(eps_1 * eps_2)
        ### arithmetic mean: (sig_1 + sig_2)/2.

        q1, sig1, eps1 = force.getParticleParameters(atom_list[0])
        q2, sig2, eps2 = force.getParticleParameters(atom_list[1])

        q12   = q1 * q2
        sig12 = (sig1 + sig2) / 2.
        eps12 = np.sqrt(eps1 * eps2)

        if self.is_active:
            if not "coulomb14Scale" in self.exclude_list:
                q12   *= self.coulomb14Scale
            if not "lj14Scale" in self.exclude_list:
                eps12 *= self.lj14Scale

        force.setExceptionParameters(
            index = force_idx,
            particle1 = atom_list[0],
            particle2 = atom_list[1],
            chargeProd = q12,
            sigma = sig12,
            epsilon = eps12,
            )

    @property
    def coulomb14Scale(self):
        return float(self._coulomb14Scale)

    @property
    def lj14Scale(self):
        return float(self._lj14Scale)

    @coulomb14Scale.setter
    def coulomb14Scale(
        self, 
        value: _UNIT_QUANTITY):
        self._coulomb14Scale = float(value)

    @lj14Scale.setter
    def lj14Scale(
        self, 
        value: _UNIT_QUANTITY):
        self._lj14Scale = float(value)


class BondContainer(ForceTypeContainer):

    N_parms    = 2
    N_atoms    = 2
    parm_names = ['k', 'length']

    def __init__(self, parms, exclude_list=list()):

        super().__init__(parms, exclude_list)
        
        self._length = parms['length']
        self._k      = parms['k']

    def update(self, force, force_idx, atom_list):

        _, _, length, k = force.getBondParameters(force_idx)
        if not "length" in self.exclude_list:
            length = self.length
        if not "k" in self.exclude_list:
            k = self.k

        if self.is_active:
            force.setBondParameters(
                force_idx,
                atom_list[0],
                atom_list[1],
                length=length,
                k=k
                )
        else:
            force.setBondParameters(
                force_idx,
                atom_list[0],
                atom_list[1],
                length=length,
                k=0.
                )

    @property
    def length(self):
        return self._length.in_units_of(_LENGTH)

    @property
    def k(self):
        return self._k.in_units_of(_FORCE_CONSTANT_BOND)

    @length.setter
    def length(
        self, 
        value: _UNIT_QUANTITY):
        self._length = value.in_unit_system(unit.md_unit_system)

    @k.setter
    def k(
        self, 
        value: _UNIT_QUANTITY):
        self._k = value.in_unit_system(unit.md_unit_system)


class AngleContainer(ForceTypeContainer):

    N_parms    = 2
    N_atoms    = 3
    parm_names = ['k', 'angle']

    def __init__(self, parms, exclude_list=list()):

        super().__init__(parms, exclude_list)

        self._angle = parms['angle']
        self._k     = parms['k']

    def update(self, force, force_idx, atom_list):

        _, _, _, angle, k = force.getAngleParameters(force_idx)
        if not "angle" in self.exclude_list:
            angle = self.angle
        if not "k" in self.exclude_list:
            k = self.k

        if self.is_active:
            force.setAngleParameters(
                force_idx,
                atom_list[0],
                atom_list[1],
                atom_list[2],
                angle=angle,
                k=k
                )
        else:
            force.setAngleParameters(
                force_idx,
                atom_list[0],
                atom_list[1],
                atom_list[2],
                angle=angle,
                k=0.
                )
    @property
    def angle(self):
        return self._angle.in_units_of(_ANGLE)

    @property
    def k(self):
        return self._k.in_units_of(_FORCE_CONSTANT_ANGLE)

    @k.setter
    def k(
        self, 
        value: _UNIT_QUANTITY):
        self._k = value.in_unit_system(unit.md_unit_system)

    @angle.setter
    def angle(
        self,
        value: _UNIT_QUANTITY):
        self._angle = value.in_unit_system(unit.md_unit_system)


class ProperTorsionContainer(ForceTypeContainer):

    N_parms    = 3
    N_atoms    = 4
    parm_names = ['k', 'phase', 'periodicity']

    def __init__(self, parms, exclude_list=list()):

        super().__init__(parms, exclude_list)

        self._periodicity = parms['periodicity']
        self._phase       = parms['phase']
        self._k           = parms['k']

    def update(self, force, force_idx, atom_list):

        _, _, _, _, periodicity, phase, k = force.getTorsionParameters(force_idx)
        if not "periodicity" in self.exclude_list:
            periodicity = self.periodicity
        if not "phase" in self.exclude_list:
            phase = self.phase
        if not "k" in self.exclude_list:
            k = self.k

        if self.is_active:
            force.setTorsionParameters(
                force_idx,
                atom_list[0],
                atom_list[1],
                atom_list[2],
                atom_list[3],
                periodicity=periodicity,
                phase=phase,
                k=k
                )
        else:
            force.setTorsionParameters(
                force_idx,
                atom_list[0],
                atom_list[1],
                atom_list[2],
                atom_list[3],
                periodicity=periodicity,
                phase=phase,
                k=0.*_FORCE_CONSTANT_TORSION
                )

    @property
    def periodicity(self):
        return float(self._periodicity)

    @property
    def phase(self):
        return self._phase.in_units_of(_ANGLE)

    @property
    def k(self):
        return self._k.in_units_of(_FORCE_CONSTANT_TORSION)

    @periodicity.setter
    def periodicity(
        self, 
        value: _UNIT_QUANTITY):
        self._periodicity = float(value)

    @phase.setter
    def phase(
        self, 
        value: _UNIT_QUANTITY):
        self._phase = value.in_unit_system(unit.md_unit_system)

    @k.setter
    def k(
        self, 
        value: _UNIT_QUANTITY):
        self._k = value.in_unit_system(unit.md_unit_system)


class MultiProperTorsionContainer(ForceTypeContainer):

    N_parms    = None
    N_atoms    = 4
    parm_names = list()

    def __init__(self, parms, exclude_list=list()):

        super().__init__(parms, exclude_list)

        self.N_atoms    = 4
        self.parm_names = list()
        self.N_parms    = len(parms)

        N_periodicity = 0
        N_phase = 0
        N_k = 0

        for p in parms:
            if p.startswith("periodicity"):
                N_periodicity += 1
                self.parm_names.append(p)
            elif p.startswith("phase"):
                N_phase += 1
                self.parm_names.append(p)
            elif p.startswith("k"):
                N_k += 1
                self.parm_names.append(p)
            else:
                raise ValueError(
                    f"Parameter {p} not understood."
                    )

            setattr(self, p, parms[p])

        assert N_periodicity == N_phase == N_k


    def __setattr__(self, name, value):
        if name.startswith("k") or name.startswith("phase"):
            value = value.in_unit_system(unit.md_unit_system)
        elif name.startswith("periodicity"):
            value = float(value)
        else:
            pass
        super(MultiProperTorsionContainer, self).__setattr__(name, value)

    def __getattribute__(self, name):
        value = super(MultiProperTorsionContainer, self).__getattribute__(name)
        if name.startswith("k"):
            value = value.in_units_of(_FORCE_CONSTANT_TORSION)
        elif name.startswith("phase"):
            value = value.in_units_of(_ANGLE)
        elif name.startswith("periodicity"):
            value = float(value)
        else:
            pass
        return value

    def set_parameter(
        self,
        name: str, 
        value: _UNIT_QUANTITY,
        force: openmm.Force,
        force_idx_list: list,
        atom_list_list: list):

        N_force_entity = len(force_idx_list)
        assert len(atom_list_list) == N_force_entity

        setattr(self, name, value)
        for i in range(N_force_entity):
            assert len(atom_list_list[i]) == self.N_atoms
            self.update(
                force, 
                force_idx_list[i], 
                atom_list_list[i],
                )

    def update(self, force, force_idx_list, atom_list):

        ### `force_idx` is an iterable
        for idx, force_idx in enumerate(force_idx_list):
            _, _, _, _, periodicity, phase, k = force.getTorsionParameters(force_idx)
            if not "periodicity" in self.exclude_list:
                periodicity = getattr(self, f"periodicity{idx}")
            if not "phase" in self.exclude_list:
                phase = getattr(self, f"phase{idx}")
            if not "k" in self.exclude_list:
                k = getattr(self, f"k{idx}")

            if self.is_active:
                force.setTorsionParameters(
                    force_idx,
                    atom_list[0],
                    atom_list[1],
                    atom_list[2],
                    atom_list[3],
                    periodicity=periodicity,
                    phase=phase,
                    k=k
                    )
            else:
                force.setTorsionParameters(
                    force_idx,
                    atom_list[0],
                    atom_list[1],
                    atom_list[2],
                    atom_list[3],
                    periodicity=periodicity,
                    phase=phase,
                    k=0.*_FORCE_CONSTANT_TORSION
                    )


class DoubleProperTorsionContainer(ForceTypeContainer):

    N_parms    = 3
    N_atoms    = 4
    parm_names = ['k0', 'k1', 'periodicity']

    def __init__(self, parms, exclude_list=list()):

        super().__init__(parms, exclude_list)

        self._phase0 = 0. * unit.radian
        self._phase1 = 0.5 * np.pi * unit.radian

        self._periodicity = parms['periodicity']
        self._k0          = parms['k0']
        self._k1          = parms['k1']

    def update(self, force, force_idx, atom_list):

        force_idx0, force_idx1 = force_idx

        _, _, _, _, periodicity, _, k0 = force.getTorsionParameters(force_idx0)
        if not "periodicity" in self.exclude_list:
            periodicity = self.periodicity
        if not "k0" in self.exclude_list:
            k0 = self.k0
        _, _, _, _, periodicity, _, k1 = force.getTorsionParameters(force_idx1)
        if not "k1" in self.exclude_list:
            k1 = self.k1

        if self.is_active:
            force.setTorsionParameters(
                force_idx0,
                atom_list[0],
                atom_list[1],
                atom_list[2],
                atom_list[3],
                periodicity=periodicity,
                phase=self._phase0,
                k=k0
                )
            force.setTorsionParameters(
                force_idx1,
                atom_list[0],
                atom_list[1],
                atom_list[2],
                atom_list[3],
                periodicity=periodicity,
                phase=self._phase1,
                k=k1
                )
        else:
            force.setTorsionParameters(
                force_idx0,
                atom_list[0],
                atom_list[1],
                atom_list[2],
                atom_list[3],
                periodicity=periodicity,
                phase=self._phase0,
                k=0.*_FORCE_CONSTANT_TORSION
                )
            force.setTorsionParameters(
                force_idx1,
                atom_list[0],
                atom_list[1],
                atom_list[2],
                atom_list[3],
                periodicity=periodicity,
                phase=self._phase1,
                k=0.*_FORCE_CONSTANT_TORSION
                )

    @property
    def periodicity(self):
        return float(self._periodicity)

    @property
    def phase0(self):
        return self._phase0.in_units_of(_ANGLE)

    @property
    def k0(self):
        return self._k0.in_units_of(_FORCE_CONSTANT_TORSION)

    @property
    def phase1(self):
        return self._phase1.in_units_of(_ANGLE)

    @property
    def k1(self):
        return self._k1.in_units_of(_FORCE_CONSTANT_TORSION)

    @periodicity.setter
    def periodicity(
        self, 
        value: _UNIT_QUANTITY):
        self._periodicity = float(value)

    @k0.setter
    def k0(
        self, 
        value: _UNIT_QUANTITY):
        self._k0 = value.in_unit_system(unit.md_unit_system)

    @k1.setter
    def k1(
        self, 
        value: _UNIT_QUANTITY):
        self._k1 = value.in_unit_system(unit.md_unit_system)


class ImproperTorsionContainer(ProperTorsionContainer):

    def __init__(self, parms):

        super().__init__(parms)


class ParameterManager(object):

    __doc__ = """
    Base class for making interfaces to changing and reallocating
    parameters over a set of system objects.
    """

    def __init__(
        self,
        forcecontainer: ForceTypeContainer,
        exclude_list: list = list()
        ) -> None:

        self.forcecontainer  = forcecontainer
        self.exclude_list    = exclude_list
        self.name            = "ParameterManager"
        self.parm_names      = forcecontainer.parm_names
        self.N_parms         = forcecontainer.N_parms
        self.N_atoms         = forcecontainer.N_atoms

        ### force_group    : An integer identifier for what is commonly referred
        ###                  as 'FF parameter type'. Within the parameter manager
        ###                  each the `force_group` coincides with an individual
        ###                  forcecontainer.
        ###
        ### forcecontainer: A container that stores the parameters for a given
        ###                  force group. It can apply these parameters to openmm
        ###                  force objects.
        ###
        ### force_entity   : An integer identifier of a specific force for a specific
        ###                  substructure in a openmm system (e.g. the middle CC bond 
        ###                  or *one* of the terminal CH bonds in CCCC). 
        ###                  Note, force entities can be inactive (e.g. some torsions 
        ###                  are turned off for a give periodicity).
        ###
        ### force_rank     : For each force_entity, we have an integer identifier
        ###                  that gives it a symmetry rank. Any set of `force_entity`
        ###                  that have the same `force_rank` must have the same
        ###                  FF parameters (e.g. the terminal CC bonds in CCCC). This
        ###                  essentially ensures symmetry in the molecule and its.
        ###                  potential function.

        ### This list stores the force entity in each OpenMM force object
        ### in each system
        ### shape is `(self.force_entity_count,)`
        self.force_entity_list    = np.array([], dtype=int)
        ### This list stores the force_group_idx of each force entity
        ### This includes activate and inactive force entities
        ### shape is `(self.force_entity_count,)`
        self.force_group_idx_list = np.array([], dtype=int)
        ### This list stores the forcecontainer object for each
        ### force group. These forcecontainers only hold the parameters
        ### and must be called with set_parameter(name, value, omm_force, force_idx)
        ### to update force `force_idx` in an openmm force object `omm_force`.
        ### shape is `(self.force_group_count,)`
        self.forcecontainer_list  = list()
        ### This list stores the rank of each force entity. Those force entities
        ### that have the same rank are considered equal due to molecualar
        ### symmetry.
        ### shape is `(self.force_entity_count,)`
        self.force_ranks          = np.array([], dtype=int)
        ### For each force entity store the `system_idx` of the system
        ### it belongs to.
        ### shape is `(self.force_entity_count,)`
        self.system_idx_list      = np.array([], dtype=int)

        self._N_systems           = 0

        ### List of atoms in each force entity
        ### shape is `(self.force_entity_count,)`
        self.atom_list            = list()
        ### List of atomic number for each atom
        ### in `self.atom_list`
        self.atomic_num_list      = list()
        ### List that stores all the systems
        ### shape is `(self.N_systems,)`
        self.system_list          = list()
        ### List that stores all the systems
        ### rdkit objects
        self.rdmol_list           = list()
        ### This list stores all the OpenMM forces
        ### shape is `(self.N_systems,)`
        self.omm_force_list       = list()

    @property
    def unique_force_group_idx_list(self):

        return np.unique(self.force_group_idx_list)

    @property
    def N_systems(self):

        return self._N_systems

    @property
    def force_group_count(self):

        return len(self.forcecontainer_list)

    @property
    def force_entity_count(self):

        return self.force_ranks.size

    @property
    def max_rank(self):

        if self.force_entity_count == 0:
            return -1
        else:
            return int(np.max(self.force_ranks))

    def remove_system(
        self,
        system_idx_list
        ):

        __doc__ = """
        Remove systems with idx. This will only delete the systems from
        the parameter manager. The parameters will remain.
        """

        import numpy as np

        to_delete = list()
        system_idx_delete_list = list()
        for system_idx in np.unique(system_idx_list):
            if isinstance(system_idx, str):
                system_name = system_idx
                system_idx  = None
                for _system_idx in range(self.N_systems):
                    system = self.system_list[_system_idx]
                    if system.name == system_name:
                        system_idx = _system_idx
                        break
                if system_idx == None:
                    continue
            if system_idx < 0:
                system_idx = self.N_systems+system_idx
            idx_list = np.where(
                    self.system_idx_list == system_idx
                    )[0]
            if idx_list.size == 0:
                continue
            to_delete.extend(idx_list.tolist())
            system_idx_delete_list.append(system_idx)
            
        system_idx_delete_list = np.unique(system_idx_delete_list).tolist()
        system_idx_delete_list = sorted(system_idx_delete_list, reverse=True)
        for system_idx in system_idx_delete_list:
            self.system_list.pop(system_idx)
            self.rdmol_list.pop(system_idx)
            self.omm_force_list.pop(system_idx)

        self._N_systems -= len(system_idx_delete_list)

        to_delete = np.unique(to_delete).tolist()
        to_delete = sorted(to_delete, reverse=True)
        _force_entity_list    = np.delete(self.force_entity_list, to_delete,    axis=0)
        _force_group_idx_list = np.delete(self.force_group_idx_list, to_delete, axis=0)
        _force_ranks          = np.delete(self.force_ranks, to_delete,          axis=0)
        _system_idx_list      = np.delete(self.system_idx_list, to_delete,      axis=0)
        _atom_list            = np.delete(self.atom_list, to_delete,            axis=0)
        _atomic_num_list      = np.delete(self.atomic_num_list, to_delete,      axis=0)

        self.force_entity_list    = _force_entity_list
        self.force_group_idx_list = _force_group_idx_list
        self.force_ranks          = _force_ranks
        self.system_idx_list      = _system_idx_list.tolist()
        self.atom_list            = _atom_list.tolist()
        self.atomic_num_list      = _atomic_num_list.tolist()

        ### Finally, reset the system_idx_list
        system_idx_list_unique = np.unique(self.system_idx_list)

        assert system_idx_list_unique.size == self.N_systems

        _system_idx_list = np.copy(self.system_idx_list)
        for sys_idx_new in range(self.N_systems):
            sys_idx_old = system_idx_list_unique[sys_idx_new]
            valids = np.where(self.system_idx_list == sys_idx_old)[0]
            _system_idx_list[valids] = sys_idx_new

        self.system_idx_list = _system_idx_list

        ### ... and the force_ranks
        force_ranks_unique = np.unique(self.force_ranks)
        N_ranks = force_ranks_unique.size

        _force_ranks = np.copy(self.force_ranks)
        for rank_new in range(N_ranks):
            rank_old = force_ranks_unique[rank_new]
            valids = np.where(self.force_ranks == rank_old)[0]
            _force_ranks[valids] = rank_new

        self.force_ranks = _force_ranks

    def rebuild_from_systems(
        self,
        system_list: list,
        lazy: bool = False) -> None:

        __doc__ = """
        Rebuild the whole data structure from a list of systems.
        If `lazy=True`, then the systems and forces are just updated
        without complete rebuild. This is usually sufficient if you
        know what you're doing.
        """

        assert len(system_list) == self._N_systems

        if lazy:
            self.omm_force_list = list()
            self.system_list    = list()
            self.rdmol_list     = list()

        else:
            forcecontainer_list_old = copy.deepcopy(self.forcecontainer_list)
            if hasattr(self, "periodicity"):
                self.__init__(self.periodicity, self.phase)
            else:
                self.__init__()

        for system in system_list:

            if lazy:
                found_force = False
                for force_idx, force in enumerate(system.openmm_system.getForces()):
                    if force.__class__.__name__ == self.force_tag:
                        found_force = True
                        break

                ### If, for whatever reason, the force tag is not part of the
                ### system object, we cannot procede here.
                if not found_force:
                    raise Exception(
                        f"Could not find force {self.force_tag} in system {system.name}.")

                self.omm_force_list.append(force_idx)
                self.system_list.append(system)
                import copy
                self.rdmol_list.append(
                    copy.deepcopy(system.rdmol)
                )
            else:
                self.add_system(system)

        if not lazy:

            ### The new forcecontainer_list can have fewer parameters than the old one.
            diff = len(forcecontainer_list_old) - len(self.forcecontainer_list)
            assert diff > -1,\
            f"Old forcecontainer list ({len(forcecontainer_list_old)}) cannot be shorter than new one ({len(self.forcecontainer_list)})."

            ### Compare each new with each old force container
            mapping_dict = dict()
            forcecontainer_list = copy.deepcopy(self.forcecontainer_list)
            for force_group_idx_new, forcecontainer_new in enumerate(forcecontainer_list):
                for force_group_idx_old, forcecontainer_old in enumerate(forcecontainer_list_old):
                    if forcecontainer_new == forcecontainer_old:
                        ### If we already found the mapping and are
                        ### here *again*, then we have two redundant types.
                        ### We want to keep them.
                        if force_group_idx_new in mapping_dict:
                            parm_name_list = self.parm_names
                            input_parms_dict = dict()
                            for parameter_name in parm_name_list:
                                value = self.get_parameter(
                                    force_group_idx_new,
                                    parameter_name
                                    )
                                input_parms_dict[parameter_name] = value
                            self.add_force_group(input_parms_dict)
                            mapping_dict[self.force_group_count - 1] = force_group_idx_old
                        else:
                            mapping_dict[force_group_idx_new] = force_group_idx_old

            assert len(mapping_dict) == len(self.forcecontainer_list)
            assert len(forcecontainer_list_old) == len(self.forcecontainer_list)

            force_group_idx_list = np.copy(self.force_group_idx_list)
            for force_group_idx_new, force_group_idx_old in mapping_dict.items():
                self.forcecontainer_list[force_group_idx_new] = forcecontainer_list_old[force_group_idx_old]
                valids = np.where(self.force_group_idx_list == force_group_idx_new)[0]
                force_group_idx_list[valids] = force_group_idx_old
            self.force_group_idx_list = force_group_idx_list

    def get_parameter(
        self, 
        force_group_idx: int, 
        name: str):

        return self.forcecontainer_list[force_group_idx].get_parameter(name)

    def set_parameter(
        self, 
        force_group_idx: int, 
        name: str, 
        value: _UNIT_QUANTITY):

        __doc__ = """
        Set the parameter of name `name` of force group `force_group_idx`
        to value `value`.
        """

        idx_list = np.where(
                self.force_group_idx_list == force_group_idx
                )[0]
        if force_group_idx == _INACTIVE_GROUP_IDX:
            forcecontainer = self.inactive_forcecontainer
        else:
            forcecontainer = self.forcecontainer_list[force_group_idx]
        force_entity_list = list()
        for idx in idx_list:
            force_entity = self.force_entity_list[idx]
            system_idx   = self.system_idx_list[idx]
            system       = self.system_list[system_idx]
            force_idx    = self.omm_force_list[system_idx]
            force        = system.openmm_system.getForce(force_idx)
            atom_list    = self.atom_list[idx]
            forcecontainer.set_parameter(
                name, 
                value, 
                force, 
                [force_entity],
                [atom_list])
            system.parms_changed = True

    def relocate_by_rank(
        self, 
        force_rank: int, 
        force_group_idx: int, 
        exclude_list: list = list()
        ) -> None:

        __doc__ = """
        Allocates all force entities of rank `force_rank` to force group
        `force_group_idx`.
        """

        idx_list = np.where(
                self.force_ranks == force_rank
                )[0]
        if force_group_idx == _INACTIVE_GROUP_IDX:
            forcecontainer = self.inactive_forcecontainer
        else:
            forcecontainer = self.forcecontainer_list[force_group_idx]
        self.force_group_idx_list[idx_list] = force_group_idx
        for idx in idx_list:
            force_entity = self.force_entity_list[idx]
            system_idx   = self.system_idx_list[idx]
            system       = self.system_list[system_idx]
            force_idx    = self.omm_force_list[system_idx]
            force        = system.openmm_system.getForce(force_idx)
            atom_list    = self.atom_list[idx]
            for name in forcecontainer.parm_names:
                if name in exclude_list:
                    continue
                value = forcecontainer.get_parameter(name)
                forcecontainer.set_parameter(
                    name, 
                    value, 
                    force, 
                    [force_entity],
                    [atom_list])
            system.parms_changed = True

    def are_force_group_same(
        self, 
        force_group_idx1: int, 
        force_group_idx2: int) -> bool:

        __doc__ = """
        Check if two force groups `force_group_idx1` and 
        `force_group_idx2` are identical.
        """

        if not force_group_idx1 in self.force_group_idx_list:
            raise ValueError("force_group_idx1 must be in self.force_group_idx_list")
        if not force_group_idx2 in self.force_group_idx_list:
            raise ValueError("force_group_idx2 must be in self.force_group_idx_list")

        if force_group_idx1 == force_group_idx2:
            return True
        forcecontainer_1 = self.forcecontainer_list[force_group_idx1]
        forcecontainer_2 = self.forcecontainer_list[force_group_idx2]
        if forcecontainer_1 == forcecontainer_2:
            return True
        return False

    def is_force_group_empty(
        self, 
        force_group_idx: int) -> bool:

        __doc__ = """
        Check if a force group `force_group_idx` is empty or not.
        """

        valids   = np.where(force_group_idx == self.force_group_idx_list)[0]
        is_empty = False
        if valids.size == 0:
            is_empty = True
        return is_empty

    def remove_force_group(
        self, 
        force_group_idx: int) -> None:

        __doc__ = """
        Remove force group `force_group_idx` from this ParameterManager.
        This does *not* mean that the force is deleted from the OpenMM system.
        """

        if force_group_idx == _INACTIVE_GROUP_IDX:
            raise ValueError("Cannot remove the inactive group.")

        if not self.is_force_group_empty(force_group_idx):
            raise ValueError(
                f"Cannot remove force group {force_group_idx}. Force group is not empty."
                )

        del self.forcecontainer_list[force_group_idx]
        valids_gt = np.where(self.force_group_idx_list > force_group_idx)
        self.force_group_idx_list[valids_gt] -= 1

    def add_force_group(
        self, 
        input_parms_dict: dict = dict()
        ) -> None:

        if not isinstance(input_parms_dict, dict):
            raise ValueError(
                f"`input_parms_dict` must be type `dict`, but is type {type(input_parms_dict)}"
                )

        default_parms_dict = copy.deepcopy(self.default_parms)
        for name in self.forcecontainer.parm_names:
            if name in input_parms_dict:
                default_parms_dict[name] = input_parms_dict[name]

        forcecontainer = self.forcecontainer(
            default_parms_dict, 
            self.exclude_list
            )

        assert len(default_parms_dict) == forcecontainer.N_parms

        self.forcecontainer_list.append(forcecontainer)

    def add_system_manager(
        self,
        system_manager: SystemManager) -> bool:

        for system in system_manager.get_systems():
            self.add_system(system)

        return True

    def add_system(
        self, 
        system: System) -> bool:

        found_force = False
        for force_idx, force in enumerate(system.openmm_system.getForces()):
            if force.__class__.__name__ == self.force_tag:
                found_force = True
                break

        ### If, for whatever reason, the force tag is not part of the
        ### system object, we cannot procede here.
        if not found_force:
            raise Exception(
                f"Could not find force {self.force_tag} in system {system.name}.")

        self.omm_force_list.append(force_idx)
        self.system_list.append(system)
        import copy
        self.rdmol_list.append(
            copy.deepcopy(system.rdmol)
            )
        ### 1.) Organize FF Parameters, check-out symmetry and ranks
        ### ========================================================
        ###
        ### Load parameters and generate a list of forcecontainer
        ### objects from the system.
        ### This list should have one forcecontainer for each
        ### force entity.
        forcecontainer_list,\
        atom_list,\
        force_entity_list = self.build_forcecontainer_list(
            force, 
            system
            )
        if self.force_entity_list.size == 0:
            _force_entity_list = np.array(
                force_entity_list,
                dtype=int
                )
        else:
            _force_entity_list = np.append(
                self.force_entity_list, 
                force_entity_list,
                axis=0
                ).astype(int)
        self.force_entity_list = _force_entity_list

        ### Now add them into forcegroups
        N_force_entites = len(forcecontainer_list)
        force_ranks     = list()
        for force_entity in range(N_force_entites):
            force_group_idx_new = None

            forcecontainer_new = forcecontainer_list[force_entity]
            found_forcecontainer_match = False
            if forcecontainer_new.is_active:
                for force_group_idx in self.unique_force_group_idx_list:
                    if force_group_idx == _INACTIVE_GROUP_IDX:
                        continue
                    forcecontainer_old = self.forcecontainer_list[force_group_idx]
                    if forcecontainer_new == forcecontainer_old:
                        force_group_idx_new = force_group_idx
                        found_forcecontainer_match = True
                        break
                if not found_forcecontainer_match:
                    ### Must set `force_group_idx_new` before appending
                    ### to `self.forcecontainer_list`
                    force_group_idx_new = len(self.forcecontainer_list)
                    self.forcecontainer_list.append(forcecontainer_new)
            else:
                force_group_idx_new = _INACTIVE_GROUP_IDX

            force_ranks.append(-1)

            ### We *must* update `self.force_group_idx_list` at the end of each
            ### iteration in order to update `self.unique_force_group_idx_list`.
            _force_group_idx_list = np.append(
                self.force_group_idx_list,
                force_group_idx_new).astype(int)
            self.force_group_idx_list = _force_group_idx_list

            _system_idx_list = np.append(
                self.system_idx_list,
                self._N_systems).astype(int)
            self.system_idx_list = _system_idx_list

        ### Here we make sure that forces that are related through symmetry
        ### are grouped together in the self.force_ranks list.
        for force_entity_1 in range(N_force_entites):
            if force_ranks[force_entity_1] != -1:
                continue
            atom_idxs_1  = atom_list[force_entity_1]
            atom_ranks_1 = tuple([system.ranks[a_idx] for a_idx in atom_idxs_1])
            next_rank    = max(self.max_rank, *force_ranks) + 1
            force_ranks[force_entity_1] = next_rank
            for force_entity_2 in range(force_entity_1 + 1, N_force_entites):
                if force_ranks[force_entity_2] != -1:
                    continue
                atom_idxs_2  = atom_list[force_entity_2]
                atom_ranks_2 = [system.ranks[a_idx] for a_idx in atom_idxs_2]
                found_forcecontainer_match = False
                for atom_ranks_2_perm in list(permutations(atom_ranks_2)):
                    if atom_ranks_2_perm == atom_ranks_1:
                        found_forcecontainer_match = True
                        break
                if found_forcecontainer_match:
                    force_ranks[force_entity_2] = next_rank

        force_ranks  = np.array(force_ranks, dtype=int)
        _force_ranks = np.append(
            self.force_ranks, 
            force_ranks).astype(int)
        self.force_ranks = _force_ranks
        ### Check that all force ranks have been matched.
        assert np.all(self.force_ranks > -1)

        atomic_num_list = list()
        for idxs in atom_list:
            for idx in idxs:
                rdatom = system.rdmol.GetAtomWithIdx(idx)
                atomic_num_list.append(rdatom.GetAtomicNum())
        self.atomic_num_list.extend(atomic_num_list)

        self.atom_list.extend(atom_list)

        ### In the last step increase the number of systems
        ### and add the system
        self._N_systems += 1

        return True


class LJC14ScaleManager(ParameterManager):

    def __init__(self, exclude_list=list()):

        super().__init__(
            forcecontainer = LJC14ScaleContainer,
            exclude_list = exclude_list,
            )

        self.force_tag = "NonbondedForce"
        self.name      = "LJQ14ScaleManager"

        self.default_parms = {
            "coulomb14Scale" : 5./6.,
            "lj14Scale"      : 5./6.,
        }

        ### The inactive force
        self.inactive_forcecontainer = self.forcecontainer(
            {
            "coulomb14Scale" : 1.,
            "lj14Scale"      : 1.,
            },
            exclude_list,
            )
        self.inactive_forcecontainer.is_active = False

    def build_forcecontainer_list(
        self, 
        force: openmm.Force, 
        system: System) -> list:

        __doc__ = """
        Takes an openmm force object and a system object as input
        and builds a list with all forces packed into force containers.
        It will not only build force containers for forces that already
        exist in the openmm force object, but also for the ones that are
        in principle possible according to the topology in system. The latter
        will automatically be set to inactive (i.e. they are not contributing
        to the potential energy of the system), whereas the former are set as
        activate (i.e. they will contribute to the potential energy of the 
        system).
        """

        forcecontainer_list = list()
        atom_list           = list()
        force_entity_list   = list()

        pair14_list = list()
        for atm_idxs in system.proper_dihedrals:
            p1 = atm_idxs[0]
            p2 = atm_idxs[3]
            pair14 = sorted([p1,p2])
            pair14_list.append(pair14)

        ### Loop over all bonds already present in force object
        ### and add force containers accordingly into force container list.
        for ljq14_idx in range(force.getNumExceptions()):
            p1, p2, q12, sig12, eps12 = force.getExceptionParameters(ljq14_idx)
            pair14 = sorted([p1,p2])
            if not pair14 in pair14_list:
                continue
            q1, sig1, eps1 = force.getParticleParameters(p1)
            q2, sig2, eps2 = force.getParticleParameters(p2)

            coulomb14Scale = q12 / (q1 * q2)
            lj14Scale      = eps12 / np.sqrt(eps1 * eps2)

            forcecontainer_list.append(
                self.forcecontainer(
                    { 
                        'coulomb14Scale' : float(coulomb14Scale),
                        'lj14Scale'      : float(lj14Scale)
                    },
                    self.exclude_list,
                    )
                )
            atom_list.append(pair14)
            force_entity_list.append(ljq14_idx)

        ### Second, loop over all *actual* 1-4 exceptions and build
        ### exceptions accordingly.
        for pair14 in pair14_list:
            if pair14 in atom_list or\
               pair14[::-1] in atom_list:
               continue
            else:
                q1, sig1, eps1 = force.getParticleParameters(p1)
                q2, sig2, eps2 = force.getParticleParameters(p2)

                q12   = q1 * q2
                sig12 = (sig1 + sig2) / 2.
                eps12 = np.sqrt(eps1 * eps2)

                ljq14_idx = force.addException(
                    particle1 = pair14[0],
                    particle2 = pair14[1],
                    chargeProd = q12,
                    sigma = sig12,
                    epsilon = eps12)
                forcecontainer_list.append(
                    self.forcecontainer(
                        { 
                            'coulomb14Scale' : 1.,
                            'lj14Scale'      : 1.,
                        },
                        self.exclude_list,
                        )
                    )
                forcecontainer_list[-1].deactivate()
                atom_list.append(pair14)
                force_entity_list.append(ljq14_idx)

        return forcecontainer_list, atom_list, force_entity_list


class BondManager(ParameterManager):

    def __init__(self, exclude_list = list()):

        super().__init__(
            forcecontainer = BondContainer,
            exclude_list = exclude_list,
            )

        self.force_tag = "HarmonicBondForce"
        self.name      = "BondManager"

        self.default_parms = {
            "k"      : 0. * _FORCE_CONSTANT_BOND,
            "length" : 0. * _LENGTH
        }

        ### The inactive force
        self.inactive_forcecontainer = self.forcecontainer(
            {
                "k"      : 0. * _FORCE_CONSTANT_BOND,
                "length" : 0. * _LENGTH
            },
            exclude_list,
            )
        self.inactive_forcecontainer.is_active = False

    def build_forcecontainer_list(
        self, 
        force: openmm.Force, 
        system: System) -> list:

        __doc__ = """
        Takes an openmm force object and a system object as input
        and builds a list with all forces packed into force containers.
        It will not only build force containers for forces that already
        exist in the openmm force object, but also for the ones that are
        in principle possible according to the topology in system. The latter
        will automatically be set to inactive (i.e. they are not contributing
        to the potential energy of the system), whereas the former are set as
        activate (i.e. they will contribute to the potential energy of the 
        system).
        """

        forcecontainer_list = list()
        atom_list           = list()
        force_entity_list   = list()
        ### Loop over all bonds already present in force object
        ### and add force containers accordingly into force container list.
        for bond_idx in range(force.getNumBonds()):
            parms    = force.getBondParameters(bond_idx)
            atm_idxs = parms[:2]
            forcecontainer_list.append(
                self.forcecontainer(
                    { 
                        'length' : parms[2],
                        'k'      : parms[3],
                    },
                    self.exclude_list,
                    )                    
                )
            atom_list.append(atm_idxs)
            force_entity_list.append(bond_idx)

        ### Second, loop over all *actual* bonds and build
        ### force objects accordingly.
        ### Note these can include atoms that are bonded
        ### but have no bond force present.
        for atm_idxs in system.bonds:
            if atm_idxs in atom_list or\
               atm_idxs[::-1] in atom_list:
               continue
            else:
                bond_idx = force.addBond(
                    atm_idxs[0],
                    atm_idxs[1],
                    0. * _LENGTH,
                    0. * _FORCE_CONSTANT_BOND)
                forcecontainer_list.append(
                    self.forcecontainer(
                        { 
                            'length' : 0. * _LENGTH,
                            'k'      : 0. * _FORCE_CONSTANT_BOND,
                        },
                        self.exclude_list,
                        )
                    )
                forcecontainer_list[-1].deactivate()
                atom_list.append(atm_idxs)
                force_entity_list.append(bond_idx)

        return forcecontainer_list, atom_list, force_entity_list


class AngleManager(ParameterManager):

    def __init__(self, exclude_list = list()):

        super().__init__(
            forcecontainer = AngleContainer,
            exclude_list = exclude_list,
            )

        self.force_tag = "HarmonicAngleForce"
        self.name      = "AngleManager"

        self.default_parms = {
            "k"     : 0. * _FORCE_CONSTANT_ANGLE,
            "angle" : 0. * _ANGLE
        }

        ### The inactive force
        self.inactive_forcecontainer = self.forcecontainer(
            {
                "k"     : 0. * _FORCE_CONSTANT_ANGLE,
                "angle" : 0. * _ANGLE
            },
            exclude_list,
            )
        self.inactive_forcecontainer.is_active = False

    def build_forcecontainer_list(
        self, 
        force: openmm.Force, 
        system: System) -> list:

        forcecontainer_list = list()
        atom_list           = list()
        force_entity_list   = list()
        ### Loop over all angles already present in force object
        ### and add force containers accordingly into force container list.
        for angle_idx in range(force.getNumAngles()):
            parms    = force.getAngleParameters(angle_idx)
            atm_idxs = parms[:3]
            forcecontainer_list.append(
                self.forcecontainer(
                    {
                        'angle' : parms[3],
                        'k'     : parms[4],
                    },
                    self.exclude_list,
                    )
                )
            atom_list.append(atm_idxs)
            force_entity_list.append(angle_idx)

        ### Second, loop over all *actual* angles and build
        ### force objects accordingly.
        for atm_idxs in system.angles:
            if atm_idxs in atom_list or\
               atm_idxs[::-1] in atom_list:
               continue
            else:
                angle_idx = force.addAngle(
                    atm_idxs[0],
                    atm_idxs[1],
                    atm_idxs[2],
                    0. * _ANGLE,
                    0. * _FORCE_CONSTANT_ANGLE)
                forcecontainer_list.append(
                    self.forcecontainer(
                        {
                            'angle' : 0. * _ANGLE,
                            'k'     : 0. * _FORCE_CONSTANT_ANGLE,
                        },
                        self.exclude_list,
                        )
                    )
                forcecontainer_list[-1].deactivate()
                atom_list.append(atm_idxs)
                force_entity_list.append(angle_idx)

        return forcecontainer_list, atom_list, force_entity_list


class DoubleProperTorsionManager(ParameterManager):

    def __init__(self, periodicity, exclude_list=list()):

        super().__init__(
            forcecontainer = DoubleProperTorsionContainer,
            exclude_list = exclude_list,
            )

        self.force_tag   = "PeriodicTorsionForce"
        self.name        = "DoubleProperTorsionManager"
        self.periodicity = periodicity

        self.phase_list  = [0. * unit.radian, 0.5 * np.pi * unit.radian]

        self.default_parms = {
            "periodicity" : self.periodicity,
            "k0"          : 0. * _FORCE_CONSTANT_TORSION,
            "k1"          : 0. * _FORCE_CONSTANT_TORSION,
        }

        ### The inactive force
        self.inactive_forcecontainer = self.forcecontainer(
            {
                'periodicity' : self.periodicity,
                'k0'          : 0. * _FORCE_CONSTANT_TORSION,
                'k1'          : 0. * _FORCE_CONSTANT_TORSION,
            },
            exclude_list,
            )
        self.inactive_forcecontainer.is_active = False

    def build_forcecontainer_list(
        self, 
        force: openmm.Force, 
        system: System) -> list:

        forcecontainer_list = list()
        atom_list           = list()
        force_entity_list   = list()

        dihedrals_dict = dict()
        for dihedral_idx in range(force.getNumTorsions()):
            parms = force.getTorsionParameters(dihedral_idx)
            if self.periodicity != parms[4]:
                continue
            check0 = abs(parms[5] - self.phase_list[0]) < 1.e-2 * unit.radian
            check1 = abs(parms[5] - self.phase_list[1]) < 1.e-2 * unit.radian
            if not (check0 or check1):
                continue
            assert check0 != check1
            atm_idxs  = parms[:4]
            is_proper = True
            for i in range(3):
                neighbor_list = system.get_neighbor_atomidxs(atm_idxs[i])
                if not atm_idxs[i+1] in neighbor_list:
                    is_proper = False
                    break
            if is_proper:
                if check0:
                    check0 = dihedral_idx
                    check1 = None
                if check1:
                    check0 = None
                    check1 = dihedral_idx
                atm_idxs = tuple(atm_idxs)
                if atm_idxs[::-1] in dihedrals_dict:
                    atm_idxs = atm_idxs[::-1]
                if atm_idxs in dihedrals_dict:
                    _check0, _check1 = dihedrals_dict[atm_idxs]
                    if _check0 == None:
                        _check0 = check0
                    if _check1 == None:
                        _check1 = check1
                    dihedrals_dict[atm_idxs] = (_check0, _check1)
                else:
                    dihedrals_dict[atm_idxs] = (check0, check1)

        ### Second, loop over all *actual* dihedrals and build
        ### force objects accordingly.
        for atm_idxs in system.proper_dihedrals:
            is_proper = True
            for i in range(3):
                neighbor_list = system.get_neighbor_atomidxs(atm_idxs[i])
                if not atm_idxs[i+1] in neighbor_list:
                    is_proper = False
                    break
            if not is_proper:
                continue
            atm_idxs = tuple(atm_idxs)
            if atm_idxs in dihedrals_dict or atm_idxs[::-1] in dihedrals_dict:
                if atm_idxs[::-1] in dihedrals_dict:
                    atm_idxs = atm_idxs[::-1]
                check0, check1 = dihedrals_dict[atm_idxs]
                dihedral_idx = None
                if check0 == None:
                    parms = force.getTorsionParameters(check1)
                    k1 = parms[6]
                    k0 = k1 * 0.
                    dihedral_idx   = check1
                    dihedral_idx_1 = check1
                    dihedral_idx_0 = force.addTorsion(
                        atm_idxs[0],
                        atm_idxs[1],
                        atm_idxs[2],
                        atm_idxs[3],
                        self.periodicity,
                        self.phase_list[0],
                        0. * _FORCE_CONSTANT_TORSION
                        )
                elif check1 == None:
                    parms = force.getTorsionParameters(check0)
                    k0 = parms[6]
                    k1 = k0 * 0.
                    dihedral_idx   = check0
                    dihedral_idx_0 = check0
                    dihedral_idx_1 = force.addTorsion(
                        atm_idxs[0],
                        atm_idxs[1],
                        atm_idxs[2],
                        atm_idxs[3],
                        self.periodicity,
                        self.phase_list[1],
                        0. * _FORCE_CONSTANT_TORSION
                        )
                else:
                    dihedral_idx   = check0
                    dihedral_idx_0 = check0
                    dihedral_idx_1 = check1
                    parms0 = force.getTorsionParameters(check0)
                    parms1 = force.getTorsionParameters(check1)
                    k0 = parms0[6]
                    k1 = parms1[6]

                forcecontainer = self.forcecontainer(
                    {
                        'periodicity' : self.periodicity,
                        'k0'          : k0,
                        'k1'          : k1,
                    },
                    self.exclude_list,
                    )

                forcecontainer_list.append(
                    forcecontainer
                    )

                if k0 == 0. * _FORCE_CONSTANT_TORSION and k1 == 0. * _FORCE_CONSTANT_TORSION:
                    forcecontainer_list[-1].deactivate()

                atom_list.append(atm_idxs)
                force_entity_list.append([dihedral_idx_0, dihedral_idx_1])

            else:
                k0 = 0. * _FORCE_CONSTANT_TORSION
                k1 = 0. * _FORCE_CONSTANT_TORSION
                dihedral_idx_0 = force.addTorsion(
                    atm_idxs[0],
                    atm_idxs[1],
                    atm_idxs[2],
                    atm_idxs[3],
                    self.periodicity,
                    self.phase_list[0],
                    0. * _FORCE_CONSTANT_TORSION
                    )
                dihedral_idx_1 = force.addTorsion(
                    atm_idxs[0],
                    atm_idxs[1],
                    atm_idxs[2],
                    atm_idxs[3],
                    self.periodicity,
                    self.phase_list[1],
                    0. * _FORCE_CONSTANT_TORSION
                    )

                forcecontainer = self.forcecontainer(
                    {
                        'periodicity' : self.periodicity,
                        'k0'          : k0,
                        'k1'          : k1,
                    },
                    self.exclude_list,
                    )
                forcecontainer_list.append(
                    forcecontainer
                    )
                forcecontainer_list[-1].deactivate()

                atom_list.append(atm_idxs)
                force_entity_list.append([dihedral_idx_0, dihedral_idx_1])

        return forcecontainer_list, atom_list, force_entity_list


class ProperTorsionManager(ParameterManager):

    def __init__(self, periodicity, phase, exclude_list=list()):

        super().__init__(
            forcecontainer = ProperTorsionContainer,
            exclude_list = exclude_list,
            )

        self.force_tag   = "PeriodicTorsionForce"
        self.name        = "ProperTorsionManager"
        self.periodicity = periodicity
        self.phase       = phase

        self.default_parms = {
            "periodicity" : self.periodicity,
            "phase"       : self.phase,
            "k"           : 0. * _FORCE_CONSTANT_TORSION,
        }

        ### The inactive force
        self.inactive_forcecontainer = self.forcecontainer(
            {
                'periodicity' : self.periodicity,
                'phase'       : self.phase,
                'k'           : 0. * _FORCE_CONSTANT_TORSION,
            },
            exclude_list,
            )
        self.inactive_forcecontainer.is_active = False

    def build_forcecontainer_list(
        self, 
        force: openmm.Force, 
        system: System) -> list:

        forcecontainer_list = list()
        atom_list           = list()
        force_entity_list   = list()
        ### Loop over all dihedrals already present in force object
        ### and add force containers accordingly into force container list.
        for dihedral_idx in range(force.getNumTorsions()):
            parms = force.getTorsionParameters(dihedral_idx)
            if self.periodicity != parms[4]:
                continue
            if self.phase != parms[5]:
                continue
            atm_idxs  = parms[:4]
            is_proper = True
            for i in range(3):
                neighbor_list = system.get_neighbor_atomidxs(atm_idxs[i])
                if not atm_idxs[i+1] in neighbor_list:
                    is_proper = False
                    break
            if is_proper:
                forcecontainer_list.append(
                    self.forcecontainer(
                        {
                            'periodicity' : parms[4],
                            'phase'       : parms[5],
                            'k'           : parms[6],
                        },
                        self.exclude_list,
                        )
                    )
                if parms[6]._value == 0.:
                    forcecontainer_list[-1].deactivate()
                atom_list.append(atm_idxs)
                force_entity_list.append(dihedral_idx)

        ### Second, loop over all *actual* dihedrals and build
        ### force objects accordingly.
        for atm_idxs in system.proper_dihedrals:
            if atm_idxs in atom_list or\
               atm_idxs[::-1] in atom_list:
               continue
            else:
                is_proper = True
                for i in range(3):
                    neighbor_list = system.get_neighbor_atomidxs(atm_idxs[i])
                    if not atm_idxs[i+1] in neighbor_list:
                        is_proper = False
                        break
                if is_proper:
                    dihedral_idx = force.addTorsion(
                        atm_idxs[0],
                        atm_idxs[1],
                        atm_idxs[2],
                        atm_idxs[3],
                        self.periodicity,
                        self.phase,
                        0. * _FORCE_CONSTANT_TORSION
                        )
                    forcecontainer_list.append(
                        self.forcecontainer(
                            {
                                'periodicity' : self.periodicity,
                                'phase'       : self.phase,
                                'k'           : 0. * _FORCE_CONSTANT_TORSION,
                            },
                            self.exclude_list,
                            )
                        )
                    forcecontainer_list[-1].deactivate()
                    atom_list.append(atm_idxs)
                    force_entity_list.append(dihedral_idx)

        return forcecontainer_list, atom_list, force_entity_list


class MultiProperTorsionManager(ParameterManager):

    def __init__(self, periodicity_list, phase_list, exclude_list=list()):

        if not len(periodicity_list) == len(phase_list):
            raise ValueError(
                "`periodicity_list` and `phase_list` must have same length."
                )

        import copy
        MultiProperTorsionContainer_cp = copy.deepcopy(MultiProperTorsionContainer)
        MultiProperTorsionContainer_cp.N_parms = len(phase_list)

        super().__init__(
            forcecontainer = MultiProperTorsionContainer_cp,
            exclude_list = exclude_list,
            )

        self.force_tag   = "PeriodicTorsionForce"
        self.name        = "MultiProperTorsionManager"
        self.periodicity = periodicity_list
        self.phase       = phase_list,

        self.periodicity_list = periodicity_list
        self.phase_list       = phase_list

        self.parm_names = list()
        self.default_parms = {}
        inactive_dict = {}
        counts = 0
        for periodicity, phase in zip(periodicity_list, phase_list):
            self.default_parms.update(
                {
                    f"periodicity{counts}" : periodicity,
                    f"phase{counts}"       : phase,
                    f"k{counts}"           : 0. * _FORCE_CONSTANT_TORSION,
                }
                )
            inactive_dict.update(
                {
                    f"periodicity{counts}" : periodicity,
                    f"phase{counts}"       : phase,
                    f"k{counts}"           : 0. * _FORCE_CONSTANT_TORSION,
                }
                )

            self.parm_names.append(f"periodicity{counts}")
            self.parm_names.append(f"phase{counts}")
            self.parm_names.append(f"k{counts}")

            counts += 1

        ### The inactive force
        self.inactive_forcecontainer = self.forcecontainer(inactive_dict)
        self.inactive_forcecontainer.is_active = False

    def build_forcecontainer_list(
        self, 
        force: openmm.Force, 
        system: System) -> list:

        initial_atom_dict = dict()
        for dihedral_idx in range(force.getNumTorsions()):
            parms = force.getTorsionParameters(dihedral_idx)
            atm_idxs = list(parms[:4])
            if atm_idxs in system.proper_dihedrals:
                is_proper = True
                for i in range(3):
                    neighbor_list = system.get_neighbor_atomidxs(atm_idxs[i])
                    if not atm_idxs[i+1] in neighbor_list:
                        is_proper = False
                        break
                if not is_proper:
                    continue
            atm_idxs = tuple(atm_idxs)
            _periodicity = parms[4]
            _phase = parms[5]
            _k = parms[6]
            key1 = (atm_idxs,       _periodicity, _phase.value_in_unit(_ANGLE))
            key2 = (atm_idxs[::-1], _periodicity, _phase.value_in_unit(_ANGLE))
            initial_atom_dict[key1] = (dihedral_idx, _k)
            initial_atom_dict[key2] = (dihedral_idx, _k)

        forcecontainer_list = list()
        atom_list           = list()
        force_entity_list   = list()
        ### Loop over all dihedrals already present in force object
        ### and add force containers accordingly into force container list.
        for atm_idxs in system.proper_dihedrals:
            is_proper = True
            for i in range(3):
                neighbor_list = system.get_neighbor_atomidxs(atm_idxs[i])
                if not atm_idxs[i+1] in neighbor_list:
                    is_proper = False
                    break
            if not is_proper:
                continue
            atm_idxs = tuple(atm_idxs)
            parms_dict = {}
            counts = 0
            dihedral_idx_list = list()
            for _, _ in zip(self.periodicity_list, self.phase_list):
                parms_dict[f"periodicity{counts}"] = self.default_parms[f"periodicity{counts}"]
                parms_dict[f"phase{counts}"] = self.default_parms[f"phase{counts}"]
                periodicity = parms_dict[f"periodicity{counts}"]
                phase = parms_dict[f"phase{counts}"]
                key = (atm_idxs, periodicity, phase.value_in_unit(_ANGLE))
                if key in initial_atom_dict:
                    dihedral_idx, k = initial_atom_dict[key]
                else:
                    dihedral_idx = None
                    k = 0. * _FORCE_CONSTANT_TORSION
                parms_dict[f"k{counts}"] = k

                if dihedral_idx == None:
                    dihedral_idx = force.addTorsion(
                        atm_idxs[0],
                        atm_idxs[1],
                        atm_idxs[2],
                        atm_idxs[3],
                        periodicity,
                        phase,
                        0. * _FORCE_CONSTANT_TORSION
                        )
                dihedral_idx_list.append(dihedral_idx)

                counts += 1

            forcecontainer_list.append(
                self.forcecontainer(
                    parms_dict,
                    self.exclude_list,
                    )
                )

            atom_list.append(atm_idxs)
            force_entity_list.append(dihedral_idx_list)

        return forcecontainer_list, atom_list, force_entity_list


class ImproperTorsionManager(ParameterManager):

    def __init__(self, periodicity, phase, exclude_list=list()):

        super().__init__(
            forcecontainer = ImproperTorsionContainer,
            exclude_list = exclude_list,
            )

        ### Afaik, the Improper Torsion are stored in the Periodic forces.
        ### At least with the currrent workflow involving off-toolkit

        self.force_tag   = "PeriodicTorsionForce"
        self.name        = "ImproperTorsionManager"
        self.periodicity = periodicity
        self.phase       = phase

        self.default_parms = {
            "periodicity" : self.periodicity,
            "phase"       : self.phase,
            "k"           : 0. * _FORCE_CONSTANT_TORSION,
        }

        ### The inactive force
        self.inactive_forcecontainer = self.forcecontainer(
            {
                'periodicity' : self.periodicity,
                'phase'       : self.phase,
                'k'           : 0. * _FORCE_CONSTANT_TORSION,
            },
            exclude_list,
            )
        self.inactive_forcecontainer.is_active = False

    def build_forcecontainer_list(
        self, 
        force: openmm.Force, 
        system: System) -> list:

        forcecontainer_list = list()
        atom_list           = list()
        force_entity_list   = list()
        ### Loop over all dihedrals already present in force object
        ### and add force containers accordingly into force container list.
        for dihedral_idx in range(force.getNumTorsions()):
            parms    = force.getTorsionParameters(dihedral_idx)
            if self.periodicity != parms[4]:
                continue
            if self.phase != parms[5]:
                continue
            atm_idxs  = parms[:4]
            is_proper = True
            for i in range(3):
                neighbor_list = system.get_neighbor_atomidxs(atm_idxs[i])
                if not atm_idxs[i+1] in neighbor_list:
                    is_proper = False
                    break
            if not is_proper:
                forcecontainer_list.append(
                    self.forcecontainer(
                        {
                            'periodicity' : parms[4],
                            'phase'       : parms[5],
                            'k'           : parms[6],
                        },
                        self.exclude_list,
                        )
                    )
                if parms[6]._value == 0.:
                    forcecontainer_list[-1].deactivate()
                central_atom = atm_idxs[0]
                outer_atoms  = atm_idxs[1:]
                #outer_atoms  = sorted(outer_atoms)
                atm_idxs     = [central_atom] + outer_atoms
                atom_list.append(atm_idxs)
                force_entity_list.append(dihedral_idx)

        ### Second, loop over all *actual* dihedrals and build
        ### force objects accordingly.
        for atm_idxs in system.improper_dihedrals:
            if atm_idxs in atom_list:
               continue
            else:
                is_proper = True
                for i in range(3):
                    neighbor_list = system.get_neighbor_atomidxs(atm_idxs[i])
                    if not atm_idxs[i+1] in neighbor_list:
                        is_proper = False
                        break
                if not is_proper:
                    dihedral_idx = force.addTorsion(
                        atm_idxs[0],
                        atm_idxs[1],
                        atm_idxs[2],
                        atm_idxs[3],
                        self.periodicity,
                        self.phase,
                        0. * _FORCE_CONSTANT_TORSION)
                    forcecontainer_list.append(
                        self.forcecontainer(
                            {
                                'periodicity' : self.periodicity,
                                'phase'       : self.phase,
                                'k'           : 0. * _FORCE_CONSTANT_TORSION,
                            },
                            self.exclude_list,
                            )
                        )
                    forcecontainer_list[-1].deactivate()
                    atom_list.append(atm_idxs)
                    force_entity_list.append(dihedral_idx)

        return forcecontainer_list, atom_list, force_entity_list


def are_containers_same(
    container1: ForceTypeContainer, 
    container2: ForceTypeContainer, 
    relax_list: list = list()
    ) -> bool:

    if not "N_parms" in relax_list:
        if container1.N_parms != container2.N_parms:
            return False

    if not "N_atoms" in relax_list:
        if container1.N_atoms != container2.N_atoms:
            return False

    if not "type" in relax_list:
        if type(container1) != type(container2):
            return False

    if not "parm_name" in relax_list:
        for parm_name in container1.parm_names:
            if not parm_name in container2.parm_names:
                return False

        for parm_name in container2.parm_names:
            if not parm_name in container1.parm_names:
                return False

    if not "parm_value" in relax_list:
        for parm_name in container2.parm_names:
            if parm_name in container1.exclude_list:
                continue
            if parm_name in container2.exclude_list:
                continue
            parm1_value = container1.get_parameter(parm_name)
            parm2_value = container2.get_parameter(parm_name)
            diff        = abs(parm1_value-parm2_value)
            if type(parm1_value) != _UNIT_QUANTITY:
                ### Sometimes parameter values are "0"
                if abs(parm1_value) < 1e-12:
                    if diff > 1e-4:
                        return False
                elif diff/abs(parm1_value) > 1e-4:
                    return False
            else:
                if abs(parm1_value) < 1e-12*parm1_value.unit:
                    if diff > 1e-4*diff.unit:
                        return False
                elif diff/abs(parm1_value) > 1e-4:
                    return False

    return True