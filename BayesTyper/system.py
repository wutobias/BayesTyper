#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================
from rdkit import Chem

import networkx as nx
from networkx.algorithms import isomorphism

import copy
import openmm
from openmm import unit

from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField

import numpy as np

import warnings

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================
from .constants import _UNIT_QUANTITY, _DEFAULT_FF, _ATOMIC_MASS
noWarnings = False

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================


class System(object):

    def __init__(
        self,
        name: str,
        offmol: Molecule,
        openmm_system: openmm.openmm.System,
        top: Topology):

        self._name   = name
        self._offmol = offmol
        self._top    = top

        ### This should created no later than here
        self.openmm_system      = openmm.XmlSerializer.serialize(openmm_system)
        self.openmm_system_init = self.openmm_system
        self._N_atoms           = openmm_system.getNumParticles()

        self._force_dict         = {
                f.__class__.__name__ : idx for idx, f in enumerate(openmm_system.getForces())}

        self.reset_system(
            reset_targets=True,
            reset_nxmol=True)

        self.parms_changed = True

        self.masses = list()
        for atm_idx in range(self.N_atoms):
            rdatom = self.rdmol.GetAtomWithIdx(atm_idx)
            self.masses.append(float(rdatom.GetMass()))
        self.masses = np.array(self.masses) * _ATOMIC_MASS

        self._N_tgt = 0

    @property
    def force_dict(self):
        
        ### Hack for backwards compatibility
        if hasattr(self, "_force_dict"):
            return self._force_dict
        else:
            openmm_system = openmm.XmlSerializer.deserialize(
                    self.openmm_system)
            self._force_dict         = {
                    f.__class__.__name__ : idx for idx, f in enumerate(openmm_system.getForces())}
            return self._force_dict

    @property
    def N_tgt(self):
        return self._N_tgt

    @property
    def rdmol_setatommap(self) -> bool:
        return self._rdmol_setatommap

    @rdmol_setatommap.setter
    def rdmol_setatommap(self, value: bool) -> None:
        self._rdmol_setatommap = value
        if self._rdmol_setatommap:
            for i, a in enumerate(self.rdmol.GetAtoms()):
                a.SetAtomMapNum(i)
        else:
            for i, a in enumerate(self.rdmol.GetAtoms()):
                a.SetAtomMapNum(0)

    @property
    def N_atoms(self):
        return self._N_atoms

    @property
    def name(self):
        return self._name

    @property
    def offmol(self):
        return self._offmol

    @property
    def top(self):
        return self._top

    @property
    def bonds(self):
        return self._bond_list

    @property
    def angles(self):
        return self._angle_list

    @property
    def proper_dihedrals(self):
        return self._proper_dihedral_list

    @property
    def improper_dihedrals(self):
        return self._improper_dihedral_list

    def reset_system(
        self, 
        reset_targets = False,
        reset_nxmol = False):

        from .tools import rdmol_to_nx

        self.openmm_system = self.openmm_system_init

        import copy
        _rdmol = copy.deepcopy(self.offmol.to_rdkit())
        self.rdmol = self.offmol.to_rdkit()
        Chem.SanitizeMol(self.rdmol)
        self.ranks = list(
            Chem.CanonicalRankAtoms(
                _rdmol, breakTies=False
                )
            )

        if reset_targets:
            self.target_list  = list()
            self._N_tgt       = 0

        if reset_nxmol:
            self.nxmol = rdmol_to_nx(self.rdmol)

        self.build_connectivity_lists()

        self.rdmol_setatommap  = False

    def build_connectivity_lists(self):

        __doc__ ="""
        Build connectivity lists containing all possible
        bonds, angles, proper dihedrals and improper dihedrals.
        """

        self._bond_list              = list()
        self._angle_list             = list()
        self._proper_dihedral_list   = list()
        self._improper_dihedral_list = list()

        bond_graph  = nx.Graph([(0,1)])
        angle_graph = nx.Graph([(0,1),
                                (1,2)])
        proper_dihedral_graph = nx.Graph([(0,1),
                                          (1,2),
                                          (2,3)])
        improper_dihedral_graph = nx.Graph([(0,1),
                                            (0,2),
                                            (0,3)])

        GM_bonds = isomorphism.GraphMatcher(
            self.nxmol,
            bond_graph
            )
        GM_angle = isomorphism.GraphMatcher(
            self.nxmol,
            angle_graph
            )
        GM_proper_dihedral = isomorphism.GraphMatcher(
            self.nxmol,
            proper_dihedral_graph
            )
        GM_improper_dihedral = isomorphism.GraphMatcher(
            self.nxmol,
            improper_dihedral_graph
            )

        for match in GM_bonds.subgraph_monomorphisms_iter():
            atm_idxs = list(match.keys())
            if not atm_idxs[::-1] in self._bond_list and \
               not atm_idxs in self._bond_list:
                self._bond_list.append(atm_idxs)

        for match in GM_angle.subgraph_monomorphisms_iter():
            atm_idxs = list(match.keys())
            if not atm_idxs[::-1] in self._angle_list and \
               not atm_idxs in self._angle_list:
                self._angle_list.append(atm_idxs)

        for match in GM_proper_dihedral.subgraph_monomorphisms_iter():
            atm_idxs = list(match.keys())
            if not atm_idxs[::-1] in self._proper_dihedral_list and \
               not atm_idxs in self._proper_dihedral_list:
                self._proper_dihedral_list.append(atm_idxs)

        for match in GM_improper_dihedral.subgraph_monomorphisms_iter():
            central_atom = list(match.keys())[0]
            outer_atoms  = list(match.keys())[1:]
            outer_atoms  = sorted(outer_atoms)
            atm_idxs     = [central_atom] + outer_atoms
            if not atm_idxs in self._improper_dihedral_list:
                self._improper_dihedral_list.append(atm_idxs)

    def get_neighbor_atomidxs(self, atm_idx):

        return list(self.nxmol.neighbors(atm_idx))

    def write(
        self,
        filename: str,
        xyz: np.ndarray,
        file_format: str = ''):

        write_system(self, xyz, filename, file_format)

    def add_target(self, target_class, target_dict):

        target = target_class(target_dict, self)
        self.target_list.append(target)
        self._N_tgt += 1

    def add_nxmol(
        self, 
        nxmol: nx.Graph) -> None:

        from .tools import atom_matcher, bond_matcher
        _bm = bond_matcher()
        _am = atom_matcher()
        _am_atomic_num = atom_matcher(["atomic_num"])
        _bm_empty      = bond_matcher([])

        assert nxmol.number_of_nodes() == self.nxmol.number_of_nodes()
        assert nxmol.number_of_edges() == self.nxmol.number_of_edges()

        GM = isomorphism.GraphMatcher(
            self.nxmol, 
            nxmol,
            _am_atomic_num,
            _bm_empty
            )

        assert GM.subgraph_is_monomorphic()
        for graph_mapping in GM.subgraph_monomorphisms_iter():
            for atm_idx in range(self.N_atoms):
                if "N_data" in self.nxmol.nodes[atm_idx]:
                    self.nxmol.nodes[atm_idx]["N_data"] += 1
                else:
                    self.nxmol.nodes[atm_idx]["N_data"] = 1
                mapped_atm_idx = graph_mapping[atm_idx]
                for key, value in nxmol.nodes[mapped_atm_idx].items():
                    if key == "atomic_num":
                        continue
                    if key in self.nxmol.nodes[atm_idx]:
                        self.nxmol.nodes[atm_idx][key].append(value)
                    else:
                        self.nxmol.nodes[atm_idx][key] = [value]
            for edge in self.nxmol.edges():
                mapped_edge = (
                    graph_mapping[edge[0]],
                    graph_mapping[edge[1]]
                    )
                if "N_data" in self.nxmol.edges[edge]:
                    self.nxmol.edges[edge]["N_data"] += 1
                else:
                    self.nxmol.edges[edge]["N_data"] = 1
                for key, value in nxmol.edges[mapped_edge].items():
                    if key in self.nxmol.edges[edge]:
                        self.nxmol.edges[edge][key].append(value)
                    else:
                        self.nxmol.edges[edge][key] = [value]
            ### Stop after first iteration.
            ### Continuing at this points generates artifically many
            ### data points.
            break


class SystemManager(object):

    def __init__(self, name="Default Manager"):

        self.__has_init   = False

        self.name         = name
        self._system_list = list()
        self._rdmol_list  = list()

        self.__has_init   = True

        self._rss_list  = list()
        self._diff_list = list()
        self._rss_dict  = dict()
        self._diff_dict = dict()

    @property
    def N_sys(self):
        return len(self._system_list)

    @property
    def N_tgt(self):
        N_tgt = 0 
        for system in self._system_list:
            N_tgt += system._N_tgt
        return N_tgt

    def reset_systems(
        self,
        reset_targets: bool = False,
        reset_nxmol: bool = False,
        ):

        for sys_idx in range(self.N_sys):
            self._system_list[sys_idx].reset_system(
                reset_targets,
                reset_nxmol,
                )

    def get_systems(
        self, 
        as_copy: bool = False,
        ):
        for system in self._system_list:
            if as_copy:
                _sys = copy.copy(system)
            else:
                _sys = system
            yield _sys

    def get_nxmols(
        self, 
        as_copy: bool = False,
        ) -> nx.Graph:
        for system in self._system_list:
            if as_copy:
                _nxmol = copy.copy(system.nxmol)
            else:
                _nxmol = system.nxmol
            yield _nxmol

    def get_system_by_idx(
        self, 
        idx: int,
        as_copy: bool = False):
        if as_copy:
            _sys = copy.copy(self._system_list[idx])
        else:
            _sys = self._system_list[idx]
        return _sys

    def get_nxmol_by_idx(
        self, 
        idx: int, 
        as_copy: bool = False):
        if as_copy:
            _nxmol = copy.copy(self._system_list[idx].nxmol)
        else:
            _nxmol = self._system_list[idx].nxmol
        return _nxmol

    def add_system(self, new_system, use_stereochemistry=False):

        from rdkit import Chem

        if use_stereochemistry:
            raise ValueError(
                f"Currently `use_stereochemistry` must be set to `False`")

        self._system_list.append(new_system)
        value = new_system.rdmol_setatommap
        new_system.rdmol_setatommap = False
        self._rdmol_list.append(
            Chem.MolToSmiles(
                new_system.offmol.to_rdkit(),
                isomericSmiles=use_stereochemistry
                )
            )
        new_system.rdmol_setatommap = value

    def has_molecule(self, molecule):

        from .tools import rdmol_to_nx

        if isinstance(molecule, nx.Graph):
            nxmol = molecule
        elif isinstance(molecule, System):
            nxmol = molecule.nxmol
        elif isinstance(molecule, Chem.Mol):
            nxmol = rdmol_to_nx(molecule)
        else:
            raise ValueError("Input type not understood.")

        for system_idx in range(self.N_sys):
            ### Use weak matching criteria
            GM = isomorphism.GraphMatcher(nxmol, 
                                          self._system_list[system_idx].nxmol,
                                          am_atomic_num,
                                          bm_empty)
            if GM.subgraph_is_isomorphic():
                return system_idx

def write_system(
    system: System, 
    xyz: np.ndarray, 
    filename: str, 
    file_format: str = ''
    ):

    def pdb():
        fopen = open(filename, "w")
        openmm.app.pdbfile.PDBFile.writeFile(system.top.to_openmm(),
                                             xyz,
                                             fopen)
        fopen.close()

    writer_dict = {
        'pdb' : pdb
    }

    if file_format:
        file_format = file_format.lower()
    else:
        for f in writer_dict.keys():
            if filename.lower().endswith("."+f):
                file_format = f
                break

    writer_dict[file_format]()


def from_qcschema(qcschema,
                  name="mol",
                  FF_name=_DEFAULT_FF,
                  partial_charges=None):

    offmol = Molecule.from_qcschema(
        qcschema, allow_undefined_stereo=True)
    use_charges = False
    if not isinstance(partial_charges, type(None)):
        from openff.units import unit
        use_charges = True
        offmol.partial_charges = partial_charges * unit.elementary_charge

    return from_offmol(offmol, name, FF_name, use_charges)


def from_offmol(offmol,
                name="mol",
                FF_name=_DEFAULT_FF,
                use_charges=False):

    offmol = copy.deepcopy(offmol)
    top    = offmol.to_topology()

    if FF_name.startswith("gaff"):
        from openmmforcefields.generators import GAFFTemplateGenerator
        from simtk.openmm.app import ForceField

        gaff = GAFFTemplateGenerator(
            molecules=offmol, 
            forcefield=FF_name
            )
        ff  = ForceField()
        ff.registerTemplateGenerator(gaff.generator)
        openmm_system = ff.createSystem(
            topology = top.to_openmm()
            )
    else:
        charges_list = None
        if use_charges:
            charges_list = [offmol]
        from openff.toolkit.typing.engines.smirnoff import ForceField
        ff     = ForceField(FF_name)
        openmm_system = ff.create_openmm_system(
                top, charge_from_molecules=charges_list)

    return System(name, offmol, openmm_system, top)
