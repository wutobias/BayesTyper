#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

from collections import OrderedDict

import networkx as nx
from networkx.algorithms import isomorphism
import copy
from rdkit import Chem

from .tools import rdmol_to_nx
from .molgraphs import ZMatrix
from .engines import OpenmmEngine
import numpy as np

import openmm
from openmm import unit

import ray

from BayesTyper import system

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================
from .constants import (_LENGTH,
                        _ANGLE,
                        _ATOMIC_MASS,
                        _ENERGY_PER_MOL,
                        _WAVENUMBER,
                        _UNIT_QUANTITY,
                        _FORCE)

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================
from .tools import atom_matcher, bond_matcher
_bm = bond_matcher()
_am = atom_matcher()


@ray.remote
def target_worker(openmm_system_dict, target_dict, return_results_dict=True):

    logP_likelihood  = 0.
    if return_results_dict:
        results_all_dict = dict()
    for sys_name in openmm_system_dict:
        if return_results_dict:
            results_all_dict[sys_name] = list()
        for openmm_system in openmm_system_dict[sys_name]:
            results_dict              = dict()
            results_dict["rss"]       = dict()
            results_dict["log_norm_factor"] = dict()
            N_tgt = len(target_dict[sys_name])
            for target_idx in range(N_tgt):
                target = copy.deepcopy(
                    target_dict[sys_name][target_idx]
                    )
                target.set_openmm_system(openmm_system)
                try:
                    target.run()
                    rss = target.rss
                    log_norm_factor = target.log_norm_factor
                    logP_likelihood += -log_norm_factor - 0.5 * np.log(2.*np.pi) - 0.5 * rss
                    results_dict["rss"][target_idx] = target.rss
                    results_dict["log_norm_factor"][target_idx] = target.log_norm_factor
                except:
                    logP_likelihood = -99999999999999999.
                    results_dict["rss"][target_idx] = -9999999999999999.
                    results_dict["log_norm_factor"][target_idx] = target.log_norm_factor

            if return_results_dict:
                results_all_dict[sys_name].append(results_dict)

    if return_results_dict:
        return logP_likelihood, results_all_dict
    else:
        return logP_likelihood


class TargetComputer(object):

    def __init__(
        self, 
        system_list, 
        target_type_list=None):

        import ray
        from .system import SystemManager

        self._target_dict = dict()

        if target_type_list == None:
            self.target_type_list = None
        elif len(target_list) == 0:
            self.target_type_list = None
        else:
            self.target_type_list = tuple(target_type_list)

        if isinstance(system_list, list):
            _system_list = system_list
        elif isinstance(system_list, SystemManager):
            _system_list = system_list._system_list

        for sys in _system_list:
            if self.target_type_list == None:
                self._target_dict[sys.name] = copy.deepcopy(
                    sys.target_list
                    )
            else:
                for target in sys.target_list:
                    self._target_dict[sys.name] = list()
                    if isinstance(target, self.target_type_list):
                        self._target_dict[sys.name].append(
                                copy.deepcopy(
                                    target
                                )
                            )

        self._target_dict = ray.put(self._target_dict)

    @property
    def target_dict(self):

        return self._target_dict

    def __call__(
        self, 
        openmm_system_dict,
        return_results_dict=True):

        worker_id = target_worker.remote(
            openmm_system_dict, 
            self.target_dict,
            return_results_dict
            )

        return worker_id


def compute_freqs(hessian, masses):

    ### Note: This routine is adapted from the following sources:
    ### https://github.com/leeping/forcebalance/blob/master/src/openmmio.py
    ### https://github.com/openforcefield/openforcefield-forcebalance/blob/master/vib_freq_target/make_vib_freq_target.py

    assert int(hessian.shape[0]/3) == masses.shape[0]

    N_atoms = int(hessian.shape[0]/3)

    ### 1.) Make some basic unit conversions.
    ### =====================================
    ### masses to g/mol
    masses_cp  = copy.copy(masses.value_in_unit(unit.dalton))
    ### Hessian to hartree/bohr (a.u. forces)
    hessian_cp = copy.copy(hessian.value_in_unit(unit.hartree / unit.bohr**2))
    ### Hessian to kg/s**2 * 10.e-23
    ### >>> a  = 1. * unit.hartree / unit.bohr**2 * unit.dalton
    ### >>> a  = a.in_units_of(unit.joule/unit.nanometer**2 * unit.dalton)
    ### >>> a *= unit.constants.AVOGADRO_CONSTANT_NA
    ### >>> print(a)
    hessian_cp *= 937582.9413466604 ### unit 1/s**-2

    ### Hessian to 1/s**2 * 10.e-23
    invert_sqrt_mass      = 1.0 / np.sqrt(masses_cp.repeat(3)) # repeat for x, y, z coords
    mass_weighted_hessian = hessian_cp * invert_sqrt_mass[:, np.newaxis] * invert_sqrt_mass[np.newaxis, :]

    ### freqs/eigvals in units 1/s**2 * 10.e-24
    eigvals       = np.linalg.eigvals(mass_weighted_hessian)
    negative_idxs = np.where(eigvals < 0.)
    ### freqs/eigvals in 1/s * 10.e-12
    freqs         = np.sqrt(np.abs(eigvals))
    freqs[negative_idxs] *= -1

    ### remove the 6 freqs with smallest abs value and corresponding normal modes
    n_remove         = 5 if N_atoms == 2 else 6
    larger_freq_idxs = np.sort(
        np.argpartition(
            np.abs(freqs),
            n_remove
            )[n_remove:]
        )
    freqs = np.sort(freqs[larger_freq_idxs])
    ### Convert 1/s * 10.e-12 to 1/cm
    ### >>> a = 1. * unit.seconds**-1
    ### >>> a = a * unit.constants.SPEED_OF_LIGHT_C**-1
    ### >>> a = a.in_units_of(unit.centimeter**-1)
    ### >>> print(a)
    freqs *= 33.3564095198152 * unit.centimeters**-1
    ### Convert to angular wavenumber to wavelength of wave
    freqs *= 0.5 / np.pi

    del invert_sqrt_mass, mass_weighted_hessian

    return freqs.in_units_of(_WAVENUMBER)


class Target(object):

    def __init__(self, target_dict, system):

        self.rss       = 0.
        self.diff      = 0.
        self.diff_dict = OrderedDict()
        self.rss_dict  = OrderedDict()

        self.target_strcs = list()
        for target_strc in target_dict["structures"]:
            target_strc = target_strc.value_in_unit(_LENGTH)
            self.target_strcs.append(target_strc)
        self.target_strcs = np.array(self.target_strcs) * _LENGTH

        self.N_strcs = len(self.target_strcs)

        self.target_rdmol = copy.deepcopy(
            target_dict["rdmol"]
            )

        self._rdmol         = system.rdmol
        self._top           = system.top
        self._N_atoms       = system.N_atoms
        self._openmm_system = system.openmm_system

        self._align_graph()

    @property
    def log_norm_factor(self):
        return 0.

    def set_openmm_system(
        self, 
        openmm_system: openmm.openmm.System):

        self._openmm_system = openmm_system

    def _align_graph(self):

        ### Correct any difference in atom ordering
        ### =======================================
        
        ### The target and system might be similar but have different
        ### atom ordering. We need to match them based on their graph
        ### representation.
        Chem.SanitizeMol(self.rdmol)
        Chem.SanitizeMol(self.target_rdmol)

        G1 = rdmol_to_nx(self.rdmol)
        G2 = rdmol_to_nx(self.target_rdmol)

        assert G1.number_of_nodes() == self.N_atoms
        assert G2.number_of_nodes() == self.N_atoms

        GM = isomorphism.GraphMatcher(
            G1, G2,
            node_match=_am,
            edge_match=_bm
            )
        assert GM.is_isomorphic()

        self._GM_mapping = GM.mapping

        self.masses = list()
        for a1, a2 in GM.mapping.items():
            rdatom = self.rdmol.GetAtomWithIdx(a1)
            self.masses.append(float(rdatom.GetMass()))
        self.masses = np.array(self.masses) * _ATOMIC_MASS

        for strc_idx in range(self.N_strcs):

            ### Two copies here are necessary, because under `ray`
            ### we cannot change elements in self.target_strc.
            target_strc_copy1 = copy.deepcopy(
                self.target_strcs[strc_idx])
            target_strc_copy2 = copy.deepcopy(
                self.target_strcs[strc_idx])

            for a1, a2 in GM.mapping.items():
                target_strc_copy1[a1] = target_strc_copy2[a2]

            self.target_strcs[strc_idx] = target_strc_copy1

    def run(self):

        pass

    def getattr(self, name):
        return getattr(self, name)
    
    def setattr(self, name, value):
        setattr(self, name, value)

    @property
    def top(self):
        return self._top

    @property
    def N_atoms(self):
        return self._N_atoms

    @property
    def openmm_system(self):
        return self._openmm_system

    @property
    def rdmol(self):
        return self._rdmol


class GeoTarget(Target):

    def __init__(
        self,
        target_dict: dict,
        system: system.System):

        super().__init__(target_dict, system)

        ### Default denomiators taken from Parsley paper:
        ### doi.org/10.26434/chemrxiv.13082561.v2
        self.denom_bond    = 0.005 * _LENGTH
        self.denom_angle   = 8.    * _ANGLE
        self.denom_torsion = 20.   * _ANGLE
        self.H_constraint  = True

        if "denom_bond" in target_dict:
            self.denom_bond = target_dict["denom_bond"]
            self.denom_bond = self.denom_bond.in_units_of(_LENGTH)
        if "denom_angle" in target_dict:
            self.denom_angle = target_dict["denom_angle"]
            self.denom_angle = self.denom_angle.in_units_of(_ANGLE)
        if "denom_torsion" in target_dict:
            self.denom_torsion = target_dict["denom_torsion"]
            self.denom_torsion = self.denom_torsion.in_units_of(_ANGLE)

        if "H_constraint" in target_dict:
            self.H_constraint = target_dict["H_constraint"]

        self.configure_target()

    @property
    def log_norm_factor(self):

        value  = self.N_bonds    * np.log(self.denom_bond.value_in_unit(_LENGTH))
        value += self.N_angles   * np.log(self.denom_angle.value_in_unit(_ANGLE))
        value += self.N_torsions * np.log(self.denom_torsion.value_in_unit(_ANGLE))

        return value

    def configure_target(self):

        self.zm = ZMatrix(self.rdmol)

        ### Note this is not necessarily the number
        ### of physical bonds in the molecule. In 
        ### zmatrix notation this is the number bond
        ### degrees of freedom.
        self.N_bonds    = float(self.N_atoms - 1)
        self.N_angles   = float(self.N_atoms - 2)
        self.N_torsions = float(self.N_atoms - 3)

        self.H_list = list()
        for z_idx in range(self.N_atoms):
            a_idx = self.zm.z2a(z_idx)
            atm   = self.rdmol.GetAtomWithIdx(a_idx)
            if atm.GetAtomicNum() == 1:
                self.H_list.append(z_idx)

        self.target_zm = list()
        for strc_idx in range(self.N_strcs):
            self.target_zm.append(
                self.zm.build_z_crds(
                    self.target_strcs[strc_idx]
                    )
                )

    def run(self):

        import copy

        ### Note that OpenmmEngine will create
        ### a copy.deepcopy copy from openmm_system,
        ### top and target_strc before doing anything
        ### with them.
        engine  = OpenmmEngine(
            self.openmm_system,
            self.top,
            self.target_strcs[0]
            )
        N_success = 0

        rss_bond    = 0.0 * _LENGTH**2
        rss_angle   = 0.0 * _ANGLE**2
        rss_torsion = 0.0 * _ANGLE**2

        diff_bond    = 0.0 * _LENGTH
        diff_angle   = 0.0 * _ANGLE
        diff_torsion = 0.0 * _ANGLE

        self.diff_dict = OrderedDict()
        self.rss_dict  = OrderedDict()

        self.rss  = 0.
        self.diff = 0.

        for strc_idx in range(self.N_strcs):
            #engine.xyz = copy.deepcopy(self.target_strcs[strc_idx])
            engine.xyz = self.target_strcs[strc_idx]

            success = engine.minimize()

            if success:
                xyz        = engine.xyz.in_units_of(_LENGTH)
                current_zm = self.zm.build_z_crds(xyz)
                N_success += 1
            else:
                continue

            target_zm = self.target_zm[strc_idx]
            for z_idx, z_value in current_zm.items():
                if z_idx == 0:
                    continue
                ### Bonds
                if z_idx > 0:
                    ### If we have constraint bonds to H atoms, we don't
                    ### want to include the bond length of those bonds.
                    if self.H_constraint:
                        if not z_idx in self.H_list:
                            diff       = abs(target_zm[z_idx][0] - z_value[0])
                            diff       = diff.in_units_of(_LENGTH)
                            diff_bond += diff
                            rss_bond  += diff**2
                    else:
                        diff       = abs(target_zm[z_idx][0] - z_value[0])
                        diff       = diff.in_units_of(_LENGTH)
                        diff_bond += diff
                        rss_bond  += diff**2
                ### Angles
                if z_idx > 1:
                    #print(target_zm[z_idx][1], z_value[1])
                    diff        = abs(target_zm[z_idx][1] - z_value[1])
                    diff        = diff.in_units_of(_ANGLE)
                    diff_angle += diff
                    rss_angle  += diff**2
                ### Torsions
                if z_idx > 2:
                    diff = abs(target_zm[z_idx][2] - z_value[2])
                    diff = diff.in_units_of(_ANGLE)
                    if diff > 180.*_ANGLE:
                        diff = diff - 360.*_ANGLE
                        diff = abs(diff)
                    diff_torsion += diff
                    rss_torsion  += diff**2

        if N_success != self.N_strcs:
            self.rss_bond     = np.inf
            self.rss_angle    = np.inf
            self.rss_torsion  = np.inf

            self.diff_bond    = np.inf * diff_bond.unit
            self.diff_angle   = np.inf * diff_angle.unit
            self.diff_torsion = np.inf * diff_torsion.unit
            
        else:
            ### RSS calcs
            self.rss_bond     = 1./(self.denom_bond)**2    * rss_bond
            self.rss_angle    = 1./(self.denom_angle)**2   * rss_angle
            self.rss_torsion  = 1./(self.denom_torsion)**2 * rss_torsion

            ### DIFF calcs
            self.diff_bond     = diff_bond
            self.diff_angle    = diff_angle
            self.diff_torsion  = diff_torsion

        self.rss_dict['bond']    = self.rss_bond
        self.rss_dict['angle']   = self.rss_angle
        self.rss_dict['torsion'] = self.rss_torsion

        self.rss  = self.rss_bond
        self.rss += self.rss_angle
        self.rss += self.rss_torsion

        self.diff_dict['bond']    = self.diff_bond
        self.diff_dict['angle']   = self.diff_angle
        self.diff_dict['torsion'] = self.diff_torsion

        self.diff  = self.diff_bond._value
        self.diff += self.diff_angle._value
        self.diff += self.diff_torsion._value

        del engine


class NormalModeTarget(Target):

    def __init__(
        self,
        target_dict: dict,
        system: system.System):

        super().__init__(target_dict, system)

        if "minimize" in target_dict:
            self.minimize  = target_dict["minimize"]
        else:
            self.minimize = True

        ### Default denomiators taken from Parsley paper:
        ### doi.org/10.26434/chemrxiv.13082561.v2
        self.denom_frq    = 200. * _WAVENUMBER

        if "denom_frq" in target_dict:
            self.denom_frq = target_dict["denom_frq"]
            self.denom_frq = self.denom_frq.in_units_of(_WAVENUMBER)

        self.target_hessian = list()
        for hes_idx in range(len(target_dict["hessian"])):
            self.target_hessian.append(
                np.array(target_dict["hessian"][hes_idx]._value) *\
                target_dict["hessian"][hes_idx].unit
                )

        assert self.N_strcs == len(self.target_hessian)

        self.configure_target()

    @property
    def log_norm_factor(self):

        value  = self.N_freqs * np.log(self.denom_frq.value_in_unit(_WAVENUMBER))

        return value

    def configure_target(self):

        ### 1) Configure all the basic stuff
        ### ================================

        if self.N_atoms == 2:
            self.N_freqs = 5
        else:
            self.N_freqs = self.N_atoms * 3 - 6
        self.target_freqs = list()

        ### 2) Compute vib frequencies
        ### ==========================
        for hes_idx in range(self.N_strcs):
            self.target_freqs.append(
                compute_freqs(
                    self.target_hessian[hes_idx], 
                    self.masses
                    )
                )

    def run(self):

        import copy

        self.diff_dict = OrderedDict()
        self.rss_dict  = OrderedDict()

        self.rss  = 0.
        self.diff = 0. * _WAVENUMBER

        engine  = OpenmmEngine(
            self.openmm_system,
            self.top,
            self.target_strcs[0]
            )

        for strc_idx in range(self.N_strcs):

            #engine.xyz = copy.deepcopy(self.target_strcs[strc_idx])
            engine.xyz = self.target_strcs[strc_idx]

            success = True
            if self.minimize:
                success = engine.minimize()
            if success:
                hessian  = engine.compute_hessian()
                ### Remove 1/mol
                hessian *= unit.constants.AVOGADRO_CONSTANT_NA**-1
                freqs    = compute_freqs(hessian, self.masses)

                diff    = abs(freqs - self.target_freqs[strc_idx])
                rss     = diff**2 * (1./self.denom_frq)**2

                self.diff_dict[strc_idx] = {i:f for i,f in enumerate(diff)}
                self.rss_dict[strc_idx]  = {i:f for i,f in enumerate(rss)}

                self.rss  += np.sum(rss)
                self.diff += np.sum(diff)
            else:
                self.rss  = np.inf
                self.diff = np.inf

                del engine

                return

        del engine


class ForceProjectionMatchingTarget(Target):

    def __init__(
        self,
        target_dict: dict,
        system: system.System
        ):

        super().__init__(target_dict, system)

        self.denom_bond    = 1. * _FORCE
        self.denom_angle   = 8.    * _FORCE
        self.denom_torsion = 20.   * _FORCE
        self.H_constraint  = True

        ### For starters, these are just the same denom as
        ### for the geo targety

        if "denom_bond" in target_dict:
            self.denom_bond = target_dict["denom_bond"]
            self.denom_bond = self.denom_bond.in_units_of(_FORCE)
        if "denom_angle" in target_dict:
            self.denom_angle = target_dict["denom_angle"]
            self.denom_angle = self.denom_angle.in_units_of(_FORCE)
        if "denom_torsion" in target_dict:
            self.denom_torsion = target_dict["denom_torsion"]
            self.denom_torsion = self.denom_torsion.in_units_of(_FORCE)

        if "H_constraint" in target_dict:
            self.H_constraint = target_dict["H_constraint"]

        self.target_forces = list()
        for target_force in target_dict["forces"]:
            has_mole = False
            for t_u in  target_force.unit.iter_top_base_units():
                if t_u[0].name == 'mole' and t_u[1] == -1.0:
                    has_mole = True
                    break
            if not has_mole:
                target_force *= unit.constants.AVOGADRO_CONSTANT_NA
            target_force = target_force.value_in_unit(_FORCE)

            target_force_copy1 = copy.deepcopy(
                target_force)
            target_force_copy2 = copy.deepcopy(
                target_force)

            for a1, a2 in self._GM_mapping.items():
                target_force_copy1[a1] = target_force_copy2[a2]

            self.target_forces.append(target_force_copy1)
        self.target_forces = np.array(self.target_forces) * _FORCE

        assert len(self.target_strcs) ==  len(self.target_forces)

        self.configure_target()

    @property
    def log_norm_factor(self):

        value  = self.N_bonds * np.log(self.denom_bond.value_in_unit(_LENGTH))
        value += self.N_angles * np.log(self.denom_angle.value_in_unit(_ANGLE))
        value += self.N_torsions * np.log(self.denom_torsion.value_in_unit(_ANGLE))

        return value

    def configure_target(self):

        self.zm = ZMatrix(self.rdmol)

        ### Note this is not necessarily the number
        ### of physical bonds in the molecule. In 
        ### zmatrix notation this is the number bond
        ### degrees of freedom.
        self.N_bonds    = float(self.N_atoms - 1)
        self.N_angles   = float(self.N_atoms - 2)
        self.N_torsions = float(self.N_atoms - 3)

        self.H_list = list()
        for z_idx in range(self.N_atoms):
            a_idx = self.zm.z2a(z_idx)
            atm   = self.rdmol.GetAtomWithIdx(a_idx)
            if atm.GetAtomicNum() == 1:
                self.H_list.append(z_idx)

        self.target_force_projection = list()
        self.B_flat_list = list()
        for strc_idx in range(self.N_strcs):

            cart_crds  = self.target_strcs[strc_idx].in_units_of(_LENGTH)
            cart_force = self.target_forces[strc_idx].in_units_of(_FORCE)

            B_flat = self.zm.build_wilson_b(
                cart_crds, 
                as_dict=False
                )
            force_q = self.zm.build_grad_projection(
                B_flat, 
                np.array(cart_force),
                as_dict=True
                )
            self.target_force_projection.append(force_q)
            self.B_flat_list.append(B_flat)

    def run(self):

        ### Note that OpenmmEngine will create
        ### a copy.deepcopy copy from openmm_system,
        ### top and target_strc before doing anything
        ### with them.
        engine  = OpenmmEngine(
            self.openmm_system,
            self.top,
            self.target_strcs[0]
            )

        rss_bond    = 0.0 * _FORCE**2
        rss_angle   = 0.0 * _FORCE**2
        rss_torsion = 0.0 * _FORCE**2

        diff_bond    = 0.0 * _FORCE
        diff_angle   = 0.0 * _FORCE
        diff_torsion = 0.0 * _FORCE

        self.diff_dict = OrderedDict()
        self.rss_dict  = OrderedDict()

        self.rss  = 0.
        self.diff = 0.

        for strc_idx in range(self.N_strcs):

            _rss_bond    = 0.0 * _FORCE**2
            _rss_angle   = 0.0 * _FORCE**2
            _rss_torsion = 0.0 * _FORCE**2

            #target_strc  = copy.deepcopy(self.target_strcs[strc_idx])
            target_strc  = self.target_strcs[strc_idx]
            target_force = self.target_force_projection[strc_idx]
            engine.xyz   = target_strc
            forces       = engine.forces.in_units_of(_FORCE)

            B_flat  = self.B_flat_list[strc_idx]
            force_q = self.zm.build_grad_projection(
                B_flat, 
                forces,
                as_dict=True
                )

            for z_idx, force_values in force_q.items():
                if z_idx == 0:
                    continue
                ### Bonds
                if z_idx > 0:
                    ### If we have constraint bonds to H atoms, we don't
                    ### want to include the bond length of those bonds.
                    if self.H_constraint:
                        if not z_idx in self.H_list:
                            diff       = abs(target_force[z_idx][0] - force_values[0])
                            diff       = diff.in_units_of(_FORCE)
                            diff_bond += diff
                            _rss_bond += diff**2
                    else:
                        diff       = abs(target_force[z_idx][0] - force_values[0])
                        diff       = diff.in_units_of(_FORCE)
                        diff_bond += diff
                        _rss_bond += diff**2
                ### Angles
                if z_idx > 1:
                    diff        = abs(target_force[z_idx][1] - force_values[1])
                    diff        = diff.in_units_of(_FORCE)
                    diff_angle += diff
                    _rss_angle += diff**2
                ### Torsions
                if z_idx > 2:
                    diff = abs(target_force[z_idx][2] - force_values[2])
                    diff = diff.in_units_of(_FORCE)
                    diff_torsion += diff
                    _rss_torsion += diff**2

            norm_factor  = 1.

            rss_bond    += _rss_bond * norm_factor
            rss_angle   += _rss_angle * norm_factor
            rss_torsion += _rss_torsion * norm_factor

        ### RSS calcs
        self.rss_bond     = 1./(self.denom_bond)**2    * rss_bond
        self.rss_angle    = 1./(self.denom_angle)**2   * rss_angle
        self.rss_torsion  = 1./(self.denom_torsion)**2 * rss_torsion

        ### DIFF calcs
        self.diff_bond     = diff_bond
        self.diff_angle    = diff_angle
        self.diff_torsion  = diff_torsion

        self.rss_dict['bond']    = self.rss_bond
        self.rss_dict['angle']   = self.rss_angle
        self.rss_dict['torsion'] = self.rss_torsion

        self.rss  = self.rss_bond
        self.rss += self.rss_angle
        self.rss += self.rss_torsion

        self.diff_dict['bond']    = self.diff_bond
        self.diff_dict['angle']   = self.diff_angle
        self.diff_dict['torsion'] = self.diff_torsion

        self.diff  = self.diff_bond._value
        self.diff += self.diff_angle._value
        self.diff += self.diff_torsion._value

        del engine


class ForceMatchingTarget(Target):

    def __init__(
        self,
        target_dict: dict,
        system: system.System):

        super().__init__(target_dict, system)

        ### Default denomiators taken from Parsley paper:
        ### doi.org/10.26434/chemrxiv.13082561.v2
        self.denom_force = 100. * _FORCE

        if "denom_force" in target_dict:
            self.denom_force = target_dict["denom_force"]
            self.denom_force = self.denom_frq.in_units_of(_FORCE)

        self.target_forces = list()
        for target_force in target_dict["forces"]:
            has_mole = False
            for t_u in  target_force.unit.iter_top_base_units():
                if t_u[0].name == 'mole' and t_u[1] == -1.0:
                    has_mole = True
                    break
            if not has_mole:
                target_force *= unit.constants.AVOGADRO_CONSTANT_NA

            target_force_copy1 = copy.deepcopy(
                target_force.value_in_unit(_FORCE)
                )
            target_force_copy2 = copy.deepcopy(
                target_force.value_in_unit(_FORCE)
                )

            for a1, a2 in self._GM_mapping.items():
                target_force_copy1[a1] = target_force_copy2[a2]

            self.target_forces.append(target_force_copy1)

        self.target_forces = np.array(self.target_forces) * _FORCE

        assert len(self.target_strcs) ==  len(self.target_forces)

    @property
    def log_norm_factor(self):

        value  = self.N_atoms * 3 * np.log(self.denom_force.value_in_unit(_FORCE))

        return value

    def run(self):

        import copy

        engine  = OpenmmEngine(
            self.openmm_system,
            self.top,
            self.target_strcs[0]
            )

        self.diff_dict = OrderedDict()
        self.rss_dict  = OrderedDict()

        self.rss  = 0.
        self.diff = 0. * _FORCE

        for strc_idx in range(self.N_strcs):
            target_strc  = self.target_strcs[strc_idx]
            target_force = self.target_forces[strc_idx]
            engine.xyz   = target_strc
            forces       = engine.forces.in_units_of(_FORCE)
            ### This same as computing length of diff vector
            ### and then taking square
            diff         = abs(forces - target_force).in_units_of(_FORCE)
            diff2        = np.sum(diff**2, axis=1)
            rss          = diff2 * 1./self.denom_force**2
            rss         /= float(self.N_atoms * 3)

            self.diff_dict[strc_idx] = diff.tolist() * _FORCE
            self.rss_dict[strc_idx]  = rss.tolist()

            self.rss  += np.sum(rss)
            self.diff += np.sum(diff)

        del engine


class EnergyTarget(Target):

    def __init__(
        self,
        target_dict: dict,
        system: system.System,
        restraint_dict: dict = dict()):

        ### ----------------------------------------
        ### 1.) Setup target structures and energies
        ### ----------------------------------------

        super().__init__(target_dict, system)

        self.denom_ene = 1. * _ENERGY_PER_MOL

        self.minimize  = False
        if "minimize" in target_dict:
            self.minimize  = target_dict["minimize"]

        self.ene_weighting  = True
        if "ene_weighting" in target_dict:
            self.ene_weighting  = target_dict["ene_weighting"]

        if "denom_ene" in target_dict:
            self.denom_ene = target_dict["denom_ene"]
            self.denom_ene = self.denom_ene.in_units_of(_ENERGY_PER_MOL)

        self.target_energies = list()
        for target_energy in target_dict["energies"]:
            has_mole = False
            for t_u in  target_energy.unit.iter_top_base_units():
                if t_u[0].name == 'mole' and t_u[1] == -1.0:
                    has_mole = True
                    break
            if not has_mole:
                target_energy *= unit.constants.AVOGADRO_CONSTANT_NA
            target_energy  = target_energy.value_in_unit(_ENERGY_PER_MOL)
            self.target_energies.append(target_energy)
        ### Substract the lowest energy from all energies
        self.target_energies = np.array(self.target_energies) * _ENERGY_PER_MOL
        min_ene              = np.min(self.target_energies)
        self.target_energies = self.target_energies - min_ene
        self.min_ene_arg     = np.argmin(self.target_energies)

        ### Find weights for energies.
        ### See Parsley paper:
        ### doi.org/10.26434/chemrxiv.13082561.v2
        self.target_denom = list()
        if self.ene_weighting:
            for strc_idx in range(self.N_strcs):
                target_energy = self.target_energies[strc_idx]
                target_energy = target_energy.value_in_unit(unit.kilocalorie_per_mole)

                if target_energy < 1.:
                    w = 1.
                elif 1. <= target_energy < 5.:
                    w = np.sqrt(1. + (target_energy - 1.)**2)
                else:
                    w = 0.
                self.target_denom.append(w)
            self.target_denom  = np.array(self.target_denom)
            self.target_denom /= np.sum(self.target_denom)
        else:
            self.target_denom  = np.zeros(self.N_strcs)
            self.target_denom[:] = 1./self.target_denom.size

        assert len(self.target_strcs) ==  len(self.target_energies)

        ### ---------------------------
        ### 2.) Setup target restraints
        ### ---------------------------

        self.restraint_atom_indices = list()
        self.restraint_k            = list()
        self.restraint_eq_value     = list()
        self.N_restraints           = 0
        if restraint_dict:
            self.N_restraints = len(restraint_dict["atom_indices"])
            if self.N_restraints > 0:
                self.restraint_atom_indices = restraint_dict["atom_indices"]
                self.restraint_k            = restraint_dict["k"]
                self.restraint_eq_value     = restraint_dict["eq_value"]

                assert len(self.restraint_k) == self.N_restraints
                assert len(self.restraint_eq_value) == self.N_strcs

    @property
    def log_norm_factor(self):

        value  = np.log(self.denom_ene.value_in_unit(_ENERGY_PER_MOL))

        return value

    def run(self):

        engine  = OpenmmEngine(
            self.openmm_system,
            self.top,
            self.target_strcs[0]
            )

        self.diff_dict = OrderedDict()
        self.rss_dict  = OrderedDict()
        self.vals_dict = OrderedDict()

        self.rss  = 0.
        self.diff = 0. * _ENERGY_PER_MOL

        #engine.xyz = copy.deepcopy(self.target_strcs[self.min_ene_arg])
        engine.xyz = self.target_strcs[self.min_ene_arg]
        ### Important: Add the forces only through the engine.
        ### The original openmm_system object must remain untouched.
        if self.N_restraints > 0:
            for restraint_id in range(self.N_restraints):
                atm_idxs  = self.restraint_atom_indices[restraint_id]
                k         = self.restraint_k[restraint_id]
                ### eq_value for first structure
                eq_value  = self.restraint_eq_value[self.min_ene_arg][restraint_id]
                if len(atm_idxs) == 2:
                    engine.add_bond_restraint(
                        atm_idxs[0],
                        atm_idxs[1],
                        k,
                        eq_value)
                elif len(atm_idxs) == 3:
                    engine.add_angle_restraint(
                        atm_idxs[0],
                        atm_idxs[1],
                        atm_idxs[2],
                        k,
                        eq_value)
                elif len(atm_idxs) == 4:
                    engine.add_torsion_restraint(
                        atm_idxs[0],
                        atm_idxs[1],
                        atm_idxs[2],
                        atm_idxs[3],
                        k,
                        eq_value)
                else:
                    NotImplementedError

        if self.minimize:
            engine.minimize()
        ref_ene = engine.pot_ene.in_units_of(_ENERGY_PER_MOL)

        for strc_idx in range(self.N_strcs):
            if strc_idx == self.min_ene_arg:
                continue
            if self.target_denom[strc_idx] == 0.:
                continue
            target_strc    = self.target_strcs[strc_idx]
            target_pot_ene = self.target_energies[strc_idx]
            engine.xyz     = target_strc

            if self.minimize:
                engine.minimize()
            pot_ene = engine.pot_ene.in_units_of(_ENERGY_PER_MOL)
            pot_ene = pot_ene - ref_ene
            diff    = pot_ene - target_pot_ene
            diff    = diff.in_units_of(_ENERGY_PER_MOL)
            diff2   = diff**2
            rss     = diff2 * self.target_denom[strc_idx] / self.denom_ene**2

            self.diff_dict[strc_idx] = diff
            self.rss_dict[strc_idx]  = rss
            self.vals_dict[strc_idx] = pot_ene

            self.rss  += rss
            self.diff += diff

        del engine
