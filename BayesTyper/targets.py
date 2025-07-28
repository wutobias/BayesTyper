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
                        _FORCE,
                        _TIMEOUT,
                        _VERBOSE)

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================
from .tools import atom_matcher, bond_matcher
_bm = bond_matcher()
_am = atom_matcher()


def build_grad_projection(wilson_b, zmatrix, grad_x, as_dict=True):

    __doc__= """
    Compute projection of gradient along zmat coordinates.

    `wilson_b` can be either wilson B matrix or
    dict of that matrix.
    `grad_x` can be either flattend coordinate list (i.e. (N,3) -> (3*N))
    or coordinate list. Note, it must be either
    """

    N_atms = len(zmatrix)
    if isinstance(wilson_b, dict):
        wilson_b_flat = list()
        for z_idx in range(1, N_atms):
            atm_idxs = zmatrix[z_idx]
            if z_idx > 0:
                wilson_b_row = np.zeros(
                    (N_atms, 3), dtype=float)
                wilson_b_row[atm_idxs[:2]] = wilson_b[z_idx][0]
                wilson_b_flat.append(wilson_b_row.flatten())
            if z_idx > 1:
                wilson_b_row = np.zeros(
                    (N_atms, 3), dtype=float)
                wilson_b_row[atm_idxs[:3]] = wilson_b[z_idx][1]
                wilson_b_flat.append(wilson_b_row.flatten())
            if z_idx > 2:
                wilson_b_row = np.zeros(
                    (N_atms, 3), dtype=float)
                wilson_b_row[atm_idxs[:4]] = wilson_b[z_idx][2]
                wilson_b_flat.append(wilson_b_row.flatten())

        wilson_b_flat = np.array(wilson_b_flat)
    else:
        wilson_b_flat = wilson_b

    if grad_x.ndim != 1:
        grad_x_flat = grad_x.value_in_unit(_FORCE).flatten()
    else:
        grad_x_flat = grad_x.value_in_unit(_FORCE)

    length_wilson_b  = 0
    length_wilson_b += N_atms * 3 * (N_atms - 1)
    length_wilson_b += N_atms * 3 * (N_atms - 2)
    length_wilson_b += N_atms * 3 * (N_atms - 3)

    if grad_x_flat.size != (N_atms * 3):
        raise ValueError(
            f"length of `grad_x_flat` is {grad_x_flat.size}, but must be {N_atms * 3}")

    if wilson_b_flat.shape[1] != (N_atms * 3):
        raise ValueError(
            f"shape of `wilson_b_flat` is {wilson_b_flat.size}, but must be {length_wilson_b}")

    if wilson_b_flat.size != (length_wilson_b):
        raise ValueError(
            f"length of `wilson_b_flat` is {wilson_b_flat.size}, but must be {length_wilson_b}")

    ### ============================= ###
    ### THIS IS THE ACTUAL PROJECTION ###
    ### ============================= ###
    ###
    ### This is the Bakken et al. approach.
    ### See 10.1063/1.1515483
    ###
    ### Btw:
    ### ... it gives the same gradient
    ### as with the approach by Peng et al. 
    ### See 10.1002/(SICI)1096-987X(19960115)17:1<49::AID-JCC5>3.0.CO;2-0
    ### u = np.eye(B_flat.shape[1])
    ### G = np.dot(B_flat, np.dot(u, B_flat.T))
    ### G_inv  = np.linalg.pinv(G)
    ### Bugx   = np.dot(B_flat, np.dot(u, grad_x_flat))
    ### grad_q = np.dot(G_inv, Bugx)

    Bt_inv = np.linalg.pinv(wilson_b_flat.T)
    grad_q = np.dot(Bt_inv, grad_x_flat)

    if as_dict:
        grad_q_dict = dict()
        z_counts    = 0
        for z_idx in range(1, N_atms):
            grad_q_dict[z_idx] = list()
            if z_idx > 0:
                grad_q_dict[z_idx].append(grad_q[z_counts])
                z_counts += 1
            if z_idx > 1:
                grad_q_dict[z_idx].append(grad_q[z_counts])
                z_counts += 1
            if z_idx > 2:
                grad_q_dict[z_idx].append(grad_q[z_counts])
                z_counts += 1

        return grad_q_dict

    return grad_q


def build_z_crds(z, crds):

    __doc__ = """
    Build zmat coordinates expecting that ordering
    of cartesian input coordinates is in ordering
    of original rdkit molecule.
    z is z matrix dict
    crds are cartesian corodinates
    """

    from .molgraphs import pts_to_bond
    from .molgraphs import pts_to_angle
    from .molgraphs import pts_to_dihedral

    z_crds_dict = dict()
    for z_idx, atm_idxs in z.items():
        z_crds_dict[z_idx] = list()
        if z_idx == 0:
            z_crds_dict[z_idx].append(
                crds[atm_idxs[0]])
        if z_idx > 0:
            dist = pts_to_bond(crds[atm_idxs[0]]*_LENGTH,
                               crds[atm_idxs[1]]*_LENGTH)
            z_crds_dict[z_idx].append(
                dist.value_in_unit(_LENGTH))
        if z_idx > 1:
            ang = pts_to_angle(crds[atm_idxs[0]]*_LENGTH,
                               crds[atm_idxs[1]]*_LENGTH,
                               crds[atm_idxs[2]]*_LENGTH)
            z_crds_dict[z_idx].append(
                ang.value_in_unit(_ANGLE))
        if z_idx > 2:
            dih = pts_to_dihedral(crds[atm_idxs[0]]*_LENGTH,
                                  crds[atm_idxs[1]]*_LENGTH,
                                  crds[atm_idxs[2]]*_LENGTH,
                                  crds[atm_idxs[3]]*_LENGTH)
            z_crds_dict[z_idx].append(
                dih.value_in_unit(_ANGLE))
    return z_crds_dict


def run_normalmodetarget(
    openmm_system, target_strcs, target_freqs, target_modes, minimize, masses, denom_frq,
    return_results_dict=False, permute=False):

    import copy
    if permute:
        from scipy import optimize

    diff_dict = OrderedDict()
    rss_dict  = OrderedDict()

    rss  = 0.
    diff = 0.

    if not unit.is_quantity(target_strcs[0]):
        _xyz = target_strcs[0] * _LENGTH
    else:
        _xyz = target_strcs[0]

    if unit.is_quantity(masses):
        masses = masses.value_in_unit(_ATOMIC_MASS)

    if unit.is_quantity(denom_frq):
        denom_frq = denom_frq.value_in_unit(_WAVENUMBER)

    engine  = OpenmmEngine(
        openmm_system,
        _xyz)

    N_strcs = len(target_strcs)
    N_freqs = len(target_freqs)
    N_success = 0
    freq_diff_list = list()
    for strc_idx in range(N_strcs):

        if not unit.is_quantity(target_strcs[strc_idx]):
            _xyz = target_strcs[strc_idx] * _LENGTH
        else:
            _xyz = target_strcs[strc_idx]

        engine.xyz = _xyz

        success = True
        if minimize:
            success = engine.minimize()
        if success:
            N_success += 1
            hessian    = engine.compute_hessian()
            ### Remove 1/mol
            hessian     /= unit.constants.AVOGADRO_CONSTANT_NA
            freqs, modes = compute_freqs(hessian, masses*_ATOMIC_MASS)
            freqs        = freqs.value_in_unit(_WAVENUMBER)
            ### re-assign frequencies by computing optimal overlap
            ### between modes
            if unit.is_quantity(target_freqs[strc_idx]):
                target_freqs[strc_idx] = target_freqs[strc_idx].value_in_unit(_WAVENUMBER)
            _diff = abs(freqs - target_freqs[strc_idx])
            _rss  = _diff**2 / denom_frq**2
            if permute:
                overlap = np.einsum('ij,ik', target_modes[strc_idx], modes)
                row_ind, col_ind = optimize.linear_sum_assignment(1.-overlap)
                modes = modes[col_ind]
                freqs = freqs[col_ind]

            _diff    = abs(freqs - target_freqs[strc_idx])
            _rss     = _diff**2 / denom_frq**2

            diff_dict[strc_idx] = {i:f for i,f in enumerate(_diff*_WAVENUMBER)}
            rss_dict[strc_idx]  = {i:f for i,f in enumerate(_rss)}

            rss  += np.sum(_rss)
            diff += np.sum(_diff)
            freq_diff_list.extend(_diff.tolist())

    if N_success == N_strcs:
        diff *= _WAVENUMBER
    else:
        diff = np.inf * _WAVENUMBER
        rss  = np.inf

    del engine

    log_norm_factor = N_freqs * np.log(denom_frq)

    if return_results_dict:
        freq_diff_list = np.array(freq_diff_list)
        results_dict = {
            "mae"       : np.mean(np.abs(freq_diff_list)),
            "mse"       : np.mean(freq_diff_list),
            "rmse"      : np.sqrt(np.mean(freq_diff_list**2)),
            "diff"      : diff,
            "diff_dict" : diff_dict,
            "rss_dict"  : rss_dict}
        return rss, log_norm_factor, results_dict
    else:
        return rss, log_norm_factor


def run_geotarget(
    openmm_system, target_strcs, target_zm_list, zmatrix, dihedral_skip, 
    denom_bond, denom_angle, denom_torsion, H_list=list(), return_results_dict=False):

    ### Note that OpenmmEngine will create
    ### a copy.deepcopy copy from openmm_system,
    ### top and target_strc before doing anything
    ### with them.
    if not unit.is_quantity(target_strcs[0]):
        _xyz = target_strcs[0] * _LENGTH
    else:
        _xyz = target_strcs[0]

    if unit.is_quantity(denom_bond):
        denom_bond = denom_bond.value_in_unit(_LENGTH)
    if unit.is_quantity(denom_angle):
        denom_angle = denom_angle.value_in_unit(_ANGLE)
    if unit.is_quantity(denom_torsion):
        denom_torsion = denom_torsion.value_in_unit(_ANGLE)

    engine  = OpenmmEngine(
        openmm_system,
        _xyz)

    N_success = 0

    rss_bond    = 0.0
    rss_angle   = 0.0
    rss_torsion = 0.0

    diff_bond    = 0.0
    diff_angle   = 0.0
    diff_torsion = 0.0

    diff_dict = dict()
    rss_dict  = dict()

    rss  = 0.
    diff = 0.

    N_strcs = len(target_strcs)

    H_constraint = len(H_list) > 0

    N_bonds     = len(zmatrix) - 1
    N_angles    = len(zmatrix) - 2
    N_torsions  = len(zmatrix) - 3

    diff_bonds_list    = list()
    diff_angles_list   = list()
    diff_torsions_list = list()
    for strc_idx in range(N_strcs):
        if not unit.is_quantity(target_strcs[strc_idx]):
            _xyz = target_strcs[strc_idx] * _LENGTH
        else:
            _xyz = target_strcs[strc_idx]

        engine.xyz = _xyz

        success = engine.minimize()

        if success:
            xyz        = engine.xyz.value_in_unit(_LENGTH)
            current_zm = build_z_crds(zmatrix, xyz)
            N_success += 1
        else:
            continue

        target_zm = target_zm_list[strc_idx]
        for z_idx in current_zm:
            z_value = current_zm[z_idx]
            if z_idx == 0:
                continue
            ### Bonds
            if z_idx > 0:
                ### These unit checks are here to guarantee
                ### backwards compatibility and have an extra
                ### layer of unit sanity checking.
                if unit.is_quantity(target_zm[z_idx][0]):
                    target_val = target_zm[z_idx][0].value_in_unit(_LENGTH)
                else:
                    target_val = target_zm[z_idx][0]
                ### If we have constraint bonds to H atoms, we don't
                ### want to include the bond length of those bonds.
                if H_constraint:
                    if not z_idx in H_list:
                        diff       = abs(target_val - z_value[0])
                        diff_bond += diff
                        rss_bond  += diff**2
                        diff_bonds_list.append(diff)
                else:
                    diff       = abs(target_val - z_value[0])
                    diff_bond += diff
                    rss_bond  += diff**2
                    diff_bonds_list.append(diff)
            ### Angles
            if z_idx > 1:
                if unit.is_quantity(target_zm[z_idx][1]):
                    target_val = target_zm[z_idx][1].value_in_unit(_ANGLE)
                else:
                    target_val = target_zm[z_idx][1]
                diff        = abs(target_val - z_value[1])
                diff_angle += diff
                rss_angle  += diff**2
                diff_angles_list.append(diff)
            ### Torsions
            if (z_idx > 2) and (z_idx not in dihedral_skip):
                if unit.is_quantity(target_zm[z_idx][2]):
                    target_val = target_zm[z_idx][2].value_in_unit(_ANGLE)
                else:
                    target_val = target_zm[z_idx][2]
                diff = abs(target_val - z_value[2]) * _ANGLE
                if diff > 180.*unit.degree:
                    diff = diff - 360.*unit.degree
                    diff = abs(diff)
                diff = diff.value_in_unit(_ANGLE)
                diff_torsion += diff
                rss_torsion  += diff**2
                diff_torsions_list.append(diff)

    if N_success != N_strcs:
        rss_bond     = np.inf
        rss_angle    = np.inf
        rss_torsion  = np.inf

        diff_bond    = np.inf * _LENGTH
        diff_angle   = np.inf * _ANGLE
        diff_torsion = np.inf * _ANGLE
        
    else:
        ### RSS calcs
        rss_bond    = 1./(denom_bond)**2    * rss_bond
        rss_angle   = 1./(denom_angle)**2   * rss_angle
        rss_torsion = 1./(denom_torsion)**2 * rss_torsion

        ### DIFF calcs
        diff_bond    = diff_bond    * _LENGTH
        diff_angle   = diff_angle   * _ANGLE
        diff_torsion = diff_torsion * _ANGLE

    rss_dict['bond']    = rss_bond
    rss_dict['angle']   = rss_angle
    rss_dict['torsion'] = rss_torsion

    rss  = rss_bond
    rss += rss_angle
    rss += rss_torsion

    diff_dict['bond']    = diff_bond
    diff_dict['angle']   = diff_angle
    diff_dict['torsion'] = diff_torsion

    diff  = diff_bond._value
    diff += diff_angle._value
    diff += diff_torsion._value

    del engine

    log_norm_factor  = N_bonds    * np.log(denom_bond)
    log_norm_factor += N_angles   * np.log(denom_angle)
    log_norm_factor += N_torsions * np.log(denom_torsion)

    if return_results_dict:
        if diff_bonds_list:
            diff_bonds_list = np.array(diff_bonds_list)
        else:
            diff_bonds_list = np.zeros(1)

        if diff_angles_list:
            diff_angles_list = np.array(diff_angles_list)
        else:
            diff_angles_list = np.zeros(1)

        if diff_torsions_list:
            diff_torsions_list = np.array(diff_torsions_list)
        else:
            diff_torsions_list = np.zeros(1)

        results_dict = {
            "mae_bond"     : np.mean(np.abs(diff_bonds_list)),
            "mse_bond"     : np.mean(diff_bonds_list),
            "rmse_bond"    : np.sqrt(np.mean(diff_bonds_list**2)),

            "mae_angle"    : np.mean(np.abs(diff_angles_list)),
            "mse_angle"    : np.mean(diff_angles_list),
            "rmse_angle"   : np.sqrt(np.mean(diff_angles_list**2)),

            "mae_torsion"  : np.mean(np.abs(diff_torsions_list)),
            "mse_torsion"  : np.mean(diff_torsions_list),
            "rmse_torsion" : np.sqrt(np.mean(diff_torsions_list**2)),

            "rss_bond"     : rss_bond,
            "rss_angle"    : rss_angle,
            "rss_torsion"  : rss_torsion,
            "diff_bond"    : diff_bond,
            "diff_angle"   : diff_angle,
            "diff_torsion" : diff_torsion,
            "rss_dict"     : rss_dict,
            "diff_dict"    : diff_dict,
            "diff"         : diff}
        return rss, log_norm_factor, results_dict
    else:
        return rss, log_norm_factor


def run_energytarget(
    openmm_system, target_strcs, target_energies, 
    denom_ene, minimize, restraint_atom_indices=list(), restraint_k=list(),
    reference_to_lowest=True, ene_weighting=True, ene_cutoff=True, mean_shift=False,
    return_results_dict=False):

    if not unit.is_quantity(target_strcs[0]):
        _xyz = target_strcs[0] * _LENGTH
    else:
        _xyz = target_strcs[0]

    if unit.is_quantity(target_energies):
        target_energies = target_energies.value_in_unit(_ENERGY_PER_MOL)

    if unit.is_quantity(denom_ene):
        denom_ene = denom_ene.value_in_unit(_ENERGY_PER_MOL)

    engine  = OpenmmEngine(
        openmm_system,
        _xyz)

    diff_dict = dict()
    rss_dict  = dict()
    vals_dict = dict()

    rss  = 0.
    diff = 0.
    
    N_restraints = len(restraint_atom_indices)
    N_strcs = len(target_strcs)

    ### Find weights for energies.
    ### See Parsley paper:
    ### doi.org/10.26434/chemrxiv.13082561.v2
    if reference_to_lowest:
        target_denom = np.ones(N_strcs, dtype=float)
        argmin_target_energy  = np.argmin(target_energies)
        delta_target_energies = np.array(target_energies) - target_energies[argmin_target_energy]
    else:
        target_denom = np.ones((N_strcs, N_strcs), dtype=float)
        _delta_target_energies = np.reshape(target_energies, (N_strcs, 1))
        delta_target_energies  = _delta_target_energies - _delta_target_energies.transpose()

    if ene_cutoff:
        _ene_unit  = (1.*unit.kilocalorie_per_mole).value_in_unit(_ENERGY_PER_MOL)
        ### Corresponds to 0.1 Ha
        _upper     = 62.5 * _ene_unit
        valids = np.where(delta_target_energies > _upper)
        target_denom[valids] = 0.
    elif ene_weighting:
        _ene_unit  = (1.*unit.kilocalorie_per_mole).value_in_unit(_ENERGY_PER_MOL)
        _lower     = _ene_unit
        _upper     = 5. * _ene_unit
        valids1 = np.where(delta_target_energies < _lower)
        valids2 = np.where((_lower < delta_target_energies) * ( _upper > delta_target_energies))
        valids3 = np.where(delta_target_energies > _upper)
        target_denom[valids1] = 1.
        target_denom[valids2] = 1./np.sqrt(_lower + (delta_target_energies[valids2] - _lower)**2)
        target_denom[valids3] = 0.

    if mean_shift:
        delta_target_energies -= np.mean(delta_target_energies)

   # _target_denom = target_denom / np.sum(target_denom, axis=0)
   # target_denom  = _target_denom

    ### Important: Add the forces only through the engine.
    ### The original openmm_system object must remain untouched.
    if N_restraints > 0:
        for restraint_id in range(N_restraints):
            atm_idxs  = restraint_atom_indices[restraint_id]
            k         = restraint_k[restraint_id]
            ### eq_value for first structure
            eq_value  = restraint_eq_value[min_ene_arg][restraint_id]
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

    energy_list = np.zeros(N_strcs)
    for strc_idx in range(N_strcs):
        if not unit.is_quantity(target_strcs[strc_idx]):
            _xyz = target_strcs[strc_idx] * _LENGTH
        else:
            _xyz = target_strcs[strc_idx]

        engine.xyz = _xyz
        if minimize:
            engine.minimize()
        energy_list[strc_idx] = engine.pot_ene.value_in_unit(_ENERGY_PER_MOL)

    if mean_shift:
        delta_energies = energy_list - np.mean(energy_list)
    else:
        if reference_to_lowest:
            delta_energies = energy_list - energy_list[argmin_target_energy]
        else:
            _delta_energies = np.reshape(energy_list, (N_strcs, 1))
            delta_energies  = _delta_energies - _delta_energies.transpose()

    _diff = (delta_energies - delta_target_energies) * target_denom
    if not reference_to_lowest:
        _diff_avg = _diff / np.sum(target_denom)
        _rss = _diff_avg**2 / denom_ene**2
    else:
        _rss = _diff**2 / denom_ene**2

    for strc_idx in range(N_strcs):
        if reference_to_lowest:
            diff_dict[strc_idx] = _diff[strc_idx] * _ENERGY_PER_MOL
            rss_dict[strc_idx]  = _rss[strc_idx]
            vals_dict[strc_idx] = delta_energies[strc_idx]
        else:
            diff_dict[strc_idx] = np.mean(_diff[strc_idx]) * _ENERGY_PER_MOL
            rss_dict[strc_idx]  = np.mean(_rss[strc_idx])
            vals_dict[strc_idx] = np.mean(delta_energies[strc_idx])

    diff = np.sum(_diff) * _ENERGY_PER_MOL
    rss  = np.sum(_rss)
    log_norm_factor = np.log(denom_ene)

    del engine

    if return_results_dict:
        results_dict = {
            "mae"          : np.mean(np.abs(_diff)),
            "mse"          : np.mean(_diff),
            "rmse"         : np.sqrt(np.mean(_diff**2)),
            "vals_dict"    : vals_dict,
            "rss_dict"     : rss_dict,
            "diff_dict"    : diff_dict,
            "diff"         : diff}
        return rss, log_norm_factor, results_dict
    else:
        return rss, log_norm_factor


def run_forcematchingtarget(
    openmm_system, target_strcs, target_forces, target_energies, denom_force, 
    ene_weighting=True, ene_cutoff=True, return_results_dict=False):

    import copy

    if not unit.is_quantity(target_strcs[0]):
        _xyz = target_strcs[0] * _LENGTH
    else:
        _xyz = target_strcs[0]

    if unit.is_quantity(denom_force):
        denom_force = denom_force.value_in_unit(_FORCE)

    engine  = OpenmmEngine(
        openmm_system,
        _xyz)

    diff_dict = dict()
    rss_dict  = dict()

    N_atoms = len(target_strcs[0])
    N_strcs = len(target_strcs)

    target_denom = np.ones(N_strcs, dtype=float)
    argmin_target_energy  = np.argmin(target_energies)
    delta_target_energies = np.array(target_energies) - target_energies[argmin_target_energy]

    if ene_cutoff:
        _ene_unit  = (1.*unit.kilocalorie_per_mole).value_in_unit(_ENERGY_PER_MOL)
        ### Corresponds to 0.1 Ha
        _upper     = 62.5 * _ene_unit
        valids = np.where(delta_target_energies > _upper)
        target_denom[valids] = 0.
    elif ene_weighting:
        _ene_unit  = (1.*unit.kilocalorie_per_mole).value_in_unit(_ENERGY_PER_MOL)
        _lower     = _ene_unit
        _upper     = 5. * _ene_unit
        valids1 = np.where(delta_target_energies < _lower)
        valids2 = np.where((_lower < delta_target_energies) * ( _upper > delta_target_energies))
        valids3 = np.where(delta_target_energies > _upper)
        target_denom[valids1] = 1.
        target_denom[valids2] = 1./np.sqrt(_lower + (delta_target_energies[valids2] - _lower)**2)
        target_denom[valids3] = 0.

    rss  = 0.
    diff = 0.

    diff_list = list()
    for strc_idx in range(N_strcs):
        target_strc  = target_strcs[strc_idx]
        target_force = target_forces[strc_idx]
        if not unit.is_quantity(target_strcs[strc_idx]):
            _xyz = target_strcs[strc_idx] * _LENGTH
        else:
            _xyz = target_strcs[strc_idx]
        engine.xyz   = _xyz
        forces       = engine.forces.value_in_unit(_FORCE).flatten()
        ### This same as computing length of diff vector
        ### and then taking square
        _diff        = abs(forces - target_force.flatten()) * target_denom[strc_idx]
        _diff2       = np.sum(_diff**2)
        _rss         = _diff2 / denom_force**2
        _rss        /= float(N_atoms * 3)

        diff_dict[strc_idx] = _diff.tolist() * _FORCE
        rss_dict[strc_idx]  = _rss.tolist()

        rss  += np.sum(rss)
        diff += np.sum(_diff)

        diff_list.extend(_diff.flatten().tolist())

    diff *= _FORCE

    log_norm_factor  = N_atoms * 3 * np.log(denom_force)

    del engine

    if return_results_dict:
        diff_list = np.array(diff_list)
        results_dict = {
            "mae"          : np.mean(np.abs(diff_list)),
            "mse"          : np.mean(diff_list),
            "rmse"         : np.sqrt(np.mean(diff_list**2)),
            "rss_dict"     : rss_dict,
            "diff_dict"    : diff_dict,
            "diff"         : diff}
        return rss, log_norm_factor, results_dict
    else:
        return rss, log_norm_factor


def run_forceprojectionmatchingtarget(
    openmm_system, target_strcs, target_force_projection, B_flat_list, zmatrix,
    denom_bond, denom_angle, denom_torsion, H_list=list(), return_results_dict=False):

    ### Note that OpenmmEngine will create
    ### a copy.deepcopy copy from openmm_system,
    ### top and target_strc before doing anything
    ### with them.
    if not unit.is_quantity(target_strcs[0]):
        _xyz = target_strcs[0] * _LENGTH
    else:
        _xyz = target_strcs[0]

    engine  = OpenmmEngine(
        openmm_system,
        _xyz)

    rss_bond    = 0.0
    rss_angle   = 0.0
    rss_torsion = 0.0

    diff_bond    = 0.0
    diff_angle   = 0.0
    diff_torsion = 0.0

    diff_dict = dict()
    rss_dict  = dict()

    rss  = 0.
    diff = 0.

    N_bonds     = len(zmatrix) - 1
    N_angles    = len(zmatrix) - 2
    N_torsions  = len(zmatrix) - 3

    N_strcs = len(target_strcs)
    H_constraint = len(H_list) > 0
    diff_bonds_list    = list()
    diff_angles_list   = list()
    diff_torsions_list = list()
    for strc_idx in range(N_strcs):

        target_force = target_force_projection[strc_idx]
        if not unit.is_quantity(target_strcs[strc_idx]):
            _xyz = target_strcs[strc_idx] * _LENGTH
        else:
            _xyz = target_strcs[strc_idx]

        engine.xyz = _xyz
        forces     = engine.forces.value_in_unit(_FORCE).flatten()

        B_flat  = B_flat_list[strc_idx]
        force_q = build_grad_projection(
            B_flat, 
            zmatrix,
            forces * _FORCE,
            as_dict=True)

        for z_idx, force_values in force_q.items():
            if z_idx == 0:
                continue
            ### Bonds
            if z_idx > 0:
                ### If we have constraint bonds to H atoms, we don't
                ### want to include the bond length of those bonds.
                if H_constraint:
                    if not z_idx in H_list:
                        diff       = abs(target_force[z_idx][0] - force_values[0])
                        diff_bond += diff
                        rss_bond  += diff**2
                else:
                    diff       = abs(target_force[z_idx][0] - force_values[0])
                    diff_bond += diff
                    rss_bond  += diff**2
                diff_bonds_list.append(diff)
            ### Angles
            if z_idx > 1:
                diff        = abs(target_force[z_idx][1] - force_values[1])
                diff_angle += diff
                rss_angle  += diff**2
                diff_angles_list.append(diff)
            ### Torsions
            if z_idx > 2:
                diff = abs(target_force[z_idx][2] - force_values[2])
                diff_torsion += diff
                rss_torsion  += diff**2
                diff_torsions_list.append(diff)

    diff_bond    *= _LENGTH
    diff_angle   *= _ANGLE
    diff_torsion *= _ANGLE

    ### RSS calcs
    rss_bond     /= denom_bond**2
    rss_angle    /= denom_angle**2
    rss_torsion  /= denom_torsion**2

    rss_dict['bond']    = rss_bond
    rss_dict['angle']   = rss_angle
    rss_dict['torsion'] = rss_torsion

    rss  = rss_bond
    rss += rss_angle
    rss += rss_torsion

    diff_dict['bond']    = diff_bond
    diff_dict['angle']   = diff_angle
    diff_dict['torsion'] = diff_torsion

    diff  = diff_bond._value
    diff += diff_angle._value
    diff += diff_torsion._value

    del engine

    log_norm_factor  = N_bonds * np.log(denom_bond)
    log_norm_factor += N_angles * np.log(denom_angle)
    log_norm_factor += N_torsions * np.log(denom_torsion)

    if return_results_dict:

        diff_bonds_list    = np.array(diff_bonds_list)
        diff_angles_list   = np.array(diff_angles_list)
        diff_torsions_list = np.array(diff_torsions_list)

        results_dict = {
            "mae_bond"     : np.mean(np.abs(diff_bonds_list)),
            "mse_bond"     : np.mean(diff_bonds_list),
            "rmse_bond"    : np.sqrt(np.mean(diff_bonds_list**2)),

            "mae_angle"    : np.mean(np.abs(diff_angles_list)),
            "mse_angle"    : np.mean(diff_angles_list),
            "rmse_angle"   : np.sqrt(np.mean(diff_angles_list**2)),

            "mae_torsion"  : np.mean(np.abs(diff_torsions_list)),
            "mse_torsion"  : np.mean(diff_torsions_list),
            "rmse_torsion" : np.sqrt(np.mean(diff_torsions_list**2)),

            "rss_bond"     : rss_bond,
            "rss_angle"    : rss_angle,
            "rss_torsion"  : rss_torsion,
            "diff_bond"    : diff_bond,
            "diff_angle"   : diff_angle,
            "diff_torsion" : diff_torsion,
            "rss_dict"     : rss_dict,
            "diff_dict"    : diff_dict,
            "diff"         : diff}
        return rss, log_norm_factor, results_dict
    else:
        return rss, log_norm_factor


def target_worker_local(openmm_system_dict, target_dict, return_results_dict=True, strip_units=True):

    import ray
    import numpy as np
    from .tools import transform_unit

    target_method_dict = {
        "GeoTarget"                     : run_geotarget,
        "NormalModeTarget"              : run_normalmodetarget,
        "EnergyTarget"                  : run_energytarget,
        "ForceMatchingTarget"           : run_forcematchingtarget,
        "ForceProjectionMatchingTarget" : run_forceprojectionmatchingtarget
        }

    logP_likelihood  = 0.
    results_all_dict = dict()
    for key in openmm_system_dict:
        sys_name, sys_key = key
        #args_dict_list = ray.get(target_dict[sys_name])
        args_dict_list = target_dict[sys_name]
        N_tgt = len(args_dict_list)
        if return_results_dict:
            results_all_dict[key] = dict()
            results_dict = dict()
        else:
            results_all_dict[key] = 0.
        openmm_system = openmm_system_dict[key]

        for target_idx in range(N_tgt):
            target_args, target_name = args_dict_list[target_idx]
            target_method = target_method_dict[target_name]
            try:
                results = target_method(
                    openmm_system, **target_args, return_results_dict=return_results_dict)
                if return_results_dict:
                    rss, log_norm_factor, _results_dict = results
                    _logP_likelihood = -log_norm_factor - 0.5 * np.log(2.*np.pi) - 0.5 * rss
                    logP_likelihood += _logP_likelihood
                    results_dict[(target_name,target_idx)] = _results_dict
                else:
                    rss, log_norm_factor = results
                    _logP_likelihood = -log_norm_factor - 0.5 * np.log(2.*np.pi) - 0.5 * rss
                    logP_likelihood += _logP_likelihood
                    results_all_dict[key] += _logP_likelihood
                
            except:
                if _VERBOSE:
                    import traceback
                    print(traceback.format_exc())
                logP_likelihood = -np.inf
                results_all_dict[key] = -np.inf
                if return_results_dict:
                    results_dict[(target_name,target_idx)] = dict()

        if return_results_dict:
            results_all_dict[key] = results_dict

    if strip_units:
        results_all_dict = transform_unit(results_all_dict)

    return logP_likelihood, results_all_dict


@ray.remote
def target_worker(openmm_system_dict, target_dict, return_results_dict=True, strip_units=True):
    return target_worker_local(
            openmm_system_dict, target_dict, return_results_dict, strip_units)


@ray.remote
class TargetComputer(object):

    def __init__(
        self, 
        system_list, 
        target_type_list=None,
        error_factor=1.):

        import ray
        from .system import SystemManager

        self._target_dict = dict()

        if target_type_list == None:
            target_type_list = None
        elif len(target_type_list) == 0:
            target_type_list = None
        else:
            target_type_list = tuple(target_type_list)

        if isinstance(system_list, list):
            _system_list = system_list
        elif isinstance(system_list, SystemManager):
            _system_list = system_list._system_list

        for sys in _system_list:
            target_list = list()
            for target in sys.target_list:            
                if target_type_list == None:
                    target_args = target.get_args()
                    for arg in target_args:
                        if arg.startswith("denom"):
                            val = target_args[arg]
                            target_args[arg] = val * error_factor
                    target_name = target.get_target_name()
                    target_list.append(
                        (target_args, target_name))
                else:
                    if isinstance(target, target_type_list):
                        target_args = target.get_args()
                        for arg in target_args:
                            if arg.startswith("denom"):
                                val = target_args[arg]
                                target_args[arg] = val * error_factor
                        target_name = target.get_target_name()
                        target_list.append(
                            (target_args, target_name))
            self._target_dict[sys.name] = target_list

    @property
    def target_dict(self):

        return self._target_dict

    def __call__(
        self, 
        openmm_system_dict,
        return_results_dict=True,
        local=False,
        strip_units=True):

        if local:
            result = target_worker_local(
                openmm_system_dict, 
                {sys_name : self.target_dict[sys_name] for sys_name, sys_key in openmm_system_dict},
                return_results_dict,
                strip_units)
            return result
        else:
            worker_id = target_worker.remote(
                openmm_system_dict, 
                {sys_name : self.target_dict[sys_name] for sys_name, sys_key in openmm_system_dict},
                return_results_dict,
                strip_units)
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
    eigvals, eigvecs = np.linalg.eigh(mass_weighted_hessian)
    negative_idxs = np.where(eigvals < 0.)
    ### freqs/eigvals in 1/s * 10.e-12
    freqs         = np.sqrt(np.abs(eigvals))
    freqs[negative_idxs] *= -1

    ### remove the 6 freqs with smallest abs value and corresponding normal modes
    n_remove = 5 if N_atoms == 2 else 6
    larger_freq_idxs = np.sort(
        np.argpartition(
            np.abs(freqs),
            n_remove
            )[n_remove:]
        )
    freqs = freqs[larger_freq_idxs]
    eigvecs = eigvecs[:,larger_freq_idxs]
    ### Convert 1/s * 10.e-12 to 1/cm
    ### >>> a = 1. * unit.seconds**-1
    ### >>> a = a * unit.constants.SPEED_OF_LIGHT_C**-1
    ### >>> a = a.in_units_of(unit.centimeter**-1)
    ### >>> print(a)
    freqs *= 33.3564095198152 * unit.centimeters**-1
    ### Convert to angular wavenumber to wavelength of wave
    freqs *= 0.5 / np.pi

    del invert_sqrt_mass, mass_weighted_hessian

    return freqs.in_units_of(_WAVENUMBER), eigvecs


class Target(object):

    def __init__(self, target_dict, system):

        self.rss       = 0.
        self.diff      = 0.
        self.diff_dict = OrderedDict()
        self.rss_dict  = OrderedDict()

        self.target_strcs = list()
        for target_strc in target_dict["structures"]:
            if unit.is_quantity(target_strc):
                target_strc = target_strc.value_in_unit(_LENGTH)
            self.target_strcs.append(np.array(target_strc))
        self.target_strcs = np.array(self.target_strcs)

        self.N_strcs = len(self.target_strcs)

        self.target_rdmol = copy.deepcopy(
            target_dict["rdmol"]
            )

        self._rdmol         = system.rdmol
        self._N_atoms       = system.N_atoms

        self._align_graph()

    @property
    def log_norm_factor(self):
        return 0.

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
        self.masses = np.array(self.masses)

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

    def get_args(self):
        return dict()

    def get_method(self):
        return None

    def getattr(self, name):
        return getattr(self, name)
    
    def setattr(self, name, value):
        setattr(self, name, value)

    @property
    def N_atoms(self):
        return self._N_atoms

    @property
    def rdmol(self):
        return self._rdmol

    def run(self, openmm_system):

        args = self.get_args()
        method = self.get_method()
        _, _, results_dict = method(
            openmm_system=openmm_system,
            return_results_dict=True,
            **args)
        for key in results_dict:
            setattr(self, key, results_dict[key])


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
        self.denom_bond = self.denom_bond.value_in_unit(_LENGTH)
        self.denom_angle = self.denom_angle.value_in_unit(_ANGLE)
        self.denom_torsion = self.denom_torsion.value_in_unit(_ANGLE)

    @property
    def log_norm_factor(self):

        value  = self.N_bonds    * np.log(self.denom_bond)
        value += self.N_angles   * np.log(self.denom_angle)
        value += self.N_torsions * np.log(self.denom_torsion)

        return value

    def configure_target(self):

        self.zm = ZMatrix(self.rdmol)
        self.dihedral_skip = list()

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
            _target_zm = self.zm.build_z_crds(
                    self.target_strcs[strc_idx]*_LENGTH,
                    with_units=False)
            self.target_zm.append(_target_zm)
            for z_idx in self.zm.z:
                if z_idx > 2:
                    aidxs = self.zm.z[z_idx]
                    a = self.target_strcs[strc_idx][aidxs[0]]
                    b = self.target_strcs[strc_idx][aidxs[1]]
                    c = self.target_strcs[strc_idx][aidxs[2]]
                    d = self.target_strcs[strc_idx][aidxs[3]]
                    b0 = a-b
                    b1 = c-b
                    b2 = d-c
                    b0 /= np.linalg.norm(b0)
                    b1 /= np.linalg.norm(b1)
                    b2 /= np.linalg.norm(b2)
                    check1 = np.abs(np.dot(b0, b1)) > 0.999
                    check2 = np.abs(np.dot(b2, b1)) > 0.999
                    if check1 or check2:
                        self.dihedral_skip.append(z_idx)

    def get_args(self):

        args = {
            "target_strcs"    : self.target_strcs,
            "target_zm_list"  : self.target_zm,
            "zmatrix"         : self.zm.z,
            "dihedral_skip"   : self.dihedral_skip,
            "denom_bond"      : self.denom_bond,
            "denom_angle"     : self.denom_angle,
            "denom_torsion"   : self.denom_torsion}
        if self.H_constraint:
            args["H_list"] = self.H_list
        else:
            args["H_list"] = list()

        return args

    def get_target_name(self):

        return "GeoTarget"

    def get_method(self):

        return run_geotarget


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
        self.denom_frq    = 200.

        if "denom_freq" in target_dict:
            self.denom_frq = target_dict["denom_freq"]
            self.denom_frq = self.denom_frq.value_in_unit(_WAVENUMBER)

        self.target_hessian = list()
        for hes_idx in range(len(target_dict["hessian"])):
            hessian  = np.array(target_dict["hessian"][hes_idx]._value)
            hessian *= target_dict["hessian"][hes_idx].unit
            self.target_hessian.append(
                hessian.value_in_unit(_FORCE/_LENGTH*unit.mole))
        self.target_hessian = np.array(self.target_hessian)

        assert self.N_strcs == len(self.target_hessian)

        self.configure_target()

    @property
    def log_norm_factor(self):

        value  = self.N_freqs * np.log(self.denom_frq)

        return value

    def configure_target(self):

        ### 1) Configure all the basic stuff
        ### ================================

        if self.N_atoms == 2:
            self.N_freqs = 5
        else:
            self.N_freqs = self.N_atoms * 3 - 6
        self.target_freqs = list()
        self.target_modes = list()

        ### 2) Compute vib frequencies
        ### ==========================
        for hes_idx in range(self.N_strcs):
            freqs, modes = compute_freqs(
                self.target_hessian[hes_idx] *_FORCE/_LENGTH*unit.mole,
                self.masses * _ATOMIC_MASS)
            self.target_freqs.append(
                freqs.value_in_unit(_WAVENUMBER))
            self.target_modes.append(
                modes)
        self.target_freqs = np.array(self.target_freqs)
        self.target_modes = np.array(self.target_modes)

    def get_args(self):

        args = {
            "target_strcs"    : self.target_strcs,
            "target_freqs"    : self.target_freqs,
            ### We dont want to get the target modes
            ### this is only necessary with permute=True.
            ### However, this does not improve results.
            "target_modes"    : list(),
            "minimize"        : self.minimize,
            "masses"          : self.masses,
            "denom_frq"       : self.denom_frq,
            "permute"         : False}

        return args

    def get_target_name(self):

        return "NormalModeTarget"

    def get_method(self):

        return run_normalmodetarget



class ForceProjectionMatchingTarget(Target):

    def __init__(
        self,
        target_dict: dict,
        system: system.System
        ):

        super().__init__(target_dict, system)

        self.denom_bond    = 1.
        self.denom_angle   = 8.
        self.denom_torsion = 20
        self.H_constraint  = True

        ### For starters, these are just the same denom as
        ### for the geo targety

        if "denom_bond" in target_dict:
            self.denom_bond = target_dict["denom_bond"]
            self.denom_bond = self.denom_bond.value_in_unit(_FORCE)
        if "denom_angle" in target_dict:
            self.denom_angle = target_dict["denom_angle"]
            self.denom_angle = self.denom_angle.value_in_unit(_FORCE)
        if "denom_torsion" in target_dict:
            self.denom_torsion = target_dict["denom_torsion"]
            self.denom_torsion = self.denom_torsion.value_in_unit(_FORCE)

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
        self.target_forces = np.array(self.target_forces)

        assert len(self.target_strcs) ==  len(self.target_forces)

        self.configure_target()

    @property
    def log_norm_factor(self):

        value  = self.N_bonds * np.log(self.denom_bond)
        value += self.N_angles * np.log(self.denom_angle)
        value += self.N_torsions * np.log(self.denom_torsion)

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

            cart_crds  = self.target_strcs[strc_idx]
            cart_force = self.target_forces[strc_idx]

            B_flat = self.zm.build_wilson_b(
                cart_crds * _LENGTH, 
                as_dict=False
                )
            force_q = self.zm.build_grad_projection(
                B_flat, 
                cart_force * _FORCE,
                as_dict=True
                )
            self.target_force_projection.append(force_q)
            self.B_flat_list.append(B_flat)

    def get_args(self):

        args = {
            "target_strcs"             : self.target_strcs,
            "target_force_projection"  : self.target_force_projection,
            "B_flat_list"              : self.B_flat_list,
            "zmatrix"                  : self.zm.z,
            "denom_bond"               : self.denom_bond,
            "denom_angle"              : self.denom_angle,
            "denom_torsion"            : self.denom_torsion}
        if self.H_constraint:
            args["H_list"] = self.H_list
        else:
            args["H_list"] = list()

        return args

    def get_target_name(self):

        return "ForceProjectionMatchingTarget"

    def get_method(self):

        return run_forceprojectionmatchingtarget


class ForceMatchingTarget(Target):

    def __init__(
        self,
        target_dict: dict,
        system: system.System):

        super().__init__(target_dict, system)

        self.denom_force = 1.0e+4

        if "denom_force" in target_dict:
            self.denom_force = target_dict["denom_force"]
            self.denom_force = self.denom_force.value_in_unit(_FORCE)

        self.ene_weighting  = True
        if "ene_weighting" in target_dict:
            self.ene_weighting  = target_dict["ene_weighting"]

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

        self.target_forces = np.array(self.target_forces)

        assert len(self.target_strcs) ==  len(self.target_forces)

    @property
    def log_norm_factor(self):

        value  = self.N_atoms * 3 * np.log(self.denom_force)

        return value

    def get_args(self):

        args = {
            "target_strcs"   : self.target_strcs,
            "target_forces"  : self.target_forces,
            "target_energies": self.target_energies,
            "ene_weighting"  : self.ene_weighting,
            "denom_force"    : self.denom_force}

        return args

    def get_target_name(self):

        return "ForceMatchingTarget"

    def get_method(self):

        return run_forcematchingtarget


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

        self.denom_ene = 1.

        self.minimize  = False
        if "minimize" in target_dict:
            self.minimize  = target_dict["minimize"]

        self.ene_weighting  = True
        if "ene_weighting" in target_dict:
            self.ene_weighting  = target_dict["ene_weighting"]

        if "denom_ene" in target_dict:
            self.denom_ene = target_dict["denom_ene"]
            self.denom_ene = self.denom_ene.value_in_unit(_ENERGY_PER_MOL)

        self.reference_to_lowest = True
        if "reference_to_lowest" in target_dict:
            self.reference_to_lowest = target_dict["reference_to_lowest"]

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

        value  = np.log(self.denom_ene)

        return value

    def get_args(self):

        if not hasattr(self, "reference_to_lowest"):
            self.reference_to_lowest = True

        args = {
            "target_strcs"           : self.target_strcs,
            "target_energies"        : self.target_energies,
            "denom_ene"              : self.denom_ene,
            "minimize"               : self.minimize,
            "restraint_atom_indices" : self.restraint_atom_indices,
            "restraint_k"            : self.restraint_k,
            "reference_to_lowest"    : self.reference_to_lowest,
            "ene_weighting"          : self.ene_weighting}

        return args

    def get_target_name(self):

        return "EnergyTarget"

    def get_method(self):

        return run_energytarget
