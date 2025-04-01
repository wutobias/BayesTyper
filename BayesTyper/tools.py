#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================
import time
import numpy as np
import copy
from rdkit.Chem import AllChem as Chem
from openmm import unit
from openff.toolkit.topology import Molecule

import warnings

from .molgraphs import ZMatrix
from .molgraphs import (pts_to_bond,
                        pts_to_angle,
                        pts_to_dihedral
                        )

from .engines import OpenmmEngine

try:
    import ray
    HAS_RAY = True
except:
    HAS_RAY = False

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================

from .constants import (_FORCE_AU,
                        _ENERGY_AU,
                        _LENGTH_AU,
                        _LENGTH,
                        _ANGLE,
                        _UNIT_QUANTITY,
                        _DEFAULT_FF)

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================

get_atomic_symbol = Chem.GetPeriodicTable().GetElementSymbol
get_atomic_weight = Chem.GetPeriodicTable().GetAtomicWeight
get_atomic_number = Chem.GetPeriodicTable().GetAtomicNumber


def benchmark_systems(
    system_list, KEY_LIST = ["rmse"], 
    print_output=True, only_global=False,
    local=False):

    from BayesTyper import targets
    from BayesTyper import engines
    import ray

    targetcomputer = targets.TargetComputer.remote(system_list)
    openmm_system_dict = dict()
    for i, sys in enumerate(system_list):
        openmm_system_dict[sys.name,0] = sys.openmm_system
    if local:
        logP, results_dict = ray.get(
            targetcomputer.__call__.remote(
                openmm_system_dict, 
                return_results_dict=True, 
                local=True))
    else:
        worker_id = ray.get(
            targetcomputer.__call__.remote(
                openmm_system_dict, 
                return_results_dict=True, 
                local=False))
        logP, results_dict = ray.get(worker_id)

    if print_output:
        output_str_list = list()
        OVERALL_STATS = dict()
        for sys_key in results_dict:
            sysname, sys_idx = sys_key
            if not only_global:
                output_str_list.append(f"{sysname}")
                output_str_list.append("".join(["="]*len(sysname)))
                output_str_list.append("")
            ### Get alphabetical sorting
            target_name_dict = dict()
            for target_key in results_dict[sys_key]:
                target_name, target_idx = target_key
                target_name_dict[target_key] = target_name
            sorted_key_list = sorted(
                target_name_dict.keys(), key=target_name_dict.get)
            for target_key in sorted_key_list:
                target_name, target_idx = target_key
                if not only_global:
                    output_str_list.append(" "*4 + target_name)
                    output_str_list.append(" "*4 + "".join(["~"]*len(target_name)))
                for error_key in results_dict[sys_key][target_key]:
                    check  = any(
                        [error_key.startswith(v) for v in KEY_LIST])
                    if check:
                        val = results_dict[sys_key][target_key][error_key]
                        error_name = error_key.replace("_"," ").upper()
                        if not only_global:
                            output_str_list.append(" "*8 + f"{error_name:15s}" + f"{val:4.2f}")
                        if target_name not in OVERALL_STATS:
                            OVERALL_STATS[target_name] = dict()
                        if error_name not in OVERALL_STATS[target_name]:
                            OVERALL_STATS[target_name][error_name] = list()
                        OVERALL_STATS[target_name][error_name].append(val)     
                if not only_global:
                    output_str_list.append("")

        import numpy as np
        for target_name in OVERALL_STATS:
            output_str_list.append(
                "GLOBAL ESTIMATE FOR " + target_name)
            output_str_list.append(
                "".join(["="]*20) + "".join(["="]*len(target_name)))
            for error_name in OVERALL_STATS[target_name]:
                val  = np.mean(OVERALL_STATS[target_name][error_name])
                output_str_list.append(
                    " "*8 + f"{error_name:15s}" + f"{val:4.2f}")
            output_str_list.append("")
        print(
            "\n".join(output_str_list))

    return results_dict


def get_rmsd(system_list, optdataset_dict):

    from BayesTyper.engines import OpenmmEngine
    from BayesTyper.constants import _LENGTH_AU
    from openmm import app
    from openmm import unit
    import mdtraj as md
    import numpy as np

    N_sys = len(system_list)
    rmsd_dict = dict()
    for sys_idx in range(N_sys):
        sys = system_list[sys_idx]
        smiles = sys.name
        top    = md.Topology.from_openmm(
                sys.top.to_openmm())
        ### First frame one will be the reference frame
        ### Second frame will be the query frame
        traj     = md.Trajectory(
                np.zeros((2, sys.N_atoms, 3)), top)
        rmsd_dict[smiles] = list()
        for conf_i in optdataset_dict[smiles]:
            if not "final_geo"in optdataset_dict[smiles][conf_i]:
                continue
            xyz = optdataset_dict[smiles][conf_i]["final_geo"]
            if not unit.is_quantity(xyz):
                xyz *= _LENGTH_AU
            engine = OpenmmEngine(
                sys.openmm_system,
                xyz)
            engine.set_xyz(xyz)
            engine.minimize()
            traj.xyz[0] = xyz.value_in_unit(unit.nanometer)
            traj.xyz[1] = engine.xyz.value_in_unit(unit.nanometer)
            rmsd = md.lprmsd(traj, traj, frame=0)
            rmsd_dict[smiles].append(rmsd.tolist()[1])

    return rmsd_dict


def write_pdb(system_list, optdataset_dict):

    from BayesTyper.engines import OpenmmEngine
    from BayesTyper.constants import _LENGTH_AU
    from openmm import app
    from openmm import unit

    N_sys = len(system_list)
    for sys_idx in range(N_sys):
        sys = system_list[sys_idx]
        smiles = sys.name
        for conf_i in optdataset_dict[smiles]:
            if not "final_geo"in optdataset_dict[smiles][conf_i]:
                continue
            xyz = optdataset_dict[smiles][conf_i]["final_geo"]
            if not unit.is_quantity(xyz):
                xyz *= _LENGTH_AU
            engine = OpenmmEngine(
                sys.openmm_system, 
                xyz
            )
            engine.set_xyz(xyz)
            engine.minimize()
            
            with open(f"./sys-{sys_idx}-conf-{conf_i}-mm.pdb", "w") as fopen:            
                app.PDBFile.writeFile(
                    sys.top.to_openmm(),
                    engine.xyz,
                    fopen
                )

            with open(f"./sys-{sys_idx}-conf-{conf_i}-qm.pdb", "w") as fopen:            
                app.PDBFile.writeFile(
                    sys.top.to_openmm(),
                    xyz,
                    fopen
                )
            

def get_plots(
    systemmanager,
    title_dict=dict(),
    name_reference_level_of_theory="b3lyp/dzvp",
    add_plots=0, 
    generate_plots=False,
    N_col=3,
    verbose=False,
    optdataset_dict_path=None, 
    torsiondataset_dict_path=None,
    skip_torsion=False,
    skip_optgeo=False,
    skip_vib=False,
    skip_offeq=False,
    enecutoff5=False,
    vib_minimize=True,):

    import pickle
    import numpy as np

    from BayesTyper.constants import (
        _ATOMIC_MASS, 
        _ENERGY_PER_MOL, 
        _ENERGY_AU
        )
    from BayesTyper.constants import (
        _FORCE_AU, 
        _LENGTH_AU, 
        _ENERGY_AU, 
        _LENGTH, 
        _ANGLE, 
        _FORCE, 
        _ENERGY_PER_MOL, 
        _WAVENUMBER
        )
    from BayesTyper.engines import OpenmmEngine
    from BayesTyper.system import SystemManager
    from BayesTyper.targets import compute_freqs
    import matplotlib.pyplot as plt
    from openmm import unit
    from .molgraphs import ZMatrix

    if optdataset_dict_path == None:
        with open("./CH_dataset_Butane_MM.pickle", "rb") as fopen:
            optdataset_dict = pickle.load(fopen)
    elif isinstance(optdataset_dict_path, dict):
        optdataset_dict = optdataset_dict_path
    else:
        with open(optdataset_dict_path, "rb") as fopen:
            optdataset_dict = pickle.load(fopen)
    if torsiondataset_dict_path == None:
        with open("./CH_torsiondataset_Butane_MM.pickle", "rb") as fopen:
            torsiondataset_dict = pickle.load(fopen)
    elif isinstance(torsiondataset_dict_path, dict):
        torsiondataset_dict = torsiondataset_dict_path
    else:
        with open(torsiondataset_dict_path, "rb") as fopen:
            torsiondataset_dict = pickle.load(fopen)
    
    if isinstance(systemmanager, list):
        systemlist = systemmanager
    elif isinstance(systemmanager, SystemManager):
        systemlist = systemmanager._system_list
    else:
        raise ValueError(
            "Type of systemmanager not understood."
        )
    N_sys = len(systemlist)

    if generate_plots:
        ### Determine number of plots to make
        N_plots = add_plots
        for sys_idx in range(N_sys):
            sys = systemlist[sys_idx]
            smiles = sys.name
            if not skip_vib:
                ### Freq.
                N_plots += 1
            if not skip_offeq:
                ### Off-Equ.
                N_plots += 1
            if not skip_optgeo:
                ### OptGeo
                N_plots += 3
            if not skip_torsion and smiles in torsiondataset_dict:
                for conf_i in torsiondataset_dict[smiles]:
                    for dih_i in torsiondataset_dict[smiles][conf_i]:
                        ### Torsion
                        N_plots += 1

        N_rows  = int(N_plots/N_col)
        if N_plots%N_col > 0:
            N_rows += 1
        fig, _axs = plt.subplots(
            nrows=N_rows, 
            ncols=N_col,
            ### Width,Height in inches
            figsize=(3.3*N_col,3.*N_rows)
        )
        axs = _axs.reshape(-1)
        
        tobe_deleted = list()
        if N_plots%N_col > 0:
            hangover = N_col - N_plots%N_col
            for h in range(hangover):
                tobe_deleted.append(N_plots+h)
        tobe_deleted = sorted(tobe_deleted)[::-1]
        for i in tobe_deleted:
            fig.delaxes(axs[i])
        _axs = np.delete(axs, tobe_deleted)
        axs = _axs

        N_plots_fig = N_plots
        N_plots = 0

    all_bond_diff_list = list()
    all_angle_diff_list = list()
    all_torsion_diff_list = list()
    all_offeq_diff_list = list()
    all_torsionene_diff_list = list()
    all_vib_diff_list = list()
    for sys_idx in range(N_sys):
        sys = systemlist[sys_idx]
        smiles = sys.name
        if verbose:
            print("Molecule:", smiles)

        masses = list()
        for rdatom in sys.rdmol.GetAtoms():
            masses.append(float(rdatom.GetMass()))
        masses = np.array(masses) * _ATOMIC_MASS

        bond_diffs    = list()
        angle_diffs   = list()
        torsion_diffs = list()
        if not skip_optgeo and smiles in optdataset_dict:
            for conf_i in optdataset_dict[smiles]:
                if not "final_geo" in optdataset_dict[smiles][conf_i]:
                    continue
                xyz    = optdataset_dict[smiles][conf_i]["final_geo"]
                if not unit.is_quantity(xyz):
                    xyz *= _LENGTH_AU
                engine = OpenmmEngine(sys.openmm_system, xyz)
                engine.set_xyz(xyz)
                zm = ZMatrix(sys.rdmol)
                target_zm = zm.build_z_crds(
                    engine.xyz.in_units_of(_LENGTH)
                    )
                success = engine.minimize()
                current_zm = zm.build_z_crds(
                    engine.xyz.in_units_of(_LENGTH)
                    )

                for z_idx, z_value in current_zm.items():
                    if z_idx == 0:
                        continue
                    ### Bonds
                    if z_idx > 0:
                        diff = target_zm[z_idx][0] - z_value[0]
                        bond_diffs.append(diff)
                    ### Angles
                    if z_idx > 1:
                        diff = abs(target_zm[z_idx][1] - z_value[1])
                        angle_diffs.append(diff)
                    ### Torsions
                    if z_idx > 2:
                        diff = abs(target_zm[z_idx][2] - z_value[2])*_ANGLE
                        diff = diff.in_units_of(_ANGLE)
                        if diff > 180.*_ANGLE:
                            diff = diff - 360.*_ANGLE
                            diff = abs(diff)
                        torsion_diffs.append(
                            diff.value_in_unit(_ANGLE))

        if len(bond_diffs):
            if generate_plots:
                axs[N_plots].hist(
                    bond_diffs,
                    histtype="step",
                    density=True,
                )
                axs[N_plots].set_xlabel(r"$\Delta$ Bond length [nm]")
                axs[N_plots].set_ylabel("Density")
                if smiles in title_dict:
                    axs[N_plots].set_title(f"Bond length error {title_dict[smiles]}")
                else:
                    axs[N_plots].set_title(f"Bond length error {smiles}")
                N_plots += 1
            all_bond_diff_list.extend(bond_diffs)

        if len(angle_diffs):
            if generate_plots:
                axs[N_plots].hist(
                    angle_diffs,
                    histtype="step",
                    density=True,
                )
                axs[N_plots].set_xlabel(r"$\Delta$ Bond angle [deg]")
                axs[N_plots].set_ylabel("Density")
                if smiles in title_dict:
                    axs[N_plots].set_title(f"Bond angle error {title_dict[smiles]}")
                else:
                    axs[N_plots].set_title(f"Bond angle error {smiles}")
                N_plots += 1
            all_angle_diff_list.extend(angle_diffs)

        if len(torsion_diffs):
            if generate_plots:
                axs[N_plots].hist(
                    torsion_diffs,
                    histtype="step",
                    density=True,
                )
                axs[N_plots].set_xlabel(r"$\Delta$ Torsion angle [deg]")
                axs[N_plots].set_ylabel("Density")
                if smiles in title_dict:
                    axs[N_plots].set_title(f"Torsion angle error {title_dict[smiles]}")
                else:
                    axs[N_plots].set_title(f"Torsion angle error {smiles}")
                N_plots += 1
            all_torsion_diff_list.extend(torsion_diffs)

        freqs_mm_list = list()
        freqs_qm_list = list()
        if not skip_vib and smiles in optdataset_dict:
            for conf_i in optdataset_dict[smiles]:
                if not "hessian" in optdataset_dict[smiles][conf_i]:
                    continue
                hessian_qm = np.array(optdataset_dict[smiles][conf_i]["hessian"])
                if not unit.is_quantity(hessian_qm):
                    hessian_qm *= _FORCE_AU / _LENGTH_AU
                if not "final_geo" in optdataset_dict[smiles][conf_i]:
                    continue
                xyz = optdataset_dict[smiles][conf_i]["final_geo"]
                if not unit.is_quantity(xyz):
                    xyz *= _LENGTH_AU
                engine = OpenmmEngine(sys.openmm_system, xyz)
                engine.set_xyz(xyz)
                if vib_minimize:
                    engine.minimize()
                hessian_mm = engine.compute_hessian() / unit.constants.AVOGADRO_CONSTANT_NA
                freqs_mm, modes_mm = compute_freqs(hessian_mm, masses)
                freqs_qm, modes_qm = compute_freqs(hessian_qm, masses)
                freqs_mm_list.extend(freqs_mm._value)
                freqs_qm_list.extend(freqs_qm._value)
                
            freqs_qm_list = np.array(freqs_qm_list)
            freqs_qm_list = np.array(freqs_qm_list)
            all_vib_diff_list.extend(
                (freqs_qm_list - freqs_mm_list).tolist())

        if generate_plots and len(freqs_mm_list) > 0:
            axs[N_plots].scatter(
                freqs_mm_list,
                freqs_qm_list,
                color="red"
            )
            axs[N_plots].set_xlabel("This FF [cm-1]")
            axs[N_plots].set_ylabel(r"$\nu$ " + f"({name_reference_level_of_theory}) [cm-1]")
            if smiles in title_dict:
                axs[N_plots].set_title(f"VibFreq {title_dict[smiles]}")
            else:
                axs[N_plots].set_title(f"VibFreq {smiles}")
            if len(freqs_mm_list)>0 and len(freqs_qm_list)>0:
                axs[N_plots].plot(
                    [min(*freqs_mm_list, *freqs_qm_list), max(*freqs_mm_list, *freqs_qm_list)],
                    [min(*freqs_mm_list, *freqs_qm_list), max(*freqs_mm_list, *freqs_qm_list)],
                    linestyle="--",
                    color="black"
                )
            N_plots += 1

        mm_ene_list = list()
        qm_ene_list = list()
        if not skip_offeq and smiles in optdataset_dict:
            for conf_i in optdataset_dict[smiles]:
                if not "ene_list" in optdataset_dict[smiles][conf_i]:
                    continue
                if not "geo_list" in optdataset_dict[smiles][conf_i]:
                    continue
                ene_list_qm = optdataset_dict[smiles][conf_i]["ene_list"]
                geo_list_qm = optdataset_dict[smiles][conf_i]["geo_list"]
                if len(geo_list_qm) == 0:
                    continue
                if len(ene_list_qm) == 0:
                    continue
                xyz         = geo_list_qm[0]
                if not unit.is_quantity(xyz):
                    xyz *= _LENGTH_AU
                engine = OpenmmEngine(sys.openmm_system, xyz)
                for ene, xyz in zip(ene_list_qm, geo_list_qm):
                    if not unit.is_quantity(ene):
                        ene *= _ENERGY_AU * unit.constants.AVOGADRO_CONSTANT_NA
                    if not unit.is_quantity(xyz):
                        xyz *= _LENGTH_AU
                    engine.set_xyz(xyz)
                    mm_ene_list.append(
                        engine.pot_ene.value_in_unit(_ENERGY_PER_MOL))
                    qm_ene_list.append(
                        ene.value_in_unit(_ENERGY_PER_MOL))

        if len(mm_ene_list)>0 and len(qm_ene_list)>0:
            qm_ene_list = np.array(qm_ene_list)
            mm_ene_list = np.array(mm_ene_list)
            min_arg     = np.argmin(qm_ene_list)
            qm_ene_list = qm_ene_list - qm_ene_list[min_arg]
            mm_ene_list = mm_ene_list - mm_ene_list[min_arg]
            if enecutoff5:
                ### Less then 5 kcal/mol
                valids      = np.where(qm_ene_list < 20.92)
                qm_ene_list = qm_ene_list[valids]
                mm_ene_list = mm_ene_list[valids]
            all_offeq_diff_list.extend(
                (qm_ene_list - mm_ene_list).tolist())

        if generate_plots and len(mm_ene_list) > 0:
            axs[N_plots].scatter(
                mm_ene_list,
                qm_ene_list,
                color="blue"
                )
            axs[N_plots].set_xlabel("Energy (this FF) [kJ/mol]")
            axs[N_plots].set_ylabel(f"Energy ({name_reference_level_of_theory}) [kJ/mol]")
            
            if smiles in title_dict:
                axs[N_plots].set_title(f"OffEqEne {title_dict[smiles]}")
            else:
                axs[N_plots].set_title(f"OffEqEne {smiles}")
            if len(mm_ene_list)>0 and len(qm_ene_list)>0:
                axs[N_plots].plot(
                    [min(*mm_ene_list, *qm_ene_list), max(*mm_ene_list, *qm_ene_list)],
                    [min(*mm_ene_list, *qm_ene_list), max(*mm_ene_list, *qm_ene_list)],
                    linestyle="--",
                    color="black"
                )
            N_plots += 1

        if not skip_torsion and smiles in torsiondataset_dict:
            for conf_i in torsiondataset_dict[smiles]:
                for dih_i in torsiondataset_dict[smiles][conf_i]:
                    if not "final_geo" in torsiondataset_dict[smiles][conf_i][dih_i]:
                        continue
                    if not "final_ene" in torsiondataset_dict[smiles][conf_i][dih_i]:
                        continue
                    if not "dih" in torsiondataset_dict[smiles][conf_i][dih_i]:
                        continue
                    xyz_list = torsiondataset_dict[smiles][conf_i][dih_i]["final_geo"]
                    ene_list = torsiondataset_dict[smiles][conf_i][dih_i]["final_ene"]
                    crd_list = torsiondataset_dict[smiles][conf_i][dih_i]["dih"]

                    if len(xyz_list) == 0:
                        continue

                    qm_ene_list = list()
                    mm_ene_list = list()
                    dih_crd_list = list()
                    xyz = xyz_list[0]
                    if not unit.is_quantity(xyz):
                        xyz *= _LENGTH_AU
                    engine = OpenmmEngine(sys.openmm_system, xyz)
                    for xyz, ene, crd in zip(xyz_list, ene_list, crd_list):
                        if not unit.is_quantity(xyz):
                            xyz *= _LENGTH_AU
                        if not unit.is_quantity(ene):
                            ene *= _ENERGY_AU * unit.constants.AVOGADRO_CONSTANT_NA
                        engine.set_xyz(xyz)
                        qm_ene_list.append(ene.value_in_unit(unit.kilojoule_per_mole))
                        mm_ene_list.append(engine.pot_ene.value_in_unit(unit.kilojoule_per_mole))
                        dih_crd_list.append(crd[0])
                    qm_ene_list = np.array(qm_ene_list)
                    mm_ene_list = np.array(mm_ene_list)
                    dih_crd_list = np.array(dih_crd_list)

                    min_ene_idx = np.argmin(qm_ene_list)
                    qm_ene_list -= qm_ene_list[min_ene_idx]
                    mm_ene_list -= mm_ene_list[min_ene_idx]
                    
                    all_torsionene_diff_list.extend(
                        (qm_ene_list - mm_ene_list).tolist())

                    crd_sort_idxs = np.argsort(dih_crd_list)

                    if verbose:
                        print("atom idxs:", torsiondataset_dict[smiles][conf_i][dih_i]["dih_idxs"])

                    if generate_plots:
                        axs[N_plots].scatter(
                            dih_crd_list[crd_sort_idxs],
                            qm_ene_list[crd_sort_idxs],
                            color="black",
                            )
                        axs[N_plots].scatter(
                            dih_crd_list[crd_sort_idxs],
                            mm_ene_list[crd_sort_idxs],
                            color="green",
                            )
                        axs[N_plots].plot(
                            dih_crd_list[crd_sort_idxs],
                            qm_ene_list[crd_sort_idxs],
                            label=f"{name_reference_level_of_theory}",
                            color="black",
                            )
                        axs[N_plots].plot(
                            dih_crd_list[crd_sort_idxs],
                            mm_ene_list[crd_sort_idxs],
                            label=f"This FF",
                            color="green",
                            )
                        axs[N_plots].set_xlabel("Dihedral angle [degree]")
                        axs[N_plots].set_ylabel("Energy [kJ/mol]")

                        if smiles in title_dict:
                            axs[N_plots].set_title(f"TorEne {title_dict[smiles]}")
                        else:
                            axs[N_plots].set_title(f"TorEne {smiles}")
                        axs[N_plots].legend(loc="upper left")
                        N_plots += 1
                
                        
    if verbose:
        if all_bond_diff_list:
            print(
                f"Bond length err {np.abs(np.mean(all_bond_diff_list)):6.3f} +/- {np.std(all_bond_diff_list):6.3f} nm"
                )
        if all_angle_diff_list:
            print(
                f"Bond angle err {np.mean(all_angle_diff_list):6.3f} +/- {np.std(all_angle_diff_list):6.3f} deg"
                )
        if all_torsion_diff_list:
            print(
                f"Torsion err {np.mean(all_torsion_diff_list):6.3f} +/- {np.std(all_torsion_diff_list):6.3f} deg"
                )

    if generate_plots:
        diff_N_plots = N_plots_fig - N_plots
        for i in range(1,diff_N_plots+1):
            fig.delaxes(axs[-i])
        
    results_dict = {
        "bond_diff"       : np.array(all_bond_diff_list),
        "angle_diff"      : np.array(all_angle_diff_list),
        "torsion_diff"    : np.array(all_torsion_diff_list),
        "offeq_diff"      : np.array(all_offeq_diff_list),
        "vib_diff"        : np.array(all_vib_diff_list),
        "torsionene_diff" : np.array(all_torsionene_diff_list)}
    
    if generate_plots:
        return fig, axs, results_dict
    else:
        return results_dict

@ray.remote
def generate_parameter_manager(sys_list, parm_mngr):

    import copy
    parm_mngr_cp = copy.deepcopy(parm_mngr)
    for sys in sys_list:
        parm_mngr_cp.add_system(sys)
    return parm_mngr_cp

def _remove_types(
    parameter_manager, 
    system_list = None, 
    remaining_type_idx = -1,
    set_inactive = False,
    ):

    _CHUNK_SIZE = 20

    from .vectors import ForceFieldParameterVector
    from .constants import _INACTIVE_GROUP_IDX

    if system_list == None:
        system_list = parameter_manager.system_list
    else:
        worker_id_list = list()
        s_list   = tuple()
        idx_list = tuple()
        for sys_idx in range(len(system_list)):
            s_list += (system_list[sys_idx],)
            idx_list += (sys_idx,)
            if len(s_list) == _CHUNK_SIZE:
                worker_id = generate_parameter_manager.remote(
                    s_list, parameter_manager)
                worker_id_list.append([worker_id, idx_list])
                s_list   = tuple()
                idx_list = tuple()
        if len(s_list) > 0:
            worker_id = generate_parameter_manager.remote(
                    s_list, parameter_manager)
            worker_id_list.append([worker_id, idx_list])
        sys_counts = 0
        for worker_id, idx_list in worker_id_list:
            _parm_mngr = ray.get(worker_id)
            parameter_manager.add_parameter_manager(_parm_mngr)
            for sys_idx in idx_list:
                system_list[sys_idx] = parameter_manager.system_list[sys_counts]
                sys_counts += 1

    pvec = ForceFieldParameterVector(
        parameter_manager,
        exclude_others=True)
    if pvec.force_group_count > 0:
        if remaining_type_idx == -1:
            remaining_type_idx = pvec.force_group_count - 1
        pvec.allocations[:] = remaining_type_idx
        pvec.apply_changes()
        remove_list = list()
        for type_i in range(pvec.force_group_count):
            if type_i != remaining_type_idx:
                remove_list.append(type_i)
        for type_i in sorted(remove_list, reverse=True):
            pvec.remove(type_i)
    if set_inactive:
        pvec.allocations[:] = _INACTIVE_GROUP_IDX
    pvec.apply_changes()

    return 1

@ray.remote
def __remove_torsion(openmm_system, improper_dihedral_list):

    import openmm

    openmm_system = openmm.XmlSerializer.deserialize(
            openmm_system)
    force_list = list()
    for f in openmm_system.getForces():
        force_list.append(
                openmm.XmlSerializer.serialize(f))
    for _ in range(openmm_system.getNumForces()):
        openmm_system.removeForce(0)
    for f in force_list:
        _f = openmm.XmlSerializer.deserialize(f)
        if isinstance(_f, openmm.PeriodicTorsionForce):
            tforce = openmm.PeriodicTorsionForce()
            for idx in range(_f.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = _f.getTorsionParameters(idx)
                is_improper = False
                for atm_idxs in improper_dihedral_list:
                    if p1 == atm_idxs[0]:
                        if set([p2,p3,p4]) == set(atm_idxs[1:]):
                            is_improper = True
                            break
                if is_improper:
                    tforce.addTorsion(p1,p2,p3,p4,periodicity,phase,k)
            openmm_system.addForce(tforce)
        else:
            openmm_system.addForce(_f)
    openmm_system = openmm.XmlSerializer.serialize(openmm_system)
    return openmm_system


def _remove_types_torsion(
        system_list):

    import openmm

    worker_id_dict = dict()
    for sys_idx in range(len(system_list)):
        sys = system_list[sys_idx]
        if isinstance(sys.openmm_system, str):
            openmm_system = sys.openmm_system
        else:
            openmm_system = openmm.XmlSerializer.serialize(
                    sys.openmm_system)
        worker_id = __remove_torsion.remote(openmm_system, sys.improper_dihedrals)
        worker_id_dict[worker_id] = sys_idx
    worker_id_list = list(worker_id_dict.keys())
    while worker_id_list:
        [worker_id], worker_id_list = ray.wait(worker_id_list)
        openmm_system = ray.get(worker_id)
        sys_idx = worker_id_dict[worker_id]
        system_list[sys_idx].openmm_system = openmm_system


def remove_types(
    systemmanager,
    mngr_list = list(),
    ):
    
    from .parameters import (
        BondManager,
        AngleManager,
        ProperTorsionManager,
        DoubleProperTorsionManager,
        MultiProperTorsionManager
    )
    from .constants import _INACTIVE_GROUP_IDX
    from . import system

    if len(mngr_list) == 0:
        mngr_list = [
            BondManager,
            AngleManager,
            MultiProperTorsionManager,
        ]

    if isinstance(systemmanager, list):
        system_list = systemmanager
    elif isinstance(systemmanager, system.SystemManager):
        system_list = systemmanager._system_list
    else:
        raise ValueError(
            f"Datatype for SystemManager not understood."
            )

    torsion_mngr_list= (
        ProperTorsionManager, 
        DoubleProperTorsionManager, 
        MultiProperTorsionManager
        )

    for mngr in mngr_list:
        if isinstance(mngr, torsion_mngr_list):
            _remove_types_torsion(system_list)
            _mngr = copy.deepcopy(mngr)
            for _ in range(_mngr.N_systems):
                _mgnr.remove_system(0)
            _remove_types(
                    parameter_manager=_mngr,
                    system_list = system_list,
                    remaining_type_idx = _INACTIVE_GROUP_IDX,
                    set_inactive = True)

        elif isinstance(mngr, BondManager):
            _remove_types(
                parameter_manager=BondManager(),
                system_list = system_list,
                set_inactive = False,
            )
        elif isinstance(mngr, AngleManager):
            _remove_types(
                parameter_manager=AngleManager(),
                system_list = system_list,
                set_inactive = False,
            )
        else:
            raise ValueError(
                "Parameter Manager not understood."
                )


def train_test_set(dataset_dict, tordataset_dict, N_sets=5, f_training=0.5):

    """
    Generate training and testsets. The input datasets must be generated using the
    `combine_datasets` method.
    """

    import numpy as np

    test_set_dict = {i:dict() for i in range(N_sets)}
    training_set_dict = {i:dict() for i in range(N_sets)}
    test_set_torsion_dict = {i:dict() for i in range(N_sets)}
    training_set_torsion_dict = {i:dict() for i in range(N_sets)}
    for i in range(N_sets):
        key_list = list(dataset_dict.keys())
        N_keys   = len(key_list)
        key_idx_list = np.arange(N_keys, dtype=int)
        np.random.shuffle(key_idx_list)
        N_max = int(f_training * N_keys)
        for idx in key_idx_list[:N_max]:
            key1 = key_list[idx]
            training_set_dict[i][key1] = dataset_dict[key1]
            if key1 in tordataset_dict:
                training_set_torsion_dict[i][key1] = tordataset_dict[key1]
        for idx in key_idx_list[N_max:]:
            key1 = key_list[idx]
            test_set_dict[i][key1] = dataset_dict[key1]
            if key1 in tordataset_dict:
                test_set_torsion_dict[i][key1] = tordataset_dict[key1]

    return training_set_dict, training_set_torsion_dict, test_set_dict, test_set_torsion_dict


def combine_datasets(dataset_path, hessian_path, torsion_path, valid_elements):

    """
    Combine different datasets that can be found in pickled files at globbed paths
    given as input arguments. These datasets must be generated by one of the `retrieve_XXX`
    methods in this module.
    """

    import pickle
    import numpy as np
    import glob

    dataset_dict = dict()
    hesdataset_dict = dict()
    tordataset_dict = dict()

    smiles_map = dict()

    ### DATASET
    ### =======
    for p in glob.glob(dataset_path):
        with open(p, "rb") as fopen:
            _dataset_dict = pickle.loads(
                fopen.read())
        for key in _dataset_dict:
            for conf in _dataset_dict[key]:
                smi = _dataset_dict[key][conf]['qcentry']['attributes']['canonical_isomeric_smiles']
                smiles_map[conf] = smi
            if key in dataset_dict:
                dataset_dict[key].update(
                    _dataset_dict[key])
            else:
                dataset_dict[key] = _dataset_dict[key]

    ### HESSIAN
    ### =======
    for p in glob.glob(hessian_path):
        with open(p, "rb") as fopen:
            _hesdataset_dict = pickle.loads(
                fopen.read())
        for key in _hesdataset_dict:
            if key in hesdataset_dict:
                hesdataset_dict[key].update(
                    _hesdataset_dict[key])
            else:
                hesdataset_dict[key] =  _hesdataset_dict[key]
    for key1 in dataset_dict:
        for key2 in dataset_dict[key1]:
            N_atoms = len(dataset_dict[key1][key2]['final_geo'])
            _key2 = key2.upper()
            if key2 in hesdataset_dict:
                dataset_dict[key1][key2]["hessian"] = np.array(
                    hesdataset_dict[key2]["result"]).reshape(3*N_atoms, 3*N_atoms).tolist()
            elif _key2 in hesdataset_dict:
                dataset_dict[key1][key2]["hessian"] = np.array(
                    hesdataset_dict[_key2]["result"]).reshape(3*N_atoms, 3*N_atoms).tolist()

    ### TORSION
    ### =======
    for p in glob.glob(torsion_path):
        with open(p, "rb") as fopen:
            _tordataset_dict = pickle.loads(
                fopen.read())
            for key in _tordataset_dict:
                for conf in _tordataset_dict[key]:
                    smi = _tordataset_dict[key][conf][0]['qcentry']['attributes']['canonical_isomeric_smiles']
                    smiles_map[conf] = smi
                if key in tordataset_dict:
                    tordataset_dict[key].update(
                        _tordataset_dict[key])
                else:
                    tordataset_dict[key] = _tordataset_dict[key]

    ### Restructure the data
    ### ====================
    _dataset_dict = dict()
    for key in dataset_dict:
        for conf in dataset_dict[key]:
            smi = smiles_map[conf]
            if smi in _dataset_dict:
                _dataset_dict[smi][conf] = dataset_dict[key][conf]
            else:
                _dataset_dict[smi] = {conf : dataset_dict[key][conf]}

    _tordataset_dict = dict()
    for key in tordataset_dict:
        for conf in tordataset_dict[key]:
            smi = smiles_map[conf]
            if smi in _tordataset_dict:
                _tordataset_dict[smi][conf] = tordataset_dict[key][conf]
            else:
                _tordataset_dict[smi] = {conf : tordataset_dict[key][conf]}

    dataset_dict    = _dataset_dict
    tordataset_dict = _tordataset_dict

    ### Kick out entries with unwanted elements
    ### =======================================
    if valid_elements:
        to_delete = list()
        for key1 in dataset_dict:
            key2 = list(dataset_dict[key1].keys())[0]
            symbols = dataset_dict[key1][key2]['qcentry']['initial_molecule']['symbols']
            for ele in symbols:
                if not ele in valid_elements:
                    to_delete.append(key1)
                    break
        for key1 in to_delete:
            del dataset_dict[key1]
            if key1 in tordataset_dict:
                del tordataset_dict[key1]

    return dataset_dict, tordataset_dict


@ray.remote
def parameterize_system(_qcentry, _geometry, _smiles, _forcefield_name, remove_types_manager_list, _partial_charges):
    from . import system
    ### Filter warnings here, because openmm-interchange will generate
    ### lots of bondhandler upconversion warnings.
    import warnings
    import json
    import qcelemental as qce
    warnings.filterwarnings('ignore')
    if isinstance(_qcentry, str):
        value_dict = json.loads(_qcentry)
        _qcentry   = qce.models.molecule.Molecule.from_data(value_dict)
    _qcentry = fix_qcentry(_qcentry, _geometry)
    try:
        system_list = [system.from_qcschema(
            _qcentry, _smiles, _forcefield_name, _partial_charges)]
        if remove_types_manager_list:
            remove_types(
                system_list,
                remove_types_manager_list)
        return system_list[0]
    except Exception as e:
        print(e)
        print(
            f"Could not build system {_smiles}")
        return None

def fix_qcentry(qcentry, geometry):

    """
    Fix some issues with qcschemas.
    """

    from openmm import unit
    import qcelemental as qce

    if isinstance(qcentry, qce.models.molecule.Molecule):
        return qcentry

    ### We don't need fixing, if these two are found
    if 'symbols' in qcentry and 'geometry' in qcentry:
        return qcentry
    
    if 'molecule' not in qcentry:
        qcentry['molecule'] = dict()
    if 'geometry' not in qcentry['molecule']:
        if unit.is_quantity(geometry):
            geometry = geometry.value_in_unit(unit.bohr)
        qcentry['molecule']['geometry'] = geometry
    if 'attributes' not in qcentry['molecule']:
        qcentry['molecule']['attributes'] = qcentry['attributes']

    return qcentry


@ray.remote
def generate_rdmol_dict(optdataset_dict):

    import warnings
    from openff.toolkit.topology import Molecule
    import qcelemental as qce
    import json
    from rdkit import Chem

    rdmol_dict = dict()
    for smiles in optdataset_dict:
        found_valid_qcentry = False
        if "qcentry" in optdataset_dict[smiles]:
            if isinstance(optdataset_dict[smiles]["qcentry"], str):
                value_dict = json.loads(
                    optdataset_dict[smiles]["qcentry"])
                qcentry    = qce.models.molecule.Molecule.from_data(value_dict)
            else:
                qcentry = optdataset_dict[smiles]["qcentry"]
            qcentry = fix_qcentry(
                    qcentry,
                    optdataset_dict[smiles]['final_geo'])
            try:
                offmol = Molecule.from_qcschema(
                    qcentry, allow_undefined_stereo=True)
                found_valid_qcentry = True
            except:
                continue
        else:
            for key in optdataset_dict[smiles]:
                if "qcentry" in optdataset_dict[smiles][key]:
                    if isinstance(optdataset_dict[smiles][key]["qcentry"], str):
                        value_dict = json.loads(
                            optdataset_dict[smiles][key]["qcentry"])
                        qcentry    = qce.models.molecule.Molecule.from_data(value_dict)
                    else:
                        qcentry = optdataset_dict[smiles][key]["qcentry"]
                    qcentry = fix_qcentry(
                            qcentry,
                            optdataset_dict[smiles][key]["final_geo"])
                    try:
                        offmol = Molecule.from_qcschema(
                            qcentry, allow_undefined_stereo=True)
                        found_valid_qcentry = True
                        break
                    except:
                        continue
        if found_valid_qcentry:
            rdmol  = offmol.to_rdkit()
            smi    = Chem.MolToSmiles(
                    rdmol,
                    isomericSmiles=False)
            rdmol_dict[smiles] = rdmol, smi
        else:
            warnings.warn(
                f"Could not load {smiles}")

    return rdmol_dict


class SystemManagerLoader(object):

    def __init__(
        self,
        query_smiles_list=None,
        error_scale_geo = 1.0,
        error_scale_vib = 1.0,
        error_scale_offeq = 1.0,
        error_scale_torsion = 1.0,
        error_scale_force = 1.0,
        optdataset_dict = None, 
        torsiondataset_dict = None,
        forcefield_name =_DEFAULT_FF,
        ene_weighting = True,
        use_geo = True,
        use_vib = True,
        use_offeq = True,
        use_torsion = True,
        use_force = False,
        add_units = False,
        force_projection = False,
        reference_to_lowest = True,
        remove_types_manager_list = list(),
        vib_minimize = True,
        verbose = False,
        ):

        import h5py
        self._ish5_torsion = False
        self._ish5_opt     = False
        if isinstance(optdataset_dict, h5py.File):
            self._ish5_opt = True
        if isinstance(torsiondataset_dict, h5py.File):
            self._ish5_torsion = True

        if isinstance(query_smiles_list, type(None)):
            self._query_smiles_list  = list(optdataset_dict.keys())
        else:
            self._query_smiles_list = query_smiles_list
        if self._ish5_opt:
            ### This replacement is necessary since
            ### any `/` in the h5py group keys is replaced with a '__'
            self._query_smiles_list = [key.replace("__", "/") for key in self._query_smiles_list]
        self._query_smiles_list  = list(set(self._query_smiles_list))
        self.error_scale_geo     = error_scale_geo
        self.error_scale_vib     = error_scale_vib
        self.error_scale_offeq   = error_scale_offeq
        self.error_scale_torsion = error_scale_torsion
        self.error_scale_force   = error_scale_force
        self.optdataset_dict     = optdataset_dict
        self.torsiondataset_dict = torsiondataset_dict
        self.forcefield_name     = forcefield_name
        self.ene_weighting       = ene_weighting
        self.use_geo             = use_geo
        self.use_vib             = use_vib
        self.use_offeq           = use_offeq
        self.use_torsion         = use_torsion
        self.use_force           = use_force
        self.add_units           = add_units
        self.force_projection    = force_projection
        self.reference_to_lowest = reference_to_lowest
        self.vib_minimize        = vib_minimize
        self.verbose             = verbose

        self.rdmol_dict = dict()
        self.rdmol_to_smiles_map_dict = dict()
        self.smiles_list = list()

        import copy
        self._remove_types_manager_list = copy.deepcopy(remove_types_manager_list)

        self.system_cache_dict   = dict()

        self._generate_rdmol_dict()


    def add_parameter_manager(self, parameter_manager):

        import copy
        self._remove_types_manager_list.append(
            copy.deepcopy(parameter_manager))
        self.clear_cache()

    def clear_cache(self):

        self.system_cache_dict.clear()

    def _check_smiles_list(self, query_smiles_list=None):

        if isinstance(query_smiles_list, type(None)):
            return self._query_smiles_list

        import warnings

        _smiles_list = list()
        for smi in query_smiles_list:
            if isinstance(smi, str):
                if smi not in self.smiles_list:
                    warnings.warn(
                        f"{smi} not in internal smiles list (length {len(self.smiles_list)})")
                    continue
                _smiles_list.extend(
                    self.rdmol_to_smiles_map_dict[smi])
            elif isinstance(smi, int):
                if smi > (len(self.smiles_list)-1):
                    warnings.warn(
                        f"{smi} not in internal smiles list (length {len(self.smiles_list)}")
                    continue
                _smiles_list.extend(
                    self.rdmol_to_smiles_map_dict[
                        self.smiles_list[smi]])
            else:
                warnings.warn(
                    f"{smi} not understood")

        return _smiles_list


    def _generate_rdmol_dict(self, exclude_small=True):

        import ray
        from openff.toolkit.topology import Molecule
        from rdkit import Chem
        if exclude_small:
            from rdkit.Chem import rdMolDescriptors

        CHUNK_SIZE = 1000
        optdataset_dict = dict()
        worker_id_list = list()
        for smi in self._query_smiles_list:
            if len(optdataset_dict) < CHUNK_SIZE:
                if smi not in self.optdataset_dict:
                    continue
                for key in self.optdataset_dict[smi]:
                    check  = "qcentry" in self.optdataset_dict[smi][key]
                    check *= "final_geo" in self.optdataset_dict[smi][key]
                    if check:
                        if self._ish5_opt:
                            qcentry   = self.optdataset_dict[smi][key]["qcentry"].asstr()[()]
                            geometry  = self.optdataset_dict[smi][key]["final_geo"][:]
                        else:
                            qcentry   = self.optdataset_dict[smi][key]["qcentry"]
                            geometry  = self.optdataset_dict[smi][key]["final_geo"]
                        optdataset_dict[smi] = {
                            "qcentry"   : qcentry,
                            "final_geo" : geometry,
                            }
                        break
            else:
                worker_id = generate_rdmol_dict.remote(
                        optdataset_dict)
                worker_id_list.append(worker_id)
                optdataset_dict.clear()
        if len(optdataset_dict) > 0:
            worker_id = generate_rdmol_dict.remote(
                    optdataset_dict)
            worker_id_list.append(worker_id)
            optdataset_dict.clear()

        self.rdmol_dict = dict()
        self.rdmol_to_smiles_map_dict = dict()
        self.smiles_list = list()
        while worker_id_list:
            [worker_id], worker_id_list = ray.wait(worker_id_list)
            _rdmol_dict = ray.get(worker_id)
            for smiles in _rdmol_dict:
                ### smiles : initial smiles
                ### smi    : canonical and not isomeric smiles
                rdmol, smi = _rdmol_dict[smiles]
                if exclude_small:
                    heavy_atom_count = rdMolDescriptors.CalcNumHeavyAtoms(rdmol)
                    if heavy_atom_count < 2:
                        continue
                self.rdmol_dict[smi] = rdmol
                if smi in self.rdmol_to_smiles_map_dict:
                    self.rdmol_to_smiles_map_dict[smi].append(smiles)
                else:
                    self.rdmol_to_smiles_map_dict[smi] = [smiles]
        self.smiles_list = list(self.rdmol_dict.keys())


    def generate_systemmanager(self, smiles_list=None):

        import copy
        import warnings

        _smiles_list = list()
        for smi in smiles_list:
            if not smi in self.system_cache_dict:
                _smiles_list.append(smi)
        _smiles_query_list = self._check_smiles_list(_smiles_list)

        system_manager, _ = generate_systemmanager(
            _smiles_query_list,
            None,
            self.error_scale_geo,
            self.error_scale_vib,
            self.error_scale_offeq,
            self.error_scale_torsion,
            self.error_scale_force,
            self.optdataset_dict,
            self.torsiondataset_dict,
            self.forcefield_name,
            self.ene_weighting,
            self.use_geo,
            self.use_vib,
            self.use_offeq,
            self.use_torsion,
            self.use_force,
            self.add_units,
            self.force_projection,
            self.reference_to_lowest,
            self._remove_types_manager_list,
            self.vib_minimize,
            verbose=False)

        for smi in smiles_list:
            if smi in system_manager._rdmol_list:
                if self.verbose:
                    print(
                        f"Adding {smi}")
                sys_idx = system_manager._rdmol_list.index(smi)
                sys     = system_manager._system_list[sys_idx]
                self.system_cache_dict[smi] = copy.deepcopy(sys)
            elif smi in self.system_cache_dict:
                if self.verbose:
                    print(
                        f"Adding cached {smi}")
                system_manager.add_system(
                    copy.deepcopy(self.system_cache_dict[smi]))
            else:
                warnings.warn(
                        f"Could not add {smi}")

        return system_manager


def generate_systemmanager(
    smiles_list,
    systemmanager=None, 
    error_scale_geo=1.0,
    error_scale_vib=1.0,
    error_scale_offeq=1.0,
    error_scale_torsion=1.0,
    error_scale_force=1.0,
    optdataset_dict=None, 
    torsiondataset_dict=None,
    forcefield_name=_DEFAULT_FF,
    ene_weighting = True,
    use_geo = True,
    use_vib = True,
    use_offeq = True,
    use_torsion = True,
    use_force = False,
    add_units=False,
    force_projection=False,
    reference_to_lowest=True,
    remove_types_manager_list=list(),
    vib_minimize=True,
    verbose=False):

    __doc__ = """
    This is a helper method for loading a dataset into a systemmanager
    and create targets. It also applies initial parameters to the systems
    under consideration.

    We can also append system to an already existing systemmangaer.

    Example of a smiles list. This list must match to keys in the optdataset and
    torsiondataset dictionaries.

    smiles_list = [
                "CCCC", # n-Butane
                "C1CCC1", # Cyclobutane
                "CC(C)C", # 2-Methylpropane
                "CC(C)=C", # 2-Methylpropene
                "CC1CC1", # Methylcyclopropane
                "CC1=CC1", # 1-Methylcyclopropene
                "C1=CC1C", # 3-Methylcyclopropene
                "C1CC=C1", # Cyclobutene
                "C1=CC=C1", # Cyclobutadiene
                "CCC=C", # 1-Butene
                "C=CC=C", # 1,3-Butadiene
                "C/C=C\C", # cis 2-Butene
                "C/C=C/C", # trans 2-Butene
                "C(#CC)C", # 2-Butyne
                "C#CC#C", # Butadiyne (Diacetylene)
        ]

    """

    import h5py

    _ish5_torsion = False
    _ish5_opt     = False
    if isinstance(optdataset_dict, h5py.File):
        _ish5_opt = True
    if isinstance(torsiondataset_dict, h5py.File):
        _ish5_torsion = True

    if use_torsion and isinstance(torsiondataset_dict, type(None)):
        import warnings
        warnings.warn("Did not provide torsion dataset. Setting `use_torsion=False`")
        use_torsion = False

    import ray
    from . import system
    from .targets import (
        NormalModeTarget, 
        GeoTarget, 
        ForceMatchingTarget, 
        ForceProjectionMatchingTarget, 
        EnergyTarget
    )
    from .constants import (
        _FORCE_AU, 
        _LENGTH_AU, 
        _ENERGY_AU, 
        _LENGTH, 
        _ANGLE, 
        _FORCE, 
        _ENERGY_PER_MOL, 
        _WAVENUMBER,
        _TIMEOUT
        )
    from rdkit import Chem

    if add_units:
        from openmm import unit
        def TO_FORCE_UNIT(x):
            if isinstance(x, unit.Quantity):
                return x.in_units_of(_FORCE_AU)
            else:
                return x * _FORCE_AU

        def TO_ENE_UNIT(x):
            if isinstance(x, unit.Quantity):
                return x.in_units_of(_ENERGY_AU)
            else:
                return x * _ENERGY_AU

        def TO_LENGTH_UNIT(x):
            if isinstance(x, unit.Quantity):
                return x.in_units_of(_LENGTH_AU)
            else:
                return x * _LENGTH_AU

        def TO_HESSIAN_UNIT(x):
            if isinstance(x, unit.Quantity):
                return x.in_units_of(_FORCE_AU / _LENGTH_AU)
            else:
                return x * _FORCE_AU / _LENGTH_AU

    else:
        TO_FORCE_UNIT   = lambda x: x
        TO_ENE_UNIT     = lambda x: x
        TO_LENGTH_UNIT  = lambda x: x
        TO_HESSIAN_UNIT = lambda x: x

    worker_id_dict = dict()
    remove_types_manager_list_id = ray.put(remove_types_manager_list)
    for smiles in smiles_list:
        if use_geo or use_offeq or use_force or use_vib:
            check = False
            if _ish5_opt:
                _smiles = smiles.replace("/", "__")
            else:
                _smiles = smiles
            for key0 in optdataset_dict[_smiles]:
                check  = "qcentry" in optdataset_dict[_smiles][key0]
                check *= "final_geo" in optdataset_dict[_smiles][key0]
                if check:
                    break
            if check:
                partial_charges = None
                if 'partial_charges' in optdataset_dict[_smiles][key0]:
                    if _ish5_opt:
                        partial_charges = optdataset_dict[_smiles][key0]['partial_charges'][:]
                    else:
                        partial_charges = optdataset_dict[_smiles][key0]['partial_charges']
                if _ish5_opt:
                    qcentry  = optdataset_dict[_smiles][key0]["qcentry"].asstr()[()]
                    geometry = optdataset_dict[_smiles][key0]["final_geo"][:]
                else:
                    qcentry  = optdataset_dict[_smiles][key0]["qcentry"]
                    geometry = optdataset_dict[_smiles][key0]["final_geo"]
                worker_id = parameterize_system.remote(
                        qcentry, geometry, smiles, 
                        forcefield_name, remove_types_manager_list_id, partial_charges)
                worker_id_dict[worker_id] = smiles
            else:
                continue
        elif use_torsion:
            check = False
            if _ish5_opt:
                _smiles = smiles.replace("/", "__")
            else:
                _smiles = smiles
            for key0 in torsiondataset_dict[_smiles]:
                check = False
                for key1 in torsiondataset_dict[_smiles][key0]:
                    check  = "qcentry" in torsiondataset_dict[_smiles][key0][key1]
                    check *= "final_geo" in torsiondataset_dict[_smiles][key0][key1]
                    if check:
                        break
                if check:
                    break
            if check:
                partial_charges = None
                if 'partial_charges' in optdataset_dict[_smiles][key0]:
                    if _ish5_opt:
                        partial_charges = optdataset_dict[_smiles][key0]['partial_charges'][:]
                    else:
                        partial_charges = optdataset_dict[_smiles][key0]['partial_charges']
                if _ish5_torsion:
                    qcentry  = torsiondataset_dict[_smiles][key0][key1]["qcentry"].asstr()[()]
                    geometry = torsiondataset_dict[_smiles][key0][key1]["final_geo"][:]
                else:
                    qcentry  = torsiondataset_dict[_smiles][key0][key1]["qcentry"]
                    geometry = torsiondataset_dict[_smiles][key0][key1]["final_geo"]
                worker_id = parameterize_system.remote(
                    qcentry, geometry, smiles, 
                    forcefield_name, remove_types_manager_list_id, partial_charges)
                worker_id_dict[worker_id] = smiles
            else:
                continue
        else:
            continue

    if systemmanager == None:
        systemmanager = system.SystemManager()

    N_data_points = 0.
    worker_id_list = list(worker_id_dict.keys())
    while worker_id_list:

        [worker_id], worker_id_list = ray.wait(worker_id_list)
        smiles = worker_id_dict[worker_id]

        if verbose:
            print("Adding", smiles)

        try:
            sys = ray.get(worker_id)
        except Exception as e:
            print(e)
            continue
        if isinstance(sys, type(None)):
            continue

        smi = Chem.MolToSmiles(
            sys.offmol.to_rdkit(),
            isomericSmiles=False,
            )
        sys_target = sys
        if smi in systemmanager._rdmol_list:
            index = systemmanager._rdmol_list.index(smi)
            sys   = systemmanager._system_list[index]
        else:
            sys._name = smi
            systemmanager.add_system(sys)

        final_geo_list = list()
        off_geo_list   = list()
        hessian_list   = list()
        force_list     = list()
        ene_list       = list()

        if use_geo or use_offeq or use_force or use_vib:
            if _ish5_opt:
                _smiles = smiles.replace("/", "__")
            else:
                _smiles = smiles

            for conf_i in optdataset_dict[_smiles]:

                if use_geo or use_vib:
                    if "final_geo" in optdataset_dict[_smiles][conf_i]:
                        final_geo_list.extend(
                            [optdataset_dict[_smiles][conf_i]["final_geo"][:]])

                if use_offeq or use_force:
                    if "geo_list" in optdataset_dict[_smiles][conf_i]:
                        if "ene_list" in optdataset_dict[_smiles][conf_i]:
                            if use_force:
                                force_list.extend(
                                    optdataset_dict[_smiles][conf_i]["force_list"][:])
                            off_geo_list.extend(
                                optdataset_dict[_smiles][conf_i]["geo_list"][:])
                            ene_list.extend(
                                optdataset_dict[_smiles][conf_i]["ene_list"][:])

                if use_vib:
                    if "hessian" in optdataset_dict[_smiles][conf_i]:
                        if "final_geo" in optdataset_dict[_smiles][conf_i]:
                            hessian_list.extend(
                                [optdataset_dict[_smiles][conf_i]["hessian"]][:])

                ### This adds graph information computed from QTAIM data to the
                ### systems. Currently we don't want to support this.
                #qgraph = get_qgraph(qtaimdataset_dict[smiles][conf_i]['cp_auto'])
                #add_basin_features(qgraph, qtaimdataset_dict[smiles][conf_i]['dens_yt'])
                #sys.add_nxmol(qgraph)

        if use_torsion:
            if _ish5_torsion:
                _smiles = smiles.replace("/", "__")
            else:
                _smiles = smiles
            if _smiles in torsiondataset_dict:
                weight = len(torsiondataset_dict[_smiles])
                for conf_i in torsiondataset_dict[_smiles]:
                    torsion_geo_list = list()
                    torsion_ene_list = list()
                    for dih_i in torsiondataset_dict[_smiles][conf_i]:
                        if not isinstance(torsiondataset_dict[_smiles][conf_i][dih_i], dict):
                            continue
                        if "final_geo" in torsiondataset_dict[_smiles][conf_i][dih_i]:
                            if "final_ene" in torsiondataset_dict[_smiles][conf_i][dih_i]:
                                torsion_geo_list.extend(
                                    torsiondataset_dict[_smiles][conf_i][dih_i]["final_geo"][:])
                                torsion_ene_list.extend(
                                    torsiondataset_dict[_smiles][conf_i][dih_i]["final_ene"][:])

                    target_dict_torsion = { "structures"          : [TO_LENGTH_UNIT(_g) for _g in torsion_geo_list],
                                            "energies"            : [TO_ENE_UNIT(_e) for _e in torsion_ene_list],
                                            "rdmol"               : sys_target.rdmol,
                                            "minimize"            : False,
                                            "denom_ene"           : 2.5 * _ENERGY_PER_MOL * error_scale_torsion, 
                                            "ene_weighting"       : ene_weighting,
                                            "reference_to_lowest" : reference_to_lowest,
                                        }
                    if torsion_geo_list:
                        if target_dict_torsion["structures"]:
                            sys.add_target(EnergyTarget, target_dict_torsion)
                            N_data_points += len(torsion_geo_list)

        if use_geo:
            if len(final_geo_list) > 0:
                target_dict_geo = { "structures"    : [TO_LENGTH_UNIT(_g) for _g in final_geo_list],
                                    "rdmol"         : sys_target.rdmol,
                                    "minimize"      : True,
                                    "H_constraint"  : False,
                                    "denom_bond"    : 5.0e-3 * _LENGTH * error_scale_geo,
                                    "denom_angle"   : 8.0e-0 * _ANGLE * error_scale_geo,
                                    "denom_torsion" : 2.0e+1 * _ANGLE * error_scale_geo,
                                  }
                if target_dict_geo["structures"]:
                    sys.add_target(GeoTarget, target_dict_geo)
                    N_data_points += len(final_geo_list) * (sys.N_atoms * 3. - 6.)

        if use_vib:
            if len(final_geo_list) > 0 and len(hessian_list) > 0:
                target_dict_nm = { "structures"   : [TO_LENGTH_UNIT(_g) for _g in final_geo_list],
                                   "rdmol"        : sys_target.rdmol,
                                   "minimize"     : vib_minimize, 
                                   "H_constraint" : False,
                                   "hessian"      : [TO_HESSIAN_UNIT(_h) for _h in hessian_list],
                                   "denom_freq"   : 200. * _WAVENUMBER * error_scale_vib, 
                                }
                if target_dict_nm["structures"]:
                    sys.add_target(NormalModeTarget, target_dict_nm)
                    N_data_points += len(final_geo_list)

        if use_offeq:
            if len(off_geo_list) > 0:
                target_dict_ene = { "structures"          : [TO_LENGTH_UNIT(_g) for _g in off_geo_list],
                                    "energies"            : [TO_ENE_UNIT(_e) for _e in ene_list],
                                    "rdmol"               : sys_target.rdmol,
                                    "minimize"            : False,
                                    "ene_weighting"       : ene_weighting,
                                    "reference_to_lowest" : reference_to_lowest, 
                                    "denom_ene"           : 2.5 * _ENERGY_PER_MOL * error_scale_offeq,
                                }
                if target_dict_ene["structures"]:
                    sys.add_target(EnergyTarget, target_dict_ene)
                    N_data_points += len(off_geo_list)

        if use_force:
            if len(off_geo_list) > 0:
                target_dict_frc = { "structures"   : [TO_LENGTH_UNIT(_g) for _g in off_geo_list],
                                    "forces"       : [TO_FORCE_UNIT(_f) for _f in force_list],
                                    "energies"     : [TO_ENE_UNIT(_e) for _e in ene_list],
                                    "rdmol"        : sys_target.rdmol,
                                    "minimize"     : False,
                                    "H_constraint" : False,
                                    "ene_weighting": ene_weighting,
                                    "denom_bond"   : 5.0e-0 * _FORCE * error_scale_force,
                                    "denom_angle"  : 5.0e-1 * _FORCE * error_scale_force,
                                    "denom_force"  : 2.0e+4 * _FORCE * error_scale_force,
                                }
                if target_dict_frc["structures"]:
                    if force_projection:
                        sys.add_target(ForceProjectionMatchingTarget, target_dict_frc)
                    else:
                        sys.add_target(ForceMatchingTarget, target_dict_frc)
                    N_data_points += len(target_dict_frc["structures"])
        
    return systemmanager, N_data_points


@ray.remote(num_cpus=2)
def compute_gaussian_fchk(input_dict):

    from subprocess import call
    import os
    import random
    import shutil

    method    = input_dict["method"]
    basis     = input_dict["basis"]
    num_procs = input_dict["num_procs"]
    xyz_str   = input_dict["xyz_str"]
    extra     = input_dict["extra"]
    name      = input_dict["name"]
    prefix    = str(random.randint(0, 999999))

    ### Local calcs on fireball.ucsd.edu
    scratch_dir_parent = os.environ.get('TMPDIR')
    if scratch_dir_parent == None:
        scratch_dir_parent = "/tmp/gauss_scratch/"
    scratch_dir = f"{scratch_dir_parent}/{prefix}"

    os.environ["GAUSS_SCRDIR"] = scratch_dir

    try:
        if not os.path.exists(scratch_dir):
            os.makedirs(scratch_dir, exist_ok=True)
        os.chdir(scratch_dir)
    except Exception as e:
        print("PYTHON EXCEPTION. Could not create folder.", e)
        return None

    try:
        gau_in = f"""
%RWF={scratch_dir}/{prefix}.rwf
%Int={scratch_dir}/{prefix}.int
%D2E={scratch_dir}/{prefix}.d2e
%skr={scratch_dir}/{prefix}.skr
%chk={prefix}.chk
%nproc={num_procs:d}

# {method}/{basis} {extra}

{name}

{xyz_str}

"""
        with open(f"{prefix}.gau", "w") as fopen:
            fopen.write(gau_in)

        call(["g16", f"{prefix}.gau"])
        call(["formchk", f"{prefix}.chk", f"{prefix}.fchk"])
        with open(f"{prefix}.fchk", "r") as fopen:
            fchk_str = fopen.read()
        shutil.rmtree(scratch_dir, ignore_errors=True)
    except Exception as e:
        print("PYTHON EXCEPTION ", e)
        os.chdir(scratch_dir_parent)
        shutil.rmtree(scratch_dir, ignore_errors=True)
        return None

    return fchk_str


@ray.remote(num_cpus=4)
def compute_critic2_qtaim_props(input_dict):

    from subprocess import call
    import os
    import random
    import shutil
    import gzip
    import numpy as np

    BOHR2ANG  = 0.529177249
    ANG2BOHR  = 1./BOHR2ANG
    PNTS_PER_BOHR = 5
    BOHR_PER_PNTS = 1./PNTS_PER_BOHR

    num_procs    = input_dict["num_procs"]
    fchk_str     = input_dict["fchk_str"]
    extra_fields = input_dict["extra_fields"]
    name         = input_dict["name"]
    prefix       = str(random.randint(0, 999999))

    output_dict = {
        "dens.cube.gz"    : None,
        "elf.cube.gz"     : None,
        "dens_yt.json"    : None,
        "elf_yt.json"     : None,
        "cp_auto.json"    : None,
        "cro"             : None,
        "cri"             : None,
    }

    ### Local calcs on fireball.ucsd.edu
    scratch_dir_parent = os.environ.get('TMPDIR')
    if scratch_dir_parent == None:
        scratch_dir_parent = "/tmp/critic2_scratch/"
    scratch_dir = f"{scratch_dir_parent}/{prefix}"

    os.environ["OMP_NUM_THREADS"] = str(num_procs)

    try:
        if not os.path.exists(scratch_dir):
            os.makedirs(scratch_dir, exist_ok=True)
        os.chdir(scratch_dir)
    except Exception as e:
        print("PYTHON EXCEPTION. Could not create folder.", e)
        return None

    with open(f"./fchk", "w") as fopen:
        fopen.write(fchk_str)
    found_cartesian = False
    crds_list       = list()
    for line in fchk_str.split("\n"):
        line = line.rstrip().lstrip().split()
        if found_cartesian:
            crds_list.extend(line)
            if len(crds_list) == N_crds:
                found_cartesian = False
                break
        else:
            if len(line) == 0:
                continue
            if line[0] == "Current" and \
                line[1] == "cartesian" and \
                line[2] == "coordinates":
                N_crds          = int(line[-1])
                N_atoms         = int(N_crds/3)
                found_cartesian = True

    crds_list = np.array(crds_list, dtype=float)
    crds_list = np.reshape(crds_list, (N_atoms,3))
    minbox    = np.min(crds_list, axis=0) - 10. * ANG2BOHR
    maxbox    = np.max(crds_list, axis=0) + 10. * ANG2BOHR
    npoints   = (maxbox-minbox) * PNTS_PER_BOHR
    minbox   *= BOHR2ANG
    maxbox   *= BOHR2ANG
    if extra_fields:
        extra_comment = ""
        output_dict[f"mep.cube.gz"] = None
        output_dict[f"uslater.cube.gz"] = None
        output_dict[f"mep_yt.json"] = None
    else:
        extra_comment = "#"

    critic2_in = f"""
### Molecule {name}
###
### Note, the default is to add 10 Ang padding in each direction
### of the unit cell for the loaded molecule.
molecule ./fchk 10
load ./fchk id wfn READVIRTUAL

### Generate cube files first
cube {minbox[0]:4.6f} {minbox[1]:4.6f} {minbox[2]:4.6f} \\
     {maxbox[0]:4.6f} {maxbox[1]:4.6f} {maxbox[2]:4.6f} \\
     {BOHR_PER_PNTS:4.6f} \\
     file dens.cube field "$wfn"
cube {minbox[0]:4.6f} {minbox[1]:4.6f} {minbox[2]:4.6f} \\
     {maxbox[0]:4.6f} {maxbox[1]:4.6f} {maxbox[2]:4.6f} \\
     {BOHR_PER_PNTS:4.6f} \\
     file elf.cube field "elf($wfn)"
{extra_comment}cube {minbox[0]:4.6f} {minbox[1]:4.6f} {minbox[2]:4.6f} \\
{extra_comment}     {maxbox[0]:4.6f} {maxbox[1]:4.6f} {maxbox[2]:4.6f} \\
{extra_comment}     {BOHR_PER_PNTS:4.6f} \\
{extra_comment}     file mep.cube field "mep($wfn)"
{extra_comment}cube {minbox[0]:4.6f} {minbox[1]:4.6f} {minbox[2]:4.6f} \\
{extra_comment}     {maxbox[0]:4.6f} {maxbox[1]:4.6f} {maxbox[2]:4.6f} \\
{extra_comment}     {BOHR_PER_PNTS:4.6f} \\
{extra_comment}     file uslater.cube field "uslater($wfn)"

### Then load them back in. We must do this in order to get
### defined sampling of points along each axis.
load dens.cube    id dens
load elf.cube     id elf
{extra_comment}load mep.cube     id mep
{extra_comment}load uslater.cube id uslater
load as "1." sizeof dens   id vol

### =========================== ###
### Normal Bader-style analysis ###
### =========================== ###

### Electron density:
### -----------------

CPREPORT verylong
reference dens
integrable dens name dens
integrable elf name elf
integrable vol name vol
integrable wfn multipoles 2 name multipole
{extra_comment}integrable mep name mep
{extra_comment}integrable uslater name uslater
yt json dens_yt.json

### ELF density:
### ------------
CPREPORT verylong
reference elf
integrable dens name dens
integrable elf name elf
integrable vol name vol
integrable wfn multipoles 2 name multipole
{extra_comment}integrable mep name mep
{extra_comment}integrable uslater name uslater
yt nnm discard "$elf < 1e-1" json elf_yt.json


### MEP density:
### ------------
{extra_comment}CPREPORT verylong
{extra_comment}reference mep
{extra_comment}integrable dens name dens
{extra_comment}integrable vol name vol
{extra_comment}integrable elf name elf
{extra_comment}integrable wfn multipoles 2 name multipole
{extra_comment}integrable mep name mep
{extra_comment}integrable uslater name uslater
{extra_comment}yt nnm json mep_yt.json

### ================= ###
### Topology analysis ###
### ================= ###

reference wfn

### Valence electron density
POINTPROP valence "$wfn:v"
### Core electron density
POINTPROP core "$wfn:c"
### Gradient electron density
POINTPROP grad "$wfn:g"
### Laplacian electron density
POINTPROP lap "$wfn:l"
### Laplacian valence electron density
POINTPROP lap-valence "$wfn:lv"
### Laplacian core electron density
POINTPROP lap-core "$wfn:lc"
### HOMO electron density
POINTPROP homo "$wfn:HOMO"
### LUMO electron density
POINTPROP lumo "$wfn:LUMO"

### Thomas-Fermi kinetic energy density.
POINTPROP gtf
### Thomas-Fermi potential energy density (uses local virial).
POINTPROP vtf
### Thomas-Fermi total energy density (uses local virial).
POINTPROP htf

### Thomas-Fermi ked with Kirzhnits gradient correction.
POINTPROP gtf_kir
### Thomas-Fermi potential energy density with Kirzhnits gradient correction (uses local virial).
POINTPROP vtf_kir
### Thomas-Fermi total energy density with Kirzhnits gradient correction (uses local virial).
POINTPROP htf_kir

### Kinetic enregy density, g-version (grho * grho).
POINTPROP gkin
### Kinetic enregy density, k-version (rho * laprho).
POINTPROP kkin

### Localized-orbital locator.
POINTPROP lol
### Electronic energy density, gkin+vir.
POINTPROP he
### Lagrangian density (-1/4 laprho).
POINTPROP lag
### The reduced density gradient,
POINTPROP rdg

### Electronic potential energy density (virial field).
POINTPROP vir
### Electron localization function.
POINTPROP elf
### Localized-orbital locator, with Kirzhnits k.e.d.
POINTPROP lol_kir

### molecular electrostatic potential.
POINTPROP mep "mep($wfn)"
### Slater potential Ux. The HF exchange energy is ∫ρ(r)Ux(r)dr.
POINTPROP uslater "uslater($wfn)"
### calculate the Schrodinger stress tensor of the reference field. The virial field is the trace of this tensor
POINTPROP stress

auto
CPREPORT cp_auto.json verylong
"""
    with open(f"cri", "w") as fopen:
        fopen.write(critic2_in)

    call(["critic2", f"cri", f"cro"])

    with open(f"dens.cube", "rb") as fopen_in:
        with gzip.open(f"dens.cube.gz", "wb") as fopen_out:
            fopen_out.writelines(fopen_in)
    with open(f"elf.cube", "rb") as fopen_in:
        with gzip.open(f"elf.cube.gz", "wb") as fopen_out:
            fopen_out.writelines(fopen_in)
    if extra_fields:
        with open(f"mep.cube", "rb") as fopen_in:
            with gzip.open(f"mep.cube.gz", "wb") as fopen_out:
                fopen_out.writelines(fopen_in)
        with open(f"uslater.cube", "rb") as fopen_in:
            with gzip.open(f"uslater.cube.gz", "wb") as fopen_out:
                fopen_out.writelines(fopen_in)

    for filename in output_dict.keys():
        if os.path.exists(filename):
            if filename.endswith(".gz"):
                with gzip.open(filename, "rb") as fopen:
                    output_dict[filename] = fopen.read()
            else:
                with open(filename, "r") as fopen:
                    output_dict[filename] = fopen.read()
        else:
            del output_dict[filename]

    os.chdir(scratch_dir_parent)
    shutil.rmtree(scratch_dir, ignore_errors=True)
    return output_dict


def local_qcng_worker(inp, local_options, program=None, procedure=None):

    import qcengine as qcng
    import os

    if "SCRATCH" in os.environ:
        qcng.get_config(task_config={"scratch_directory": "$SCRATCH"})
        local_options["scratch_directory"] = os.environ["SCRATCH"]
    else:
        qcng.get_config(task_config={"scratch_directory": "/tmp"})
        local_options["scratch_directory"] = "/tmp"

    if procedure == None:
        ret = qcng.compute(
            inp, 
            program,
            task_config=local_options,
            )
        return ret
    else:
        pro = qcng.compute_procedure(
            inp, 
            procedure,
            task_config=local_options,
        )
        return pro

if HAS_RAY:
    @ray.remote(num_cpus=4)
    def remote_qcng_worker(inp, local_options, program, procedure):
        return local_qcng_worker(inp, local_options, program, procedure)

def qcng_worker(inp, local_options={}, program=None, procedure=None):
    if HAS_RAY:
        return remote_qcng_worker.remote(
            inp, 
            local_options,
            program,
            procedure
        )
    else:
        return local_qcng_worker(
            inp, 
            local_options,
            program,
            procedure
        )

def qcmol_to_rdmol(qcmol):

    __doc__ = """
    Converts qcmol to rdkit. Not very reliable. Better use
    Molecule from openff-toolkit:

    `rdmol = Molecule.from_qcschema(...).to_rdkit()`
    """

    if isinstance(qcmol, dict):
        atomic_numbers = qcmol['atomic_numbers']
        connectivity   = qcmol['connectivity']
        mol_charge     = qcmol['molecular_charge']
        N_atoms        = len(atomic_numbers)
    else:
        atomic_numbers = qcmol.atomic_numbers
        connectivity   = qcmol.connectivity
        mol_charge     = qcmol.molecular_charge
        N_atoms        = len(atomic_numbers)

    bond_dict = {
        1 : Chem.BondType.SINGLE,
        2 : Chem.BondType.DOUBLE,
        3 : Chem.BondType.TRIPLE,
    }

    emol    = Chem.EditableMol(Chem.Mol())
    bo_list = np.zeros(N_atoms, dtype=float)
    for conn in connectivity:
        atom1, atom2, bond = conn
        bo_list[atom1]    += bond
        bo_list[atom2]    += bond
    for atom_idx in range(N_atoms):
        at_num = atomic_numbers[atom_idx] 
        atom = Chem.Atom(
            int(at_num)
            )
        if at_num == 7:
            if bo_list[atom_idx] > 3.:
                atom.SetFormalCharge(1)
        emol.AddAtom(atom)
    for conn in connectivity:
        atom1, atom2, bond = conn
        emol.AddBond(
            atom1, 
            atom2, 
            bond_dict[int(bond)]
            )

    try:
        mol = emol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def correct_angle(
    geometry: np.ndarray,
    dih_idxs: np.ndarray,
    dih_target: _UNIT_QUANTITY):

    ### If the actual eq value is very different from the
    ### target value, save the actual value. Note, this is
    ### highly unlikely.
    ang = pts_to_dihedral(
            *(geometry[tuple(dih_idxs)] * unit.bohrs)
            )
    ang = ang.in_units_of(unit.degree)
    dih_target = dih_target.in_units_of(unit.degree)
    if abs(ang._value - dih_target._value) > 0.1:
        return ang
    else:
        return dih_target


def has_Hbond(
    atomic_numbers: np.ndarray,
    geometry: np.ndarray,
    connectivity: list):

    DIST_CUT  = 2.5  * unit.angstrom
    ANGLE_CUT = 120. * unit.degree

    N_atoms        = len(atomic_numbers)
    geometry       = geometry * unit.bohrs
    geometry       = geometry.in_units_of(unit.angstrom)

    connectivity_dict = dict()
    for c in connectivity:
        if not c[0] in connectivity_dict:
            connectivity_dict[c[0]] = [c[1]]
        else:
            connectivity_dict[c[0]].append(c[1])
        if not c[1] in connectivity_dict:
            connectivity_dict[c[1]] = [c[0]]
        else:
            connectivity_dict[c[1]].append(c[0])

    don_list = list()
    acc_list = list()
    hyd_list = list()
    for atm_idx in range(N_atoms):
        atm_n = atomic_numbers[atm_idx]
        ### Acceptors:
        ### Is atm_idx O/N?
        if atm_n in [7,8]:
            acc_list.append(atm_idx)
        for atm_idx_nghbr in connectivity_dict[atm_idx]:
            atm_n_nghbr = atomic_numbers[atm_idx_nghbr]
            ### Donors:
            ### Is atm_idx O/N and bound to hydrogen?
            if atm_n in [7,8] and atm_n_nghbr == 1:
                don_list.append(atm_idx)
                hyd_list.append(atm_idx_nghbr)

    if len(don_list) == 0 or len(acc_list) == 0:
        return False

    for acc_idx in acc_list:
        for don_idx, hyd_idx in zip(don_list, hyd_list):
            if acc_idx == don_idx:
                continue
            acc_crd = geometry[acc_idx]
            don_crd = geometry[don_idx]
            hyd_crd = geometry[hyd_idx]
            dist    = pts_to_bond(hyd_crd, acc_crd)
            dist    = dist.in_units_of(unit.angstrom)
            angle   = pts_to_angle(don_crd, hyd_crd, acc_crd)
            angle   = angle.in_units_of(unit.degree)
            if dist < DIST_CUT and angle > ANGLE_CUT:
                return True
    return False


def retrieve_complete_torsiondataset(
    smiles_list: list = list(),
    with_units: bool = True,
    n_conformers: int = 5,
    program = "openmm",
    model = {"method": "openff_unconstrained-1.3.0", "basis": "smirnoff"},
    ncores = 2,
    memory = 2,
    torsion_smarts = ["[*:1]~[*!r:2]-[*!r:3]~[*:4]"],
    grid_spacing = 15.,
    check_Hbond = True,):

    import qcelemental as qcel
    import qcengine as qcng

    dataset_dict = dict()
    worker_dict  = dict()
    for smiles in smiles_list:

        dataset_dict[smiles] = dict()
        worker_dict[smiles]  = dict()

        offmol  = Molecule.from_smiles(smiles)
        rdmol   = offmol.to_rdkit()
        rank_list = list(Chem.CanonicalRankAtoms(rdmol, breakTies=False))
        torsion_rdmol_list = [Chem.MolFromSmarts(smarts) for smarts in torsion_smarts]
        torsion_list = list()
        for torsion_rdmol in torsion_rdmol_list:
            for match in rdmol.GetSubstructMatches(torsion_rdmol):
                if len(match) != 4:
                    warnings.warn(
                        f"Skipping {match} for {smiles}.")
                rank1 = rank_list[match[1]]
                rank2 = rank_list[match[2]]
                found_torsion = False
                for torsion in torsion_list:
                    rank1_q = rank_list[torsion[1]]
                    rank2_q = rank_list[torsion[2]]
                    if torsion == match:
                        found_torsion = True
                    if rank1 == rank1_q and rank2 == rank2_q:
                        found_torsion = True
                    if rank1 == rank2_q and rank2 == rank1_q:
                        found_torsion = True
                    if found_torsion:
                        break
                if not found_torsion:
                    torsion_list.append(match)

        ### Generate a reasonable conformer
        offmol.generate_conformers(
            n_conformers=n_conformers
        )
        qcentry = offmol.to_qcschema()
        ### First minimize energy for everything
        for conf_i in range(len(offmol.conformers)):
            dataset_dict[smiles][conf_i] = dict()
            worker_dict[smiles][conf_i]  = list()

            qcentry.geometry[:] = offmol.conformers[conf_i].value_in_unit(_LENGTH_AU)
            qcemol = qcel.models.Molecule.from_data(copy.deepcopy(qcentry).dict())
            for match in torsion_list:
                inp = {
                    "driver" : "gradient",
                    "model"  : model,
                    }
                td_spec = {
                    "keywords" : {
                        "dihedrals"       : [match],
                        "grid_spacing"    : [grid_spacing],
                        "dihedral_ranges" : [(-180., 180)]

                    },
                    "input_specification" : inp,
                    "initial_molecule"    : [qcemol],
                    "optimization_[Bspec"   : {
                        "keywords": {
                            "program"  : program,
                            "coordsys" : "tric",
                            "enforce"  : 0.1,
                            "epsilon"  : 0.0,
                            "reset"    : True,
                            "qccnv"    : True,
                            "molcnv"   : False,
                            "check"    : 0,
                            "trust"    : 0.1,
                            "tmax"     : 0.3,
                            "maxiter"  : 600,
                            "convergence_set": "GAU",
                            "constraints": {}
                        },

                        "procedure" : "geomeTRIC",
                    },
                }

                worker_id = qcng_worker(
                    td_spec, 
                    local_options={
                        "memory" : memory, 
                        "ncores" : ncores
                        },
                    procedure="torsiondrive",
                    )

                worker_dict[smiles][conf_i].append(worker_id)
                dataset_dict[smiles][conf_i] = dict()

        
    for smiles in worker_dict.keys():
        for conf_i in worker_dict[smiles].keys():
            for dih_counter, worker_id in enumerate(worker_dict[smiles][conf_i]):
                if HAS_RAY:
                    opt = ray.get(worker_id)
                else:
                    opt = worker_id
                if isinstance(opt, qcng.util.FailedOperation):
                    warnings.warn(f"Found failed TorsionDrive Optimization at smiles={smiles} conf_i={conf_i} dih_counter={dih_counter}.")
                    continue

                dih_vals_list = opt.optimization_history.keys()
                dih_idxs      = opt.keywords.dihedrals
                final_geo     = list()
                final_ene     = list()
                final_dih     = list()

                for dih_val in dih_vals_list:
                    ### TODO: This will fail with multi-dihedral scans.
                    ###       Find a better way to convert str->float that
                    ###       won't break with multi-dihedrals.
                    dih_vals_float = float(dih_val)
                    try:
                        qcmol = opt.final_molecules[dih_val]
                    except:
                        warnings.warn(f"Could not retrieve dih={dih_val}. Skipping.")
                        continue

                    mol_has_hbond = False
                    if check_Hbond:
                        mol_has_hbond = has_Hbond(
                            qcmol.atomic_numbers,
                            qcmol.geometry,
                            qcmol.connectivity
                            )
                    if not mol_has_hbond:
                        geo_cp  = np.copy(qcmol.geometry).tolist()
                        ene_cp  = opt.final_energies[dih_val]
                        if with_units:
                            geo_cp *= _LENGTH_AU
                            ene_cp *= _ENERGY_AU
                        final_geo.append(geo_cp)
                        final_ene.append(ene_cp)
                        final_dih.append(list())
                        dih_val_float = correct_angle(
                            qcmol.geometry,
                            dih_idxs,
                            dih_vals_float * unit.degree)
                        if with_units:
                            dih_val_float = dih_val_float.in_units_of(
                                _ANGLE
                                )
                        else:
                            dih_val_float = dih_val_float._value
                        final_dih[-1].append(dih_val_float)
                    else:
                        warnings.warn(f"Found hbond for dih={dih_val}. Skipping.")

                print("Finishing", smiles, conf_i, dih_counter)
                dataset_dict[smiles][conf_i][dih_counter] = {
                    "final_geo"  : final_geo,
                    "final_ene"  : final_ene,
                    "dih"        : final_dih,
                    "dih_idxs"   : dih_idxs,
                    "qcentry"    : opt.initial_molecule[0].dict()
                }

    return dataset_dict


def retrieve_complete_dataset(
    smiles_list: list = list(),
    with_units: bool = True,
    n_conformers: int = 5,
    n_samples: int = 100,
    program = "openmm",
    model = {"method": "openff_unconstrained-1.3.0", "basis": "smirnoff"},
    ncores = 2,
    memory = 2,
    generate_forces_bonds = True,
    generate_forces_angles = False,
    generate_forces_torsions = False):

    import qcelemental as qcel
    import qcengine as qcng

    if program in ["psi4"]:
        numerical_hessian = False
    else:
        numerical_hessian = True

    diff_hes = 1.0e-4 * _LENGTH

    dataset_dict = dict()
    rdmol_dict   = dict()
    zmat_dict    = dict()
    worker_dict  = dict()
    nconf_dict   = dict()
    for smiles in smiles_list:
        offmol  = Molecule.from_smiles(smiles)
        top     = offmol.to_topology()
        rdmol   = offmol.to_rdkit()
        zmat    = ZMatrix(rdmol)
        rdmol_dict[smiles] = rdmol
        zmat_dict[smiles]  = zmat

        ### =================== ###
        ### GENERATE CONFORMERS ###
        ### =================== ###
        offmol.generate_conformers(
            n_conformers=n_conformers
        )
        qcentry = offmol.to_qcschema()

        ### =================== ###
        ### OPTIMIZE CONFORMERS ###
        ### =================== ###
        dataset_dict[smiles] = dict()
        worker_dict[smiles]  = dict()
        nconf_dict[smiles]   = len(offmol.conformers)
        for conf_i in range(len(offmol.conformers)):
            dataset_dict[smiles][conf_i] = dict()
            worker_dict[smiles][conf_i] = dict()
            worker_dict[smiles][conf_i]["geo"] = None
            worker_dict[smiles][conf_i]["hes"] = None
            worker_dict[smiles][conf_i]["frc"] = list()

            qcentry.geometry[:] = offmol.conformers[conf_i].value_in_unit(_LENGTH_AU)
            qcemol = qcel.models.Molecule.from_data(copy.deepcopy(qcentry).dict())
            inp = {
                "driver" : "gradient",
                "model"  :  model,
                }
            opt = {
                "keywords": {
                    "program"  : program,
                    "coordsys" : "dlc",
                    "enforce"  : 0.0,
                    "epsilon"  : 1e-05,
                    "reset"    : True,
                    "qccnv"    : False,
                    "molcnv"   : False,
                    "check"    : 0,
                    "trust"    : 0.1,
                    "tmax"     : 0.3,
                    "maxiter"  : 600,
                    "convergence_set": "GAU",
                    "constraints": {},
                },
                "input_specification" : inp,
                "initial_molecule"    : qcemol,
            }
            _worker_id = qcng_worker(
                opt, 
                local_options = { 
                    "memory" : memory, 
                    "ncores" : ncores
                    },
                procedure = "geometric",
                )
            worker_dict[smiles][conf_i]["geo"] = _worker_id

    N_remaining_jobs = 999
    while N_remaining_jobs > 0:
        time.sleep(5)
        N_remaining_jobs = 0
        for smiles in smiles_list:
            rdmol = rdmol_dict[smiles]
            zmat  = zmat_dict[smiles]
            for conf_i in range(nconf_dict[smiles]):
                if HAS_RAY:
                    worker_id_list, not_ready_list = ray.wait(
                        [worker_dict[smiles][conf_i]["geo"]]
                        )
                    if not_ready_list:
                        N_remaining_jobs += 1
                        continue
                    else:
                        worker_id = worker_id_list[0]
                        opt = ray.get(worker_id)
                else:
                    worker_id = worker_dict[smiles][conf_i]["geo"]
                    opt = worker_id

                if isinstance(opt, qcng.util.FailedOperation):
                    warnings.warn(f"Found failed Geometry Optimization for smiles={smiles} conf_i={conf_i}.")
                    warnings.warn(f"{opt}")
                    continue

                ### =========================== ###
                ### STORE ALL DATA AFTER OPTGEO ###
                ### =========================== ###
                qcentry   = opt.final_molecule
                final_geo = qcentry.geometry.tolist()
                final_ene = opt.energies[-1]
                final_frc = -1. * opt.trajectory[-1].return_result
                final_frc = final_frc.tolist()

                if with_units:
                    final_geo *= _LENGTH_AU
                    final_ene *= _ENERGY_AU
                    final_frc *= _FORCE_AU

                dataset_dict[smiles][conf_i] = {
                    "final_geo"     : final_geo,
                    "final_ene"     : final_ene,
                    "final_frc"     : final_frc,
                    "qcentry"       : opt.initial_molecule.dict(),
                    "qcentry_final" : qcentry.dict(),
                    }

                ### =========================== ###
                ### SUBMIT HESSIAN CALCULATIONS ###
                ### =========================== ###

                if numerical_hessian:
                    N_atoms = len(qcentry.atomic_numbers)

                    ### Default diff taken from Parsley paper:
                    ### doi.org/10.26434/chemrxiv.13082561.v2
                    diff = diff_hes.value_in_unit(_LENGTH_AU)
                    xyz  = copy.deepcopy(qcentry.geometry[:])

                    worker_hes_list = list()
                    for atm_idx in range(N_atoms):
                        worker_hes_list.append(list())
                        for crd_idx in range(3):
                            worker_hes_list[-1].append(list())
                            for diff_factor in [1., -2.]:
                                xyz[atm_idx][crd_idx] += diff_factor * diff
                                qcemol = qcel.models.Molecule(
                                    geometry = copy.deepcopy(xyz),
                                    symbols = qcentry.symbols,
                                    connectivity = qcentry.connectivity,
                                )
                                inp = {
                                    "driver"   : "gradient",
                                    "model"    : model,
                                    "molecule" : qcemol
                                    }
                                _worker_id = qcng_worker(
                                    inp, 
                                    local_options = { 
                                        "memory" : memory, 
                                        "ncores" : ncores
                                        },
                                    program = program,
                                    procedure = None,
                                    )
                                worker_hes_list[-1][-1].append(_worker_id)
                            ### !!! Reset XYZ !!!
                            ### =================
                            xyz[atm_idx][crd_idx] += diff
                    worker_dict[smiles][conf_i]['hes'] = worker_hes_list

                else:
                    inp = {
                        "driver"   : "hessian",
                        "model"    : model,
                        "molecule" : qcentry
                        }
                    _worker_id = qcng_worker(
                        inp, 
                        local_options = { 
                            "memory" : memory, 
                            "ncores" : ncores
                            },
                        program = program,
                        procedure = None,
                        )
                    worker_dict[smiles][conf_i]['hes'] = _worker_id

                ### ========================= ###
                ### SUBMIT FORCE CALCULATIONS ###
                ### ========================= ###

                z_crds_initial = zmat.build_z_crds(qcentry.geometry * _LENGTH_AU, with_units=True)
                for _ in range(n_samples):
                    z_crds_copy = copy.deepcopy(z_crds_initial)
                    for z_idx in range(1, zmat.N_atms):
                        z_crd   = z_crds_copy[z_idx]
                        ### unit is nm
                        if generate_forces_bonds:
                            ### Unit is nm
                            z_crd[0] += np.random.normal(0, 2.0e-3) * _LENGTH
                        if len(z_crd) > 1 and generate_forces_angles:
                            ### Unit is deg
                            z_crd[1] += np.random.normal(0, 2.0e+0) * _ANGLE
                        if len(z_crd) > 2 and generate_forces_torsions:
                            ### Unit is deg
                            z_crd[2] += np.random.normal(0, 5.0e+0) * _ANGLE
                        z_crds_copy[z_idx] = z_crd 
                    qcentry_cp = copy.deepcopy(qcentry)
                    crds3d     = zmat.build_cart_crds(z_crds_copy)
                    crds3d     = crds3d.value_in_unit(_LENGTH_AU)
                    qcemol     = qcel.models.Molecule(
                        geometry = crds3d,
                        symbols = qcentry.symbols,
                        connectivity = qcentry.connectivity,
                    )
                    inp = {
                        "driver"   : "gradient",
                        "model"    : model,
                        "molecule" : qcemol
                        }
                    _worker_id = qcng_worker(
                        inp, 
                        local_options = { 
                            "memory" : memory, 
                            "ncores" : ncores
                            },
                        program = program,
                        procedure = None,
                        )
                    worker_dict[smiles][conf_i]['frc'].append(_worker_id)

                ### =============================== ###
                ### FINALLY REMOVE GEOOPT WORKER_ID ###
                ### =============================== ###
                
                ### Set ObjectID to None. Otherwise this loop
                ### may run forever...
                worker_dict[smiles][conf_i]["geo"] = None
        
    for smiles in smiles_list:
        rdmol = rdmol_dict[smiles]
        N_atoms = rdmol.GetNumAtoms()
        for conf_i in range(nconf_dict[smiles]):

            ### =================== ###
            ### GATHER HESSIAN DATA ###
            ### =================== ###

            if worker_dict[smiles][conf_i]["hes"] != None:

                if numerical_hessian:
                    missing_data = False
                    worker_hes_list = worker_dict[smiles][conf_i]["hes"]
                    ### Adapted from ForceBalance routine in openmmio.py
                    ### https://github.com/leeping/forcebalance/blob/master/src/openmmio.py
                    hessian = np.zeros((N_atoms*3,
                                        N_atoms*3), dtype=float)
                    diff    = diff_hes.value_in_unit(_LENGTH_AU)
                    coef    = 1.0 / (diff * 2) # 1/2 step width
                    for atm_idx in range(N_atoms):
                        for crd_idx in range(3):
                            if HAS_RAY:
                                grad_plus  = ray.get(worker_hes_list[atm_idx][crd_idx][0])
                                grad_minus = ray.get(worker_hes_list[atm_idx][crd_idx][1])
                            else:
                                grad_plus  = worker_hes_list[atm_idx][crd_idx][0]
                                grad_minus = worker_hes_list[atm_idx][crd_idx][1]
                            if isinstance(grad_plus, qcng.util.FailedOperation):
                                warnings.warn(f"Found failed Hessian Calculation for smiles={smiles} conf_i={conf_i}.")
                                missing_data = True
                            if isinstance(grad_minus, qcng.util.FailedOperation):
                                warnings.warn(f"Found failed Hessian Calculation for smiles={smiles} conf_i={conf_i}.")
                                missing_data = True

                            if missing_data:
                                break

                            ### This is already the gradient
                            ### No need to convert it to force
                            grad_plus  = grad_plus.return_result
                            grad_minus = grad_minus.return_result

                            hessian[atm_idx*3+crd_idx] = np.ravel((grad_plus - grad_minus) * coef)

                        if missing_data:
                            break
                    if missing_data:
                        break

                    ### make hessian symmetric by averaging upper right and lower left
                    hessian += hessian.T
                    hessian *= 0.5

                    if with_units:
                        hessian *= _FORCE_AU * _LENGTH_AU**-1

                else:
                    worker_id = worker_dict[smiles][conf_i]["hes"]
                    if HAS_RAY:
                        opt = ray.get(worker_id)
                    else:
                        opt = worker_id
                    if isinstance(opt, qcng.util.FailedOperation):
                        warnings.warn(f"Found failed Hessian Calculation for smiles={smiles} conf_i={conf_i}.")
                        continue

                    hessian = opt.return_result.tolist()

                    if with_units:
                        hessian *= _FORCE_AU * _LENGTH_AU**-1

                dataset_dict[smiles][conf_i]["hessian"] = hessian

            else:
                dataset_dict[smiles][conf_i]["hessian"] = []

            ### ================= ###
            ### GATHER FORCE DATA ###
            ### ================= ###

            if worker_dict[smiles][conf_i]['frc'] != None:

                ene_list   = list()
                force_list = list()
                geo_list   = list()
                for worker_id in worker_dict[smiles][conf_i]['frc']:
                    if HAS_RAY:
                        opt = ray.get(worker_id)
                    else:
                        opt = worker_id
                    if isinstance(opt, qcng.util.FailedOperation):
                        warnings.warn(f"Found failed Force Calculation for smiles={smiles} conf_i={conf_i}.")
                        continue
                    force_cp = copy.copy(-1. * opt.return_result).tolist()
                    if opt.properties.scf_total_energy == None:
                        ene_cp   = copy.copy(opt.properties.return_energy)
                    else:
                        ene_cp   = copy.copy(opt.properties.scf_total_energy)
                    geo_cp   = copy.copy(opt.molecule.geometry).tolist()
                    geo_cp  *= _LENGTH_AU
                    geo_cp   = geo_cp.value_in_unit(_LENGTH_AU)
                    if with_units:
                        force_cp *= _FORCE_AU
                        ene_cp   *= _ENERGY_AU
                        geo_cp   *= _LENGTH_AU
                    force_list.append(force_cp)
                    ene_list.append(ene_cp)
                    geo_list.append(geo_cp)

                dataset_dict[smiles][conf_i]["force_list"] = force_list
                dataset_dict[smiles][conf_i]["geo_list"]   = geo_list
                dataset_dict[smiles][conf_i]["ene_list"]   = ene_list

            else:

                dataset_dict[smiles][conf_i]["force_list"] = list()
                dataset_dict[smiles][conf_i]["geo_list"]   = list()
                dataset_dict[smiles][conf_i]["ene_list"]   = list()

    return dataset_dict


def retrieve_qtaim_data(
    input_dataset_dict,
    method="b3lyp",
    basis="dgdzvp",
    extra="EmpiricalDispersion=GD3",
    extra_fields = False, ### Computing these things is rather expensive. Omit for now.
    save_cube = False, ### Save cube files
    save_cri = False, ### Save Critic2 in/out files
    ):

    from openmm import unit
    import json

    fchk_worker_dict = dict()
    for smiles in input_dataset_dict:
        fchk_worker_dict[smiles] = dict()
        for conf_i in input_dataset_dict[smiles]:
            qcentry   = input_dataset_dict[smiles][conf_i]["qcentry"]
            final_geo = input_dataset_dict[smiles][conf_i]["final_geo"] * unit.bohr
            symbols   = input_dataset_dict[smiles][0]["qcentry"]["symbols"].tolist()
            N_atoms   = len(symbols)
            charge    = qcentry["molecular_charge"]
            multiplicity = qcentry["molecular_multiplicity"]
            xyz_str   = f"{int(charge)}, {int(multiplicity)}\n"
            for atm_idx in range(N_atoms):
                crd3d    = final_geo[atm_idx].value_in_unit(unit.angstrom)
                xyz_str += f"{symbols[atm_idx]} {crd3d[0]} {crd3d[1]} {crd3d[2]} \n"
            input_dict = {
                "method"    : method,
                "basis"     : basis,
                "num_procs" : 2,
                "xyz_str"   : xyz_str,
                "extra"     : extra,
                "name"      : smiles
            }
            fchk_worker_dict[smiles][conf_i] = compute_gaussian_fchk.remote(input_dict)

    qtaim_worker_dict = dict()
    for smiles in input_dataset_dict:
        qtaim_worker_dict[smiles] = dict()
        for conf_i in input_dataset_dict[smiles]:
            fchk_str = ray.get(fchk_worker_dict[smiles][conf_i])
            input_dict = {
                "num_procs"    : 4,
                "fchk_str"     : fchk_str,
                "extra_fields" : extra_fields,
                "name"         : smiles
            }
            qtaim_worker_dict[smiles][conf_i] = compute_critic2_qtaim_props.remote(input_dict)

    dataset_dict = dict()
    for smiles in input_dataset_dict:
        dataset_dict[smiles] = dict()
        for conf_i in input_dataset_dict[smiles]:
            dataset_dict[smiles][conf_i] = dict()
            output_dict = ray.get(qtaim_worker_dict[smiles][conf_i])
            for filename, filecontent in output_dict.items():
                if filename.endswith(".cube.gz") or filename.endswith(".cube"):
                    if save_cube:
                        dataset_dict[smiles][conf_i][filename] = filecontent
                elif filename.endswith("cri") or filename.endswith("cro"):
                    if save_cri:
                        dataset_dict[smiles][conf_i][filename] = filecontent
                elif filename.endswith(".json"):
                    description = filename.replace(".json", "")
                    dataset_dict[smiles][conf_i][description] = json.loads(filecontent)
                else:
                    ### Nix mache ...
                    pass

    return dataset_dict


def retrieve_dataset(
    dataset_name_list: list = list(),
    name_list: list = list(),
    method: list = list(),
    basis: list = list(),
    driver: str = "hessian",
    element_list: list = list(),
    with_units: bool = True,
    merge: bool = True,
    fractal_client=None):
    
    import qcportal as ptl

    if driver.lower() == "hessian":
        driver = ptl.singlepoint.SinglepointDriver.hessian
    elif driver.lower() == "gradient":
        driver = ptl.singlepoint.SinglepointDriver.gradient
    else:
        driver = None

    if fractal_client == None:
        fractal_client = ptl.PortalClient("https://api.qcarchive.molssi.org:443/")

    method_cp = [m.lower() for m in method]
    basis_cp  = [b.lower() for b in basis]
    element_list_cp = list()
    for element in element_list:
        if isinstance(element, str):
            element_list_cp.append(get_atomic_number(element))
        elif isinstance(element, int):
            if not element > 0:
                raise ValueError(
                    "Each element number must be greater than 0")
            element_list_cp.append(element)
        else:
            raise ValueError(
                f"element {element} not understood.")
    element_list_cp = list(set(element_list_cp))
    element_list_cp = sorted(element_list_cp)

    if not dataset_name_list:
        dataset_name_list = fractal_client.list_datasets(
            "singlepoint", 
            aslist=True)

    dataset_dict = dict()
    exclude_list = list()
    for dataset_name in dataset_name_list:
        print(f"Scanning dataset {dataset_name} ...")
        ds = fractal_client.get_dataset(
            'singlepoint', 
            dataset_name
            )
        if len(list(ds.entry_names)) == 0:
            continue
        
        if method_cp:
            if not spec_dict['method'].lower() in method_cp:
                continue
        if basis_cp:
            if not spec_dict['basis'].lower() in basis_cp:
                continue

        for name in ds.entry_names:
            print(f"Scanning entry {name} ...")
            if name_list:
                if not name in name_list:
                    continue
            entry = ds.get_entry(name)
            molecule_id = entry.molecule.id
            for result_record in fractal_client.query_singlepoints(molecule_id=molecule_id, driver=driver):
                if method_cp:
                    if not result_record.specification.method.lower() == method_cp:
                        continue
                if basis_cp:
                    if not result_record.specification.basis.lower() == basis_cp:
                        continue
                if not result_record.status.lower() == 'complete':
                    warnings.warn(
                        f"Could not retrieve {name}. Status {result_record.status}. Skipping."
                        )
                    continue
            
                qcmol              = result_record.molecule
                qcmol_element_list = copy.copy(qcmol.atomic_numbers)
                qcmol_element_list = list(set(qcmol_element_list))
                qcmol_element_list = sorted(qcmol_element_list)
                if element_list_cp:
                    if not qcmol_element_list == element_list_cp:
                        continue
                cp_result  = np.copy(result_record.return_result).tolist()
                cp_geo     = np.copy(qcmol.geometry)
                if with_units:
                    ### Convert to Hessian units
                    if driver == "hessian":
                        cp_result *= _FORCE_AU 
                        cp_result *= _LENGTH_AU**-1
                    cp_geo *= _LENGTH_AU

                dataset_dict[name] = {
                    "result"    : cp_result,
                    "structure" : cp_geo
                }

    return dataset_dict


def retrieve_optimization_dataset(
    dataset_name_list: list = list(),
    name_list: list = list(),
    method: list = list(),
    basis: list = list(),
    pattern_list: list = list(),
    element_list: list = list(),
    with_units: bool = True,
    merge: bool = True,
    fractal_client=None,
    generate_forces_bonds: bool = True,
    generate_forces_angles: bool = False,
    generate_forces_torsions: bool = False,
    n_samples: int = 100):

    import qcelemental as qcel
    import qcengine as qcng

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    ### This should work for SMILES *and* SMARTS
    pattern_mol_list = [Chem.MolFromSmarts(p) for p in pattern_list]
    if None in pattern_mol_list:
        raise ValueError(
            "Could not read all patterns in pattern_list.")

    if fractal_client == None:
        import qcportal as ptl
        fractal_client = ptl.PortalClient(
            "https://api.qcarchive.molssi.org:443/"
            )

    method_cp = [m.lower() for m in method]
    basis_cp  = [b.lower() for b in basis]
    element_list_cp = list()
    for element in element_list:
        if isinstance(element, str):
            element_list_cp.append(get_atomic_number(element))
        elif isinstance(element, int):
            if not element > 0:
                raise ValueError(
                    "Each element number must be greater than 0")
            element_list_cp.append(element)
        else:
            raise ValueError(
                f"element {element} not understood.")
    element_list_cp = list(set(element_list_cp))
    element_list_cp = sorted(element_list_cp)

    if not dataset_name_list:
        dataset_name_list = fractal_client.list_datasets(
            "optimization", 
            aslist=True
            )

    dataset_dict = dict()
    exclude_list = list()
    for dataset_name in dataset_name_list:
        print(f"Scanning dataset {dataset_name} ...")
        ds = fractal_client.get_dataset(
            'optimization',
             dataset_name
            )

        if not name_list:
            name_list_query = ds.entry_names
        else:
            name_list_query = copy.copy(name_list)

        for name in name_list_query:
            print(f"Scanning entry {name} ...")

            try:
                entry    = ds.get_entry(name)
                qcrecord = ds.get_record(
                    name, 
                    specification_name="default"
                    )
            except:
                continue
            if not qcrecord.status.lower() == "complete":
                warnings.warn(
                    f"Could not retrieve {name}. Status {qcrecord.status}. Skipping."
                    )
                continue
            if method_cp:
                if not qcrecord.specification.qc_specification.method.lower() in method_cp:
                    continue
            if basis_cp:
                if not qcrecord.specification.qc_specification.basis.lower() in basis_cp:
                    continue

            qcmol = qcrecord.final_molecule
            try:
                ### First try openff
                rdmol = Molecule.from_qcschema(entry).to_rdkit()
            except:
                rdmol = None
                pass
            try:
                if rdmol == None:
                    ### Then try building from qcmol directly
                    rdmol = qcmol_to_rdmol(qcmol)
            except:
                rdmol = None
                pass

            if rdmol != None:
                if Chem.MolToSmiles(rdmol) in exclude_list:
                    continue
            if pattern_mol_list and rdmol != None:
                found_pattern = True
                for p_mol in pattern_mol_list:
                    if not rdmol.HasSubstructureMatch(p_mol):
                        found_pattern = False
                        exclude_list.append(Chem.MolToSmiles(rdmol))
                        break
                if not found_pattern:
                    continue
            if pattern_mol_list and rdmol == None:
                continue
            qcmol_element_list = copy.copy(qcmol.atomic_numbers)
            qcmol_element_list = list(set(qcmol_element_list))
            qcmol_element_list = sorted(qcmol_element_list)
            ### List of retrieval elements and actual elements must
            ### match exactly.
            if element_list_cp:
                if not sorted(qcmol_element_list) == element_list_cp:
                    if rdmol != None:
                        exclude_list.append(Chem.MolToSmiles(rdmol))
                    continue

            force_list = list()
            geo_list   = list()
            ene_list   = list()
            if generate_forces_bonds or generate_forces_angles or generate_forces_torsions:
                print("Generating forces ...")
                rdmol = None
                try:
                    rdmol = Molecule.from_qcschema(entry).to_rdkit()
                except:
                    rdmol = None
                    pass
                try:
                    if rdmol == None:
                        rdmol = qcmol_to_rdmol(qcmol)
                except:
                    rdmol = None
                    pass
                if rdmol == None:
                    warnings.warn(f"Could not generate forces for {name}. Skipping.")
                    continue
                zmat = ZMatrix(rdmol)
                geo  = qcrecord.final_molecule.geometry
                z_crds_initial = zmat.build_z_crds(geo * _LENGTH_AU, with_units=True)
                worker_list = list()
                for _ in range(n_samples):
                    z_crds_copy = copy.deepcopy(z_crds_initial)
                    for z_idx in range(1, zmat.N_atms):
                        rdatm1  = rdmol.GetAtomWithIdx(zmat.z[z_idx][0])
                        rdatm2  = rdmol.GetAtomWithIdx(zmat.z[z_idx][1])
                        number1 = rdatm1.GetAtomicNum()
                        number2 = rdatm2.GetAtomicNum()
                        z_crd   = z_crds_copy[z_idx]

                        if generate_forces_bonds:
                            ### Unit is nm
                            z_crd[0] += np.random.normal(0, 2.0e-3) * _LENGTH
                        if len(z_crd) > 1 and generate_forces_angles:
                            ### Unit is deg
                            z_crd[1] += np.random.normal(0, 2.0e+0) * _ANGLE
                        if len(z_crd) > 2 and generate_forces_torsions:
                            ### Unit is deg
                            z_crd[2] += np.random.normal(0, 5.0e+0) * _ANGLE
                        z_crds_copy[z_idx] = z_crd 

                    crds3d    = zmat.build_cart_crds(z_crds_copy).value_in_unit(unit.angstrom)
                    input_str = ""
                    for atm_idx in range(zmat.N_atms):
                        rdatm   = rdmol.GetAtomWithIdx(atm_idx)
                        number  = rdatm.GetAtomicNum()
                        element = get_atomic_symbol(number)
                        crds    = crds3d[atm_idx]
                        input_str += f"{element} {crds[0]}, {crds[1]}, {crds[2]}\n"
                    qcemol = qcel.models.Molecule.from_data(input_str)
                    inp = {
                        "molecule": qcemol,
                        "driver": "gradient",
                        "model": {
                            "method": qcrecord.specification.qc_specification.method, 
                            "basis": qcrecord.specification.qc_specification.basis
                            }
                        }
                    worker_id = qcng_worker(
                        inp, 
                        local_options={"memory": 2, "ncores": 4}
                        )
                    worker_list.append(worker_id)

                for worker_id in worker_list:
                    if HAS_RAY:
                        ret = ray.get(worker_id)
                    else:
                        ret = worker_id
                    if isinstance(ret, qcng.util.FailedOperation):
                        warnings.warn(f"Found failed Force Calculation.")
                        continue
                    force_cp = copy.copy(-1. * ret.return_result).tolist()
                    ene_cp   = copy.copy(ret.properties.scf_total_energy)
                    geo_cp   = copy.copy(ret.molecule.geometry).tolist()
                    geo_cp  *= _LENGTH_AU
                    geo_cp   = geo_cp.value_in_unit(_LENGTH_AU)
                    if with_units:
                        force_cp *= _FORCE_AU
                        ene_cp   *= _ENERGY_AU
                        geo_cp   *= _LENGTH_AU
                    force_list.append(force_cp)
                    ene_list.append(ene_cp)
                    geo_list.append(geo_cp)

            else:
                for result_records in qcrecord.trajectory:
                    _return_result = np.array(result_records.return_result)
                    force_cp  = np.copy(-1. * _return_result).tolist()
                    if with_units:
                        force_cp *= _FORCE_AU
                    force_list.append(force_cp)
                    geo_cp  = np.copy(result_records.molecule.geometry).tolist()
                    if with_units:
                        geo_cp *= _LENGTH_AU
                    geo_list.append(geo_cp)
                for ene in qcrecord.energies:
                    ene_cp = ene
                    if with_units:
                        ene_cp = ene * _ENERGY_AU
                    ene_list.append(ene_cp)

            final_geo = qcrecord.final_molecule.geometry
            final_geo = np.copy(final_geo).tolist()
            final_ene = qcrecord.energies

            if with_units:
                final_geo *= _LENGTH_AU
                final_ene *= _ENERGY_AU

            if merge:
                if rdmol != None:
                    rdmol = Chem.RemoveHs(rdmol)
                    smiles_name = Chem.MolToSmiles(rdmol)
                else:
                    smiles_name = name
            else:
                smiles_name = name

            if not smiles_name in dataset_dict:
                dataset_dict[smiles_name] = dict()
            if name in dataset_dict[smiles_name]:
                name += f"-{len(dataset_dict[smiles_name])}"
            dataset_dict[smiles_name][name] = {
                "final_geo"  : final_geo,
                "final_ene"  : final_ene,
                "qcentry"    : entry.dict(),
                "force_list" : force_list,
                "geo_list"   : geo_list,
                "ene_list"   : ene_list,
                }

    return dataset_dict


def retrieve_torsiondrive_dataset(
    dataset_name_list: list = list(),
    name_list: list = list(),
    method: list = list(),
    basis: list = list(),
    pattern_list: list = list(),
    element_list: list = list(),
    check_Hbond: bool = True,
    extra_data: bool = True,
    with_units: bool = True,
    merge: bool = True,
    fractal_client=None):

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    ### This should work for SMILES *and* SMARTS
    pattern_mol_list = [Chem.MolFromSmarts(p) for p in pattern_list]
    if None in pattern_mol_list:
        raise ValueError(
            "Could not read all patterns in pattern_list.")

    if fractal_client == None:
        import qcportal as ptl
        fractal_client = ptl.PortalClient(
            "https://api.qcarchive.molssi.org:443/")

    method_cp = [m.lower() for m in method]
    basis_cp  = [b.lower() for b in basis]
    element_list_cp = list()
    for element in element_list:
        if isinstance(element, str):
            element_list_cp.append(get_atomic_number(element))
        elif isinstance(element, int):
            if not element > 0:
                raise ValueError(
                    "Each element number must be greater than 0")
            element_list_cp.append(element)
        else:
            raise ValueError(
                f"element {element} not understood.")
    element_list_cp = list(set(element_list_cp))
    element_list_cp = sorted(element_list_cp)

    if not dataset_name_list:
        dataset_name_list = fractal_client.list_datasets(
            "torsiondrive", 
            aslist=True)

    dataset_dict = dict()
    exclude_list = list()
    for dataset_name in dataset_name_list:
        print(f"Scanning dataset {dataset_name} ...")
        ds = fractal_client.get_dataset(
            'torsiondrive',
             dataset_name
             )

        if not name_list:
            name_list_query = ds.entry_names
        else:
            name_list_query = copy.copy(name_list)

        for name in name_list_query:
            print(f"Scanning entry {name} ...")

            try:
                entry       = ds.get_entry(name)
                qcrecord_td = ds.get_record(
                    name, 
                    specification_name="default"
                    )
            except:
                continue

            opt_history = qcrecord_td.minimum_optimizations
            try:
                dih_vals_list = list(opt_history.keys())
            except:
                warnings.warn(f"Could not retrieve {name}. Skipping.")
                continue

            if method_cp:
                if not qcrecord_td.specification.qc_specification.method.lower() in method_cp:
                    continue
            if basis_cp:
                if not qcrecord_td.specification.qc_specification.basis.lower() in basis_cp:
                    continue

            try:
                qcmol = qcrecord_td.minimum_optimizations[dih_vals_list[0]].final_molecule
            except:
                continue
            try:
                ### First try openff
                rdmol = Molecule.from_qcschema(entry).to_rdkit()
            except:
                rdmol = None
                pass
            try:
                if rdmol == None:
                    ### Then try building from qcmol directly
                    rdmol = qcmol_to_rdmol(qcmol)
            except:
                rdmol = None
                pass

            if pattern_mol_list and rdmol != None:
                found_pattern = True
                for p_mol in pattern_mol_list:
                    if not rdmol.HasSubstructureMatch(p_mol):
                        found_pattern = False
                        exclude_list.append(Chem.MolToSmiles(rdmol))
                        break
                if not found_pattern:
                    continue
            if pattern_mol_list and rdmol == None:
                continue
            qcmol_element_list = copy.copy(qcmol.atomic_numbers)
            qcmol_element_list = list(set(qcmol_element_list))
            qcmol_element_list = sorted(qcmol_element_list)
            ### List of retrieval elements and actual elements must
            ### match exactly.
            if element_list_cp:
                if not sorted(qcmol_element_list) == element_list_cp:
                    if rdmol != None:
                        exclude_list.append(Chem.MolToSmiles(rdmol))
                    continue

            dih_idxs   = qcrecord_td.specification.keywords.dihedrals
            N_torsions = len(dih_idxs)
            
            final_geo = list()
            final_ene = list()
            final_dih = list()

            geo_list   = list()
            ene_list   = list()
            dih_list   = list()
            force_list = list()

            for dih_vals in dih_vals_list:
                if qcrecord_td.minimum_optimizations[dih_vals].status.lower() != 'complete':
                    continue
                dih_vals_float = np.array(list(dih_vals), dtype=float)
                try:
                    qcmol = qcrecord_td.minimum_optimizations[dih_vals].final_molecule
                except:
                    warnings.warn(f"Could not retrieve {name}, {dih_vals}. Skipping.")
                    continue
                mol_has_hbond = False
                if check_Hbond:
                    mol_has_hbond = has_Hbond(
                        qcmol.atomic_numbers,
                        qcmol.geometry,
                        qcmol.connectivity
                        )
                if not mol_has_hbond:
                    geo_cp  = np.copy(qcmol.geometry).tolist()
                    ene_cp  = qcrecord_td.final_energies[dih_vals]
                    if with_units:
                        geo_cp *= _LENGTH_AU
                        ene_cp *= _ENERGY_AU
                    final_geo.append(geo_cp)
                    final_ene.append(ene_cp)
                    final_dih.append(list())
                    for dih_id in range(N_torsions):
                        dih_val_float = correct_angle(
                            qcmol.geometry,
                            dih_idxs,
                            dih_vals_float[dih_id]*unit.degree)
                        if with_units:
                            dih_val_float = dih_val_float.in_units_of(
                                _ANGLE
                                )
                        else:
                            dih_val_float = dih_val_float._value
                        final_dih[-1].append(dih_val_float)
                else:
                    warnings.warn(f"Found hbond for {name}, {dih_vals}. Skipping.")

                if extra_data:
                    ### Loop over all non-eq geometries for dih coordinate
                    ### from minimization trajecory
                    for qcrecord in qcrecord_td.optimizations[dih_vals]:
                        if qcrecord.status.lower() != 'complete':
                            continue
                        for result_records in qcrecord.trajectory:
                            _return_result = np.array(result_records.return_result)
                            force_cp  = np.copy(-1. * _return_result).tolist()
                            if with_units:
                                force_cp *= _FORCE_AU
                            force_list.append(force_cp)
                            geo_cp  = np.copy(result_records.molecule.geometry).tolist()
                            if with_units:
                                geo_cp *= _LENGTH_AU
                            geo_list.append(geo_cp)
                            dih_list.append(list())
                            for dih_id in range(N_torsions):
                                dih_val_float = correct_angle(
                                                    qcmol.geometry,
                                                    dih_idxs,
                                                    dih_vals_float[dih_id]*unit.degree)
                                if with_units:
                                    dih_val_float = dih_val_float.in_units_of(
                                        _ANGLE
                                        )
                                else:
                                    dih_val_float = dih_val_float._value
                                dih_list[-1].append(dih_val_float)

                        for ene in qcrecord.energies:
                            ene_cp = ene
                            if with_units:
                                ene_cp = ene * _ENERGY_AU
                            ene_list.append(ene_cp)

            if merge:
                if rdmol != None:
                    rdmol = Chem.RemoveHs(rdmol)
                    smiles_name = Chem.MolToSmiles(rdmol)
                else:
                    smiles_name = name
            else:
                smiles_name = name

            if not smiles_name in dataset_dict:
                dataset_dict[smiles_name] = dict()
            if name in dataset_dict[smiles_name]:
                name += f"-{len(dataset_dict[smiles_name])}"
            ### Must have a dict at lowest layer due to 
            ### comapbility with output generated from
            ### `retrieve_complete_torsiondataset`.
            dataset_dict[smiles_name][name] = {
                0 : {
                        "final_geo"  : final_geo,
                        "final_ene"  : final_ene,
                        "dih"        : final_dih,
                        "dih_idxs"   : dih_idxs,
                        "qcentry"    : entry.dict(),
                        "force_list" : force_list,
                        "geo_list"   : geo_list,
                        "ene_list"   : ene_list,
                        "dih_list"   : dih_list
                    }
                }

    return dataset_dict


class RayWrapper(object):

    def __init__(self, is_ray=True):

        self.is_ray = True

    def __call__(self, func):

        if self.is_ray:
            def rayfunc(*args, **kwargs):
                f = func.remote(*args, **kwargs)
                return f
        else:
            rayfunc = func

        return rayfunc


class NoAccessError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def is_type_or_raise(name, query_object, target_type):
    if isinstance(query_object, target_type):
        return True
    raise TypeError(f"{name} is of type {type(query_object)} but must be {target_type}")

def is_one_of_types_or_raise(name, query_object, target_type_list):
    for target_type in target_type_list:
        if isinstance(query_object, target_type):
            return True
    raise TypeError(f"{name} is of type {type(query_object)} but must be {target_type_list}")


class matcher(object):

    def __init__(self, props):

        self.props = props

    def __call__(self, x, y):

        for prop in self.props:
            if x[prop] != y[prop]:
                return False
        return True

class atom_matcher(matcher):
    
    def __init__(self, props=None):

        if props == None:
            props = [
                "atomic_num", 
                "hybridization",
                #"chirality", 
                "is_aromatic"
                ]
        super().__init__(props)

class bond_matcher(matcher):

    def __init__(self, props=None):

        if props == None:
            props = ["bond_type"]
        super().__init__(props)

def rdmol_map(rdmol1, rdmol2):

    rdmol1_cp = copy.deepcopy(rdmol1)
    rdmol2_cp = copy.deepcopy(rdmol2)

    Chem.SanitizeMol(rdmol1_cp)
    Chem.SanitizeMol(rdmol2_cp)

    G1 = rdmol_to_nx(rdmol1_cp)
    G2 = rdmol_to_nx(rdmol2_cp)

    GM = isomorphism.GraphMatcher(G1, G2,
                                  node_match=atom_matcher,
                                  edge_match=bond_matcher)

    return GM.mapping.items()

def rdmol_to_nx(rdmol, node_features=None, edge_features=None):

    import networkx as nx

    if node_features == None:
        node_features = {
                   "atomic_num"    : Chem.rdchem.Atom.GetAtomicNum,
                   "hybridization" : Chem.rdchem.Atom.GetHybridization,
                   #"chirality"     : Chem.rdchem.Atom.GetChiralTag,
                   "is_aromatic"   : Chem.rdchem.Atom.GetIsAromatic
                   }

    if edge_features == None:
        edge_features = {
                   "bond_type" : Chem.rdchem.Bond.GetBondTypeAsDouble
                   }

    G = nx.Graph()
    for atm in rdmol.GetAtoms():
        node_dict = {name : func(atm) for name, func in node_features.items()}
        G.add_node(atm.GetIdx(), 
                    **node_dict
                    )

    for bnd in rdmol.GetBonds():
        atm1      = bnd.GetBeginAtom()
        atm2      = bnd.GetEndAtom()
        bond_dict = {name : func(bnd) for name, func in edge_features.items()}
        G.add_edge(atm1.GetIdx(),
                   atm2.GetIdx(),
                   **bond_dict
                   )

    return G
