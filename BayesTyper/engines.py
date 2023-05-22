#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================
import copy
import numpy as np
import openmm
from openmm import unit
from openmm import vec3
from openmm import app

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================
from .constants import (_LENGTH,
                        _ANGLE,
                        _ENERGY_PER_MOL,
                        _FORCE,
                        _ATOMIC_MASS,
                        _FORCE_CONSTANT_BOND,
                        _FORCE_CONSTANT_ANGLE,
                        _FORCE_CONSTANT_TORSION)

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================


class OpenmmEngine(object):

    """
    Base class for all actions. This class basically just wraps around OpenMM.
    To work well with this class, it is a good idea to look at the OpenMM
    package at http://openmm.org/ .

    Examples:
    ---------
    system = OpenmmEngine()

    """

    def __init__(self, 
        openmm_system: openmm.openmm.System, 
        top: app.topology.Topology,
        xyz: np.ndarray,
        restraints_list: list = list(),
        platform_name: str = "CPU",
        platform_properties = {"Threads" : "1"}) -> None:

        if restraints_list:
            self.openmm_system = copy.deepcopy(openmm_system)
            self.top           = copy.deepcopy(top)
        else:
            self.openmm_system = openmm_system
            self.top           = top

        self.platform_name       = platform_name
        self.platform_properties = platform_properties

        self.N_atoms = self.openmm_system.getNumParticles()

        self.atomic_masses = np.zeros(self.N_atoms, dtype=float) * _ATOMIC_MASS
        for atm_idx in range(self.N_atoms):
            self.atomic_masses[atm_idx] = self.openmm_system.getParticleMass(atm_idx)

        ### Add restraint forces
        ### --------------------
        self._nonrestraint_force_group_set = -1
        if restraints_list:
            self._nonrestraint_force_group_set = set()
            for force in self.openmm_system.getForces():
                self._nonrestraint_force_group_set.add(force.getForceGroup())
            self._torsion_restraint_id_list  = list()

            ### Torsions
            r_force = openmm.CustomTorsionForce(
                "0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535"
                )
            r_force.addPerTorsionParameter("k")
            r_force.addPerTorsionParameter("theta0")
            r_force.setForceGroup(31)

            self._system_torsion_restraint_id = self.openmm_system.addForce(r_force)

        ### Initialize the simulation objects
        ### ---------------------------------
        self.create_openmm(xyz.in_units_of(unit.nanometers))

    ### Because sometimes I'm lazy and not consistent.
    def set_xyz(self, xyz):
        self.xyz = xyz

    def add_bond_restraint(
        self,
        atm_idx1,
        atm_idx2,
        k,
        eq_value):

        raise NotImplementedError

    def set_bond_restraint(
        self,
        restraint_id, 
        atm_idx1,
        atm_idx2,
        k, 
        eq_value):

        raise NotImplementedError

    def add_angle_restraint(
        self,
        atm_idx1,
        atm_idx2,
        atm_idx3,
        k,
        eq_value):

        raise NotImplementedError

    def set_angle_restraint(
        self,
        restraint_id, 
        atm_idx1,
        atm_idx2,
        atm_idx3,
        k, 
        eq_value):

        raise NotImplementedError

    def add_torsion_restraint(
        self,
        atm_idx1,
        atm_idx2,
        atm_idx3,
        atm_idx4,
        k,
        eq_value):

        r_force = self.openmm_system.getForce(self._system_torsion_restraint_id)
        r_force_idx = r_force.addTorsion(
            atm_idx1,
            atm_idx2,
            atm_idx3,
            atm_idx4,
            [
                k.in_units_of(_FORCE_CONSTANT_TORSION), 
                eq_value.in_units_of(_ANGLE)
            ],
            )
        self._torsion_restraint_id_list.append(r_force_idx)
        self.create_openmm(self.xyz)


    def set_torsion_restraint(
        self,
        restraint_id, 
        atm_idx1,
        atm_idx2,
        atm_idx3,
        atm_idx4,
        k, 
        eq_value):

        r_force     = self.openmm_system.getForce(self._system_torsion_restraint_id)
        r_force_idx = self._torsion_restraint_id_list[restraint_id]
        r_force.setTorsionParameters(
            r_force_idx,
            atm_idx1,
            atm_idx2,
            atm_idx3,
            atm_idx4,
            [
                k.in_units_of(_FORCE_CONSTANT_ANGLE), 
                eq_value.in_units_of(_ANGLE)
            ],
            )

        r_force.updateParametersInContext(self.openmm_simulation.context)
        self.update_state()


    @property
    def xyz(self):
        pos = self.openmm_state.getPositions(asNumpy=True).in_units_of(_LENGTH)
        return pos


    @xyz.setter
    def xyz(self, xyz):
        self.openmm_simulation.context.setPositions(xyz)
        self.update_state()


    @property
    def pot_ene(self):
        pot_ene = self.openmm_state.getPotentialEnergy().in_units_of(_ENERGY_PER_MOL)
        return pot_ene

    @property
    def forces(self):
        return self.openmm_state.getForces(asNumpy=True).in_units_of(_FORCE)

        ### The code below computes forces numerically. This actually
        ### does not make a difference, so we continue use the openmm
        ### forces directly.

        #diff = 1.0e-7 * _LENGTH
#
#        #### Adapted from ForceBalance routine in openmmio.py
#        #### https://github.com/leeping/forcebalance/blob/master/src/openmmio.py
#        #grad = np.zeros((self.N_atoms, 3), dtype=float) * _ENERGY_PER_MOL / _LENGTH
#        #coef = 1.0 / (diff * 2.) # 1/2 step width
#        #xyz  = copy.copy(self.xyz.in_units_of(_LENGTH))
#
#        #for atm_idx in range(self.N_atoms):
#        #    for crd_idx in range(3):
#        #        xyz[atm_idx][crd_idx] += diff
#        #        self.xyz               = xyz
#        #        grad_plus              = np.copy(self.pot_ene.in_units_of(_ENERGY_PER_MOL))
#
#        #        xyz[atm_idx][crd_idx] -= 2*diff
#        #        self.xyz               = xyz
#        #        grad_minus             = np.copy(self.pot_ene.in_units_of(_ENERGY_PER_MOL))
#
#        #        xyz[atm_idx][crd_idx] += diff
#        #        self.xyz               = xyz
#
#        #        grad[atm_idx,crd_idx]  = (grad_plus - grad_minus) * coef
#
#        #### Convert to forces
#        #grad *= -1.
#
        #return grad

    def create_openmm(self, xyz):

        self.openmm_integrator = openmm.LangevinIntegrator(
            300.,
            1.0/unit.picoseconds,
            2.0*unit.femtoseconds)
        self.openmm_integrator.setConstraintTolerance(0.00001)
        self.openmm_platform   = openmm.Platform.getPlatformByName(
            self.platform_name
            )
        self.openmm_simulation = openmm.app.Simulation(
            self.top,
            self.openmm_system,
            self.openmm_integrator,
            self.openmm_platform,
            self.platform_properties)
        self.xyz = xyz

        self.update_state()

    def update_state(self):

        self.openmm_state = self.openmm_simulation.context.getState(
            getPositions=True,
            getEnergy=True,
            getForces=True,
            groups=self._nonrestraint_force_group_set
            )

        ### Uncomment this to get the full system energy *including*
        ### the harmonic restraint terms.
#        self.openmm_state = self.openmm_simulation.context.getState(getPositions=True,
#                                                                    getEnergy=True,
#                                                                    getForces=True)

    def minimize(
        self, 
        crit=1e-2 * _ENERGY_PER_MOL):

        success = True

        crit  = crit.in_units_of(unit.kilojoule * unit.mole**-1)
        steps = int(max(1, -1 * np.log10(crit._value)))

        ### E1 and E2 are just for testing purpose
        #E1 = self.pot_ene

        #self.openmm_simulation.minimizeEnergy()
        #self.update_state()
        #return success

        try:
            ### This is mainly from forcebalance code.
            ### See https://github.com/leeping/forcebalance/blob/128f9e50f87ac3f5162ffd490ee63c975758f781/src/openmmio.py
            for logc in np.linspace(0, np.log10(crit._value), steps):
                tol = 10**logc * unit.kilojoule * unit.mole**-1
                ### Note: The forcebalance implementation uses maxIterations=100000
                ###       here. However, that makes the overall performance *very* slow.
                self.openmm_simulation.minimizeEnergy(
                    tolerance=tol, 
                    maxIterations=1000
                    )
            self.update_state()
            ene_min_old = self.pot_ene
            for _ in range(1000):
                self.openmm_simulation.minimizeEnergy(
                    tolerance=crit, 
                    maxIterations=10
                    )
                self.update_state()
                if abs(self.pot_ene - ene_min_old) < (crit * 10.):
                    break
                ene_min_old = self.pot_ene
                #E2 = self.pot_ene
                #print("E1:", E1, "E2:", E2)
        except:
            success = False

        return success

    def compute_hessian(self, diff=1.0e-4 * _LENGTH):

        FORCE_UNIT  = unit.joule * unit.nanometer**-1 * unit.mole**-1
        LENGTH_UNIT = unit.nanometer

        ### Default diff taken from Parsley paper:
        ### doi.org/10.26434/chemrxiv.13082561.v2
        diff = diff.in_units_of(LENGTH_UNIT)

        ### Adapted from ForceBalance routine in openmmio.py
        ### https://github.com/leeping/forcebalance/blob/master/src/openmmio.py
        hessian = np.zeros((self.N_atoms*3,
                            self.N_atoms*3), dtype=float)
        coef    = 1.0 / (diff._value * 2) # 1/2 step width
        xyz     = copy.copy(self.xyz.in_units_of(LENGTH_UNIT))

        for atm_idx in range(self.N_atoms):
            for crd_idx in range(3):
                xyz[atm_idx][crd_idx] += diff
                self.xyz               = xyz
                grad_plus              = -np.copy(self.forces.value_in_unit(FORCE_UNIT))

                xyz[atm_idx][crd_idx] -= 2*diff
                self.xyz               = xyz
                grad_minus             = -np.copy(self.forces.value_in_unit(FORCE_UNIT))

                xyz[atm_idx][crd_idx] += diff
                self.xyz               = xyz

                hessian[atm_idx*3+crd_idx] = np.ravel((grad_plus - grad_minus) * coef)

        ### make hessian symmetric by averaging upper right and lower left
        hessian += hessian.T
        hessian *= 0.5

        hessian *= FORCE_UNIT * LENGTH_UNIT**-1
        hessian  = hessian.in_units_of(_FORCE * _LENGTH**-1)

        return hessian