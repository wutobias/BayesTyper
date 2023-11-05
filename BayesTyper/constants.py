from openmm import unit
import numpy as np

_FORCE_TAGS     = ['Constraints',
                   'Bonds',
                   'Angles',
                   'ProperTorsions',
                   'ImproperTorsions',
                   'GBSA',
                   'vdW',
                   'Electrostatics',
                   'LibraryCharges',
                   'ToolkitAM1BCC',
                   'ChargeIncrementModel']

_DEFAULT_FF = "openff_unconstrained-1.3.0.offxml"

### Some standard units that control the output
### of every routine.
_LENGTH         = unit.nanometer
_ANGLE_DEG      = unit.degree
_ANGLE_RAD      = unit.radian
_ANGLE          = _ANGLE_DEG
_ENERGY         = unit.kilojoule
_ENERGY_PER_MOL = _ENERGY * unit.mole**-1
_FORCE          = _ENERGY_PER_MOL * _LENGTH**-1
_FORCE_CONSTANT_BOND    = _ENERGY_PER_MOL * _LENGTH**-2
_FORCE_CONSTANT_ANGLE   = _ENERGY_PER_MOL * _ANGLE_RAD**-2
_FORCE_CONSTANT_TORSION = _ENERGY_PER_MOL
_WAVENUMBER     = unit.centimeters**-1
_ATOMIC_MASS    = unit.dalton

### Some automic unit units
_LENGTH_AU      = unit.bohr
_ENERGY_AU      = unit.hartree
_FORCE_AU       = _ENERGY_AU * _LENGTH_AU**-1

### Some unit conversion factors w/o "unit units"
_BOHR_TO_NM       = 0.0529177
_NM_TO_BOHR       = 1./0.0529177
_BOHR_TO_ANGSTROM = 0.529177
_ANGSTROM_TO_BOHR = 1./0.529177


### Misc constants
_SEED            = 42
_UNIT_QUANTITY   = unit.quantity.Quantity
_INACTIVE_GROUP_IDX = -99
_TIMEOUT         = 20 # timeout in seconds
_VERBOSE         = False