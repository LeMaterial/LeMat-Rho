import json
import bz2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from typing import Tuple, Optional

MatPESStaticSet_KSPACING: float = 0.22
#Nelect values for MATPES POTCARs
NELECT = {'Ac': 11.0, 'Ag': 11.0, 'Al': 3.0, 'Am': 17.0, 'Ar': 8.0, 'As': 5.0, 'At': 7.0, 'Au': 11.0, 'B': 3.0, 'Ba': 10.0, 'Be': 4.0, 'Bi': 5.0, 'Br': 7.0, 'C': 4.0, 'Ca': 10.0, 'Cd': 12.0, 'Ce': 12.0, 'Cf': 20.0, 'Cl': 7.0, 'Cm': 18.0, 'Co': 9.0, 'Cr': 12.0, 'Cs': 9.0, 'Cu': 17.0, 'Dy': 20.0, 'Er': 22.0, 'Eu': 17.0, 'F': 7.0, 'Fe': 14.0, 'Fr': 9.0, 'Ga': 13.0, 'Gd': 18.0, 'Ge': 14.0, 'H': 1.0, 'He': 2.0, 'Hf': 10.0, 'Hg': 12.0, 'Ho': 21.0, 'I': 7.0, 'In': 13.0, 'Ir': 9.0, 'K': 9.0, 'Kr': 8.0, 'La': 11.0, 'Li': 3.0, 'Lu': 9.0, 'Mg': 8.0, 'Mn': 13.0, 'Mo': 12.0, 'N': 5.0, 'Na': 7.0, 'Nb': 11.0, 'Nd': 14.0, 'Ne': 8.0, 'Ni': 16.0, 'Np': 15.0, 'O': 6.0, 'Os': 14.0, 'P': 5.0, 'Pa': 13.0, 'Pb': 14.0, 'Pd': 10.0, 'Pm': 15.0, 'Po': 16.0, 'Pr': 13.0, 'Pt': 10.0, 'Pu': 16.0, 'Ra': 10.0, 'Rb': 9.0, 'Re': 13.0, 'Rh': 15.0, 'Rn': 8.0, 'Ru': 14.0, 'S': 6.0, 'Sb': 5.0, 'Sc': 11.0, 'Se': 6.0, 'Si': 4.0, 'Sm': 16.0, 'Sn': 14.0, 'Sr': 10.0, 'Ta': 11.0, 'Tb': 19.0, 'Tc': 13.0, 'Te': 6.0, 'Th': 12.0, 'Ti': 10.0, 'Tl': 13.0, 'Tm': 23.0, 'U': 14.0, 'V': 11.0, 'W': 14.0, 'Xe': 8.0, 'Y': 11.0, 'Yb': 24.0, 'Zn': 12.0, 'Zr': 12.0}
MAGMOMS = {
    "Ce": 5,
    "Ce3+": 1,
    "Co": 0.6,
    "Co3+": 0.6,
    "Co4+": 1,
    "Cr": 5,
    "Dy3+": 5,
    "Er3+": 3,
    "Eu": 10,
    "Eu2+": 7,
    "Eu3+": 6,
    "Fe": 5,
    "Gd3+": 7,
    "Ho3+": 4,
    "La3+": 0.6,
    "Lu3+": 0.6,
    "Mn": 5,
    "Mn3+": 4,
    "Mn4+": 3,
    "Mo": 5,
    "Nd3+": 3,
    "Ni": 5,
    "Pm3+": 4,
    "Pr3+": 2,
    "Sm3+": 5,
    "Tb3+": 6,
    "Tm3+": 2,
    "V": 5,
    "W": 5,
    "Yb3+": 1
}

def estimate_nbands(composition, lncl=False, npar=None, magmoms_override=None):
    """
    Estimate NBANDS for a given composition dictionary.
    
    Args:
        composition (dict): e.g., {"Fe": 2, "O": 3}
        lncl (bool): If LNONCOLLINEAR should be considered (default False)
        npar (int): Value of NPAR (default None, means not set)
        magmoms_override (dict): e.g., {"Fe": 4.0, "O": 0.0}, overrides defaults.
        
    Returns:
        int: Estimated NBANDS
    """
    # Total number of ions (atoms)
    n_ions = sum(composition.values())
    
    # Compute total NELECT
    nelect = sum(NELECT[el] * amt for el, amt in composition.items())
    
    # Compute total MAGMOM
    magmoms = []
    for el, amt_f in composition.items():
        # If override provided, use it; otherwise use MAGMOMS dict, else 0
        amt = int(amt_f)
        if abs(amt-amt_f)>1e-7:
            raise ValueError(f"Composition value for {el} is not an integer: {amt_f}")
        mm = 0
        if magmoms_override and el in magmoms_override:
            mm = magmoms_override[el]
        else:
            mm = MAGMOMS.get(el, 0.6)
        magmoms += [mm] * amt
    n_mag = sum(magmoms)
    n_mag = np.floor((n_mag + 1) / 2)
    
    # Possible NBANDS from the formula
    possible_val_1 = np.floor((nelect + 2) / 2) + max(np.floor(n_ions / 2), 3)
    possible_val_2 = np.floor(nelect * 0.6)

    n_bands = max(possible_val_1, possible_val_2) + n_mag
    
    # Noncollinear spins
    if lncl:
        n_bands *= 2
    
    # NPAR correction
    if npar:
        n_bands = (np.floor((n_bands + npar - 1) / npar)) * npar
    
    return int(n_bands)

def get_kpoint_grid_from_kspacing(structure: Structure, kspacing: float = MatPESStaticSet_KSPACING) -> Tuple[int, int, int]:
    """
    Compute the k-point grid dimensions (Nkx, Nky, Nkz) from the desired KSPACING (Å⁻¹).

    Args:
        structure (Structure): A pymatgen Structure object.
        kspacing (float): Desired k-point spacing in Å⁻¹. Default is 0.22 Å⁻¹.

    Returns:
        tuple[int, int, int]: The k-point grid dimensions (Nkx, Nky, Nkz).
    """
    rec_lattice = structure.lattice.reciprocal_lattice
    b_vectors = rec_lattice.matrix  # reciprocal lattice vectors in Å⁻¹

    Nk = []
    for bi in b_vectors:
        length_bi = np.linalg.norm(bi)
        Ni = max(1, int(np.ceil(length_bi / kspacing))) 
        Nk.append(Ni)

    return tuple(Nk)


def get_kpoint_grid_from_kspacing_lattice(lattice: np.array, kspacing: float = MatPESStaticSet_KSPACING) -> Tuple[int, int, int]:
    """
    Compute the k-point grid dimensions (Nkx, Nky, Nkz) from the desired KSPACING (Å⁻¹).

    Args:
        lattice (np.array): A 3x3 numpy array representing the lattice vectors in direct space (in Å).
        kspacing (float): Desired k-point spacing in Å⁻¹. Default is 0.22 Å⁻¹.

    Returns:
        tuple[int, int, int]: The k-point grid dimensions (Nkx, Nky, Nkz).
    """
    b_vectors = np.linalg.inv(lattice).T  # Convert lattice to reciprocal space

    Nk = []
    for bi in b_vectors:
        length_bi = np.linalg.norm(bi)
        Ni = max(1, int(np.ceil(length_bi / kspacing * 2 * np.pi)))  # Convert to reciprocal space units
        Nk.append(Ni)

    return tuple(Nk)


def estimate_irreducible_kpoints(spacegroup_number: int, kpts: Tuple[int, int, int]) -> int:
    """
    Estimate the number of irreducible k-points based on the space group number.

    Args:
        spacegroup_number (int): Space group number (1–230).
        kpts (tuple[int, int, int]): The full k-point grid dimensions.

    Returns:
        int: Estimated number of irreducible k-points.
    """
    if spacegroup_number >= 195:
        irreducible_fraction = 0.08  # Cubic
    elif spacegroup_number >= 168:
        irreducible_fraction = 0.12  # Hexagonal
    elif spacegroup_number >= 75:
        irreducible_fraction = 0.25  # Tetragonal
    elif spacegroup_number >= 16:
        irreducible_fraction = 0.30  # Orthorhombic
    elif spacegroup_number >= 3:
        irreducible_fraction = 0.40  # Monoclinic
    else:
        irreducible_fraction = 0.50  # Triclinic

    total_kpts = np.prod(kpts)
    N_irreducible_kpts = int(np.ceil(total_kpts * irreducible_fraction))
    return N_irreducible_kpts


def get_cost_info_from_structure(composition: dict, lattice: np.array, spg: int) -> Tuple[int, Tuple[int, int, int], int, int, float]:
    """
    Estimate computational cost parameters from a pymatgen Structure object.

    Args:
        composition (dict): Composition dictionary, e.g., {"Fe": 2, "O": 3}.
        lattice (np.array): A 3x3 numpy array representing the lattice vectors in direct space (in Å).
        spg (Optional[int]): Space group number (default None).

    Returns:
        tuple:
            - Nbands (int): Estimated number of bands.
            - Kpoints (tuple[int, int, int]): K-point grid dimensions.
            - spg (int): Space group number.
            - N_irreducible_kpts (int): Estimated number of irreducible k-points.
            - cost_estimate (float): Heuristic estimate of computational cost.
    """
    Nbands = estimate_nbands(composition, lncl=False)
    Kpoints = get_kpoint_grid_from_kspacing_lattice(lattice, MatPESStaticSet_KSPACING)
    N_irreducible_kpts = estimate_irreducible_kpoints(spg, Kpoints)
    cost_estimate = Nbands**5 * N_irreducible_kpts

    return Nbands, Kpoints, spg, N_irreducible_kpts, cost_estimate
