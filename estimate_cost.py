import json
import bz2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from typing import Tuple

MatPESStaticSet_KSPACING: float = 0.22

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


def get_cost_info_from_structure(structure: Structure) -> Tuple[int, Tuple[int, int, int], int, int, float]:
    """
    Estimate computational cost parameters from a pymatgen Structure object.

    Args:
        structure (Structure): A pymatgen Structure object.

    Returns:
        tuple:
            - Nbands (int): Estimated number of bands.
            - Kpoints (tuple[int, int, int]): K-point grid dimensions.
            - spg (int): Space group number.
            - N_irreducible_kpts (int): Estimated number of irreducible k-points.
            - cost_estimate (float): Heuristic estimate of computational cost.
    """
    from pymatgen.io.vasp.sets import MatPESStaticSet

    inputset = MatPESStaticSet(structure)
    Nbands = inputset.estimate_nbands()
    Kpoints = get_kpoint_grid_from_kspacing(structure, MatPESStaticSet_KSPACING)
    spg = structure.get_space_group_info()[1]
    N_irreducible_kpts = estimate_irreducible_kpoints(spg, Kpoints)
    cost_estimate = Nbands**5 * N_irreducible_kpts

    return Nbands, Kpoints, spg, N_irreducible_kpts, cost_estimate
