"""CHGCAR file I/O for the SALTED arm.

A thin wrapper over pymatgen's ``Chgcar``. The wrapper adds two things
on top of the bare pymatgen API:

* ``n_electrons`` rescaling. The CHGCAR convention is
  ``integrated_density = sum(rho) * cell_volume / N_grid = N_electrons``.
  Our predicted densities come from an L2-projected basis with no
  guarantee on the integral; we have to rescale so VASP doesn't
  silently fix the electron count for us at startup (which would
  defeat the speedup measurement).

* ``ase.Atoms`` input/output to match the rest of the salted_ft
  pipeline. pymatgen's ``Structure`` is converted via
  ``AseAtomsAdaptor`` and back.

These two helpers are the boundary between the predicted-density
tensor world and the VASP-input file world. The actual SCF speedup
measurement lives in the entalsim ``StructureVASPSinglePoint`` maker
(separate stack).
"""

from __future__ import annotations

from pathlib import Path

import ase
import numpy as np


def write_chgcar(
    density: np.ndarray,
    atoms: ase.Atoms,
    path: str | Path,
    n_electrons: float | None = None,
) -> None:
    """Write a real-space density grid to a VASP CHGCAR file.

    Parameters
    ----------
    density :
        Real-space density on a regular grid, shape ``(Nx, Ny, Nz)``.
    atoms :
        Periodic structure; provides cell + species ordering.
    path :
        Output file path.
    n_electrons :
        If given (and > 0), rescale the density so the file's integrated
        density equals this value. VASP reads this as the total electron
        count when starting with ``ICHARG=1``; getting it right is
        what makes the SCF-speedup comparison meaningful.
    """
    if density.ndim != 3:
        raise ValueError(
            f"density must be a 3D grid (Nx, Ny, Nz); got shape {density.shape}"
        )
    if n_electrons is not None and n_electrons <= 0:
        raise ValueError(
            f"n_electrons must be > 0; got {n_electrons}. Use None to skip rescaling."
        )

    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.outputs import Chgcar

    structure = AseAtomsAdaptor.get_structure(atoms)
    rho = np.asarray(density, dtype=np.float64).copy()

    if n_electrons is not None:
        cell_volume = float(structure.lattice.volume)
        n_grid = int(np.prod(rho.shape))
        current_total = rho.sum() * cell_volume / n_grid
        if current_total != 0.0:
            rho *= n_electrons / current_total

    # pymatgen's Chgcar stores density as the per-cell sum (not per-grid-point);
    # i.e. rho_stored = rho * cell_volume in its convention. The Chgcar
    # constructor expects the data dict to use the same convention as VASP's
    # CHGCAR file format, which is rho * volume. We multiply here so the
    # round-trip via Chgcar.from_file preserves our user-facing rho.
    chgcar_data = {"total": rho * float(structure.lattice.volume)}
    chgcar = Chgcar(structure, chgcar_data)
    chgcar.write_file(str(path))


def read_chgcar(path: str | Path) -> tuple[np.ndarray, ase.Atoms]:
    """Read a CHGCAR file and return ``(density, atoms)``.

    Returns
    -------
    density : np.ndarray of shape ``(Nx, Ny, Nz)``, the density per
        grid point (the inverse of write_chgcar's convention).
    atoms : ase.Atoms
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.outputs import Chgcar

    chgcar = Chgcar.from_file(str(path))
    cell_volume = float(chgcar.structure.lattice.volume)
    # VASP stores density * volume; undo that for the user-facing density.
    rho = np.asarray(chgcar.data["total"], dtype=np.float64) / cell_volume
    atoms = AseAtomsAdaptor.get_atoms(chgcar.structure)
    return rho, atoms
