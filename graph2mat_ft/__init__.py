"""Graph2Mat-arm infrastructure for the r2SCAN benchmark (PARKED).

PARKED 2026-05-25. Reasoning (see
``../plan_graph2mat_parked_2026-05-25.md``):

Graph2Mat's native target is a per-pair atom-centered density
matrix ``D_ab``. VASP outputs only a grid density (not D_ab in any
localized basis), so training Graph2Mat on VASP r2SCAN would
require inventing a CHGCAR -> D_ab projection. Standard LSQR on
that is a 10^6 x 10^6 dense linear system per structure; the
matrix-free + neighbor-cutoff variant is multi-week research-grade
engineering with its own quality ceiling to validate.

For the LeMat-Rho 3-arm comparison (ChargE3Net, DeepDFT, SALTED),
Graph2Mat is parked. The code below is correct as scaffolding and
ships with green tests; it can be revived if (1) we switch the
training set to a code that natively outputs D_ab (SIESTA, ...) or
(2) someone invests in the matrix-free projection.

The basis adapter (PointBasis) and IO re-export are still useful
in their own right; left in place.
"""

from graph2mat_ft.basis import basis_table_for_species, point_basis_for_species
from graph2mat_ft.io import read_chgcar, write_chgcar
from graph2mat_ft.model import Graph2MatModel
from graph2mat_ft.projection import (
    make_basis_configuration,
    pack_coeffs_to_point_labels,
    unpack_point_labels_to_coeffs,
)

__all__ = [
    "Graph2MatModel",
    "basis_table_for_species",
    "make_basis_configuration",
    "pack_coeffs_to_point_labels",
    "point_basis_for_species",
    "read_chgcar",
    "unpack_point_labels_to_coeffs",
    "write_chgcar",
]
