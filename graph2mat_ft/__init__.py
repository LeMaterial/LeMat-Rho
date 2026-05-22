"""Graph2Mat-arm infrastructure for the r2SCAN benchmark.

Parallel to ``salted_ft`` but targeting Graph2Mat
(``BIG-MAP/graph2mat``). Stacked PR layout (mirror of SALTED):

* ``basis.py`` (PR zeta-alpha) -- ``BasisSpec`` -> ``PointBasis``
* ``projection.py`` (PR zeta-beta) -- density grid <-> density matrix
* ``model.py`` (PR zeta-gamma) -- ``Graph2MatModel`` wrapper
* ``io.py`` (PR zeta-delta) -- shared CHGCAR I/O (probably reuses
   ``salted_ft.io``)

The basis we project onto, the comparison metric (NMAPE/RMSE/NRMSE)
and the CHGCAR I/O are shared with the SALTED arm so the two models
land in the same comparison table.
"""

from graph2mat_ft.basis import basis_table_for_species, point_basis_for_species

__all__ = ["basis_table_for_species", "point_basis_for_species"]
