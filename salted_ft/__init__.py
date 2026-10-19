"""SALTED-arm basis-expansion infrastructure for the r2SCAN benchmark.

This package wraps rholearn (`lab-cosmo/rholearn`) and provides the
projection/reconstruction bridge between LeMat-Rho VASP CHGCAR data
and the rholearn training/inference pipeline.

Layout (stacked PRs, see `plan_salted_graph2mat_basis_choice_may_20_pm.md`):

* ``basis.py`` (PR α)  — ``BasisSpec`` dataclass + shape helpers.
* ``projection.py`` (PR β) — VASP CHGCAR ↔ basis coefficients.
* ``model.py`` (PR γ) — ``SALTEDModel`` wrapper for rholearn.
* ``io.py`` (PR δ) — coefficients/grid ↔ pymatgen ``Chgcar``.
"""

from salted_ft.basis import BasisSpec

__all__ = ["BasisSpec"]
