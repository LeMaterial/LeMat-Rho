"""CHGCAR file I/O for the Graph2Mat arm.

The Graph2Mat arm uses the same on-disk format as the SALTED arm
(VASP CHGCAR + pymatgen). To avoid drift between the two arms, the
canonical implementation lives in ``salted_ft.io`` and this module
re-exports it.

Downstream code that wants the Graph2Mat namespace
(``from graph2mat_ft.io import read_chgcar, write_chgcar``) gets
the same helpers as the SALTED arm, including the
``n_electrons`` rescaling that ICHARG=1 needs.
"""

from __future__ import annotations

from salted_ft.io import read_chgcar, write_chgcar

__all__ = ["read_chgcar", "write_chgcar"]
