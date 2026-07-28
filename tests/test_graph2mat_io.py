"""TDD tests for the Graph2Mat IO surface (PR zeta-delta).

graph2mat_ft.io should expose the same read_chgcar / write_chgcar
helpers as salted_ft.io, sharing a single implementation (no
duplicate code). These tests pin that the re-exports are the
identical callable, so a fix in salted_ft.io automatically
propagates to the Graph2Mat arm.
"""

from __future__ import annotations


def test_read_chgcar_is_reexport():
    from graph2mat_ft.io import read_chgcar as g2m_read
    from salted_ft.io import read_chgcar as salted_read

    assert g2m_read is salted_read


def test_write_chgcar_is_reexport():
    from graph2mat_ft.io import write_chgcar as g2m_write
    from salted_ft.io import write_chgcar as salted_write

    assert g2m_write is salted_write
