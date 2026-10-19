"""Tests for the boa_ft periodic neighbor graph.

The BOA density decoder matches every edge with its flip, so the graph must
contain exactly one edge per ordered pair (plus self loops), and it must respect
periodic boundary conditions. These cases are hand-checked against a small cubic
cell so the minimum-image logic is pinned independent of the mldft package.
"""

from __future__ import annotations

import torch

from boa_ft.transforms import pbc_radius_edge_index


def _edge_set(edge_index: torch.Tensor) -> set:
    """Return the set of (src, dst) tuples in an edge index."""
    return {(int(s), int(d)) for s, d in edge_index.t().tolist()}


class TestPBCRadiusEdgeIndex:
    """Minimum-image radius graph on a 4 A cubic cell."""

    def test_periodic_neighbor_across_boundary(self):
        """Atoms 0.1 A apart across the cell face must be connected.

        Direct distance is 3.9 A, but the nearest periodic image is 0.1 A away
        (3.9 - 4.0). With radius 1.0 A the cross edges must appear; an
        open-boundary graph would miss them.
        """
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.9]])
        cell = torch.eye(3) * 4.0
        edges = _edge_set(pbc_radius_edge_index(pos, cell, radius=1.0))
        assert edges == {(0, 0), (1, 1), (0, 1), (1, 0)}

    def test_no_edge_when_min_image_beyond_cutoff(self):
        """With radius 0.05 A the 0.1 A cross distance is out of range."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.9]])
        cell = torch.eye(3) * 4.0
        edges = _edge_set(pbc_radius_edge_index(pos, cell, radius=0.05))
        assert edges == {(0, 0), (1, 1)}

    def test_loop_false_drops_self_edges(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.9]])
        cell = torch.eye(3) * 4.0
        edges = _edge_set(pbc_radius_edge_index(pos, cell, radius=1.0, loop=False))
        assert edges == {(0, 1), (1, 0)}

    def test_single_atom_has_one_self_loop_only(self):
        """Periodic images of a lone atom (4 A away) must not add extra self edges.

        Duplicate (0, 0) columns would break the decoder's flip matching, so a
        single atom yields exactly one self loop even though its own images sit
        within a larger cutoff.
        """
        pos = torch.tensor([[1.0, 1.0, 1.0]])
        cell = torch.eye(3) * 4.0
        edge_index = pbc_radius_edge_index(pos, cell, radius=5.0)
        assert edge_index.shape == (2, 1)
        assert _edge_set(edge_index) == {(0, 0)}

    def test_edge_set_is_symmetric(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        cell = torch.eye(3) * 10.0
        edges = _edge_set(pbc_radius_edge_index(pos, cell, radius=1.5))
        for src, dst in edges:
            assert (dst, src) in edges

    def test_returns_long_2xe(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cell = torch.eye(3) * 10.0
        edge_index = pbc_radius_edge_index(pos, cell, radius=1.5)
        assert edge_index.dtype == torch.long
        assert edge_index.shape[0] == 2
