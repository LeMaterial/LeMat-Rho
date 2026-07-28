"""PBC-aware neighbor graph for the BOA transform pipeline.

BOA's stock edge builder (``boa.data.transforms.AddRadiusEdgeIndex``, which
wraps ``mldft``'s ``AddRadiusEdgeIndex``) calls
``torch_geometric.nn.radius_graph``. That is open-boundary (it ignores the
cell) and it drags in the compiled ``torch_cluster`` extension, which has no
ROCm/AMD wheel. LeMat-Rho holds periodic solids, so we replace it with a
minimum-image radius graph built from the lattice directly.

We keep exactly one edge per ordered atom pair (plus self loops), matching the
output structure of ``radius_graph(..., loop=True)``. BOA's density decoder
(``ChgLightningModule.orbital_inference``) matches every edge ``(i, j)`` with
its flip ``(j, i)`` by finding unique columns, so duplicating a pair across
several periodic images would break that bookkeeping. The periodic summation of
the density itself is still handled downstream inside ``GTOs.forward`` when
``pbc=True``; this graph only decides which atom pairs carry coefficients and
exchange messages.

.. code-block:: python

    import torch
    from boa_ft.transforms import pbc_radius_edge_index

    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.9]])
    cell = torch.eye(3) * 4.0
    edge_index = pbc_radius_edge_index(pos, cell, radius=1.0)
"""

from __future__ import annotations

import torch


def _cell_heights(cell: torch.Tensor) -> torch.Tensor:
    """Perpendicular distance between opposite faces for each lattice vector.

    Used to decide how many periodic images we must generate so that no
    neighbor within ``radius`` is missed.

    Parameters
    ----------
    cell : torch.Tensor
        Lattice matrix of shape ``(3, 3)`` with lattice vectors as rows.

    Returns
    -------
    torch.Tensor
        Shape ``(3,)`` interplanar spacings.
    """
    volume = torch.det(cell).abs()
    heights = []
    for i in range(3):
        # The two lattice vectors spanning the face opposite to vector i.
        j, k = [d for d in range(3) if d != i]
        face_area = torch.linalg.cross(cell[j], cell[k]).norm()
        heights.append(volume / (face_area + 1e-12))
    return torch.stack(heights)


def pbc_radius_edge_index(
    pos: torch.Tensor,
    cell: torch.Tensor,
    radius: float,
    loop: bool = True,
) -> torch.Tensor:
    """Minimum-image radius graph under periodic boundary conditions.

    An ordered pair ``(i, j)`` is connected when the closest periodic image of
    atom ``j`` lies within ``radius`` of atom ``i``. Self loops ``(i, i)`` are
    included when ``loop`` is True (their minimum-image distance is 0).

    Parameters
    ----------
    pos : torch.Tensor
        Cartesian atom coordinates, shape ``(N, 3)`` (Angstrom).
    cell : torch.Tensor
        Lattice matrix, shape ``(3, 3)`` with lattice vectors as rows (Angstrom).
    radius : float
        Neighbor cutoff (Angstrom).
    loop : bool
        Whether to include self loops. Defaults to True to match
        ``radius_graph(loop=True)``.

    Returns
    -------
    torch.Tensor
        Edge index of shape ``(2, E)`` and dtype long. Row 0 is the source,
        row 1 the destination. The graph is symmetric.
    """
    # float64 keeps distance comparisons stable for near-cutoff neighbors.
    pos = pos.to(torch.float64)
    cell = cell.to(torch.float64)

    # Enough images along each axis to cover the cutoff even for skewed cells.
    heights = _cell_heights(cell)
    n_rep = torch.ceil(torch.as_tensor(radius) / heights).long()
    axis_ranges = [torch.arange(-int(n_rep[d]), int(n_rep[d]) + 1) for d in range(3)]
    # Integer image offsets (S, 3) mapped to Cartesian shift vectors (S, 3).
    offsets = torch.cartesian_prod(*axis_ranges).to(torch.float64)
    shifts = offsets @ cell

    # dvec[i, j, s] = (pos[j] + shift[s]) - pos[i]; take the closest image per pair.
    dvec = pos[None, :, None, :] + shifts[None, None, :, :] - pos[:, None, None, :]
    dist_min = dvec.norm(dim=-1).min(dim=-1).values  # (N, N)

    mask = dist_min <= radius
    mask.fill_diagonal_(bool(loop))

    src, dst = torch.nonzero(mask, as_tuple=True)
    return torch.stack([src, dst], dim=0).long()


class PBCRadiusEdgeIndex:
    """Drop-in periodic replacement for ``boa.data.transforms.AddRadiusEdgeIndex``.

    Mirrors that class's interface (``radius`` and ``name``) so it slots into the
    BOA transform config unchanged, but builds a minimum-image graph from
    ``sample.cell`` instead of an open-boundary ``radius_graph``.

    .. code-block:: yaml

        - _target_: boa_ft.transforms.PBCRadiusEdgeIndex
          radius: 3.0
          name: edge_index
    """

    def __init__(self, radius: float, name: str = "edge_index"):
        """
        Parameters
        ----------
        radius : float
            Neighbor cutoff (Angstrom).
        name : str
            Attribute name to store the edge index under. Defaults to
            ``"edge_index"``.
        """
        self.radius = radius
        self.name = name

    def __call__(self, sample):
        """Add the periodic edge index to ``sample`` under ``self.name``.

        Parameters
        ----------
        sample : mldft.ml.data.components.of_data.OFData
            Sample carrying ``pos`` (N, 3) and ``cell`` ((1, 3, 3) or (3, 3)).

        Returns
        -------
        mldft.ml.data.components.of_data.OFData
            The same sample with the edge index attribute added.
        """
        # Imported here so the pure geometry helpers above stay importable
        # (and unit-testable) without the mldft package installed.
        from mldft.ml.data.components.of_data import Representation

        cell = torch.as_tensor(sample.cell)
        # ConvertToOFData stores the cell as (1, 3, 3); collapse the batch dim.
        if cell.dim() == 3:
            cell = cell[0]
        edge_index = pbc_radius_edge_index(
            torch.as_tensor(sample.pos), cell, self.radius
        )
        sample.add_item(self.name, edge_index, Representation.NONE)
        return sample
