"""BasisSpec — the atom-centered radial × angular basis used by the SALTED arm.

The density expansion is
::

    rho(r) = sum_i  sum_{nlm}  c_{i,nlm}  phi_{n}(|r - r_i|)  Y_{lm}(r - r_i)

with ``phi_n`` a Gaussian radial of width ``sigma_n`` and ``Y_lm`` a real
spherical harmonic.

Numbers locked in Phase A4 of
``plan_salted_graph2mat_basis_choice_may_20_pm.md`` (2026-05-20):
``max_l=4``, ``n_radial=4``, ``sigma=(0.5, 1.0, 2.0, 4.0)``, ``cutoff=4.0``.
That gives 100 coefficients per atom (4 × (4+1)²), which lands the trained
model in the same parameter-count ballpark as ChargE3Net for fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BasisSpec:
    """Configuration of the atom-centered Gaussian × Y_lm basis.

    Parameters
    ----------
    max_l :
        Maximum angular momentum, inclusive. Real spherical harmonics
        Y_lm with ``l = 0..max_l`` and ``m = -l..l`` are used.
    n_radial :
        Number of radial channels. Must match ``len(sigma)``.
    sigma :
        Gaussian widths (Angstrom), one per radial channel.
    cutoff :
        Radial cutoff (Angstrom) beyond which basis functions are zero.
        Should match the cutoff used by the neighbor-list / graph
        constructor of the downstream ML model.
    """

    max_l: int = 4
    n_radial: int = 4
    sigma: tuple[float, ...] = field(default=(0.5, 1.0, 2.0, 4.0))
    cutoff: float = 4.0

    def __post_init__(self) -> None:
        # All validation goes here so a malformed spec raises at construction
        # time, not deep inside a tensor op three PRs from now.
        if self.max_l < 0:
            raise ValueError(
                f"max_l must be >= 0; got {self.max_l}. "
                "Use max_l=0 for an s-only basis."
            )
        if self.n_radial < 1:
            raise ValueError(
                f"n_radial must be >= 1; got {self.n_radial}. "
                "A basis with zero radial channels has no expressive power."
            )
        if len(self.sigma) != self.n_radial:
            raise ValueError(
                f"n_radial ({self.n_radial}) must equal len(sigma) "
                f"({len(self.sigma)}); each radial channel needs its own width."
            )
        if any(s <= 0 for s in self.sigma):
            raise ValueError(
                f"sigma values must be positive (Gaussian widths); got {self.sigma}."
            )
        if self.cutoff <= 0:
            raise ValueError(
                f"cutoff must be > 0; got {self.cutoff}. "
                "A nonpositive cutoff makes the basis identically zero."
            )

    @property
    def n_angular_components(self) -> int:
        """Number of real-Ylm components for l = 0..max_l: sum_l (2l + 1) = (max_l + 1)^2."""
        return (self.max_l + 1) ** 2

    @property
    def n_coeffs_per_atom(self) -> int:
        """Coefficients per atom: n_radial channels × angular components."""
        return self.n_radial * self.n_angular_components

    def total_coeffs_shape(self, n_atoms: int) -> tuple[int, int]:
        """Shape of the per-structure coefficients tensor."""
        return (n_atoms, self.n_coeffs_per_atom)
