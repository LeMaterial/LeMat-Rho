"""TDD tests for the SALTED-arm BasisSpec dataclass.

Locks down the basis numbers chosen in
``plan_salted_graph2mat_basis_choice_may_20_pm.md`` (Phase A4):

* ``max_l = 4``
* ``n_radial = 4`` (uniform across species in v1)
* ``sigma = (0.5, 1.0, 2.0, 4.0)`` Å — geometric radial-width ladder
* ``cutoff = 4.0`` Å — matches ChargE3Net's KdTree cutoff
* ``n_coeffs_per_atom == n_radial * (max_l + 1) ** 2`` == 100

These numbers are referenced by every downstream PR (projection,
reconstruction, model wrapper, VASP I/O). Pinning them here means a
later edit shows up as a single failing test, not a silent drift.
"""

from __future__ import annotations

import pytest


class TestBasisSpecDefaults:
    """Default BasisSpec must match the A4 lockdown."""

    def test_default_max_l_is_four(self):
        from salted_ft.basis import BasisSpec

        assert BasisSpec().max_l == 4

    def test_default_n_radial_is_four(self):
        from salted_ft.basis import BasisSpec

        assert BasisSpec().n_radial == 4

    def test_default_sigma_ladder(self):
        from salted_ft.basis import BasisSpec

        # Geometric ladder over tight + valence + diffuse regimes.
        assert BasisSpec().sigma == (0.5, 1.0, 2.0, 4.0)

    def test_default_cutoff_matches_charge3net(self):
        from salted_ft.basis import BasisSpec

        # ChargE3Net's KdTreeGraphConstructor uses cutoff=4.0; the SALTED-arm
        # uses the same so atom-neighbor structure is identical between models.
        assert BasisSpec().cutoff == pytest.approx(4.0)

    def test_default_n_coeffs_per_atom_is_100(self):
        """4 radial * (4+1)^2 angular = 100 coefficients per atom."""
        from salted_ft.basis import BasisSpec

        assert BasisSpec().n_coeffs_per_atom == 100


class TestBasisSpecArithmetic:
    """n_coeffs_per_atom must equal n_radial * (max_l + 1)^2 for any valid spec."""

    @pytest.mark.parametrize(
        "max_l,n_radial,expected",
        [
            (0, 1, 1),  # one s function
            (1, 2, 8),  # 2 * (1 + 3) = 8
            (2, 3, 27),  # 3 * (1 + 3 + 5) = 27
            (4, 4, 100),  # the production default
            (6, 4, 196),  # 4 * (1 + 3 + 5 + 7 + 9 + 11 + 13) = 196
        ],
    )
    def test_n_coeffs_formula(self, max_l, n_radial, expected):
        from salted_ft.basis import BasisSpec

        spec = BasisSpec(
            max_l=max_l,
            n_radial=n_radial,
            sigma=tuple(0.5 * 2**i for i in range(n_radial)),
            cutoff=5.0,
        )
        assert spec.n_coeffs_per_atom == expected

    def test_n_radial_matches_sigma_length(self):
        """sigma is the per-radial-channel width; len(sigma) must equal n_radial."""
        from salted_ft.basis import BasisSpec

        with pytest.raises(ValueError, match=r"n_radial.*sigma"):
            BasisSpec(max_l=2, n_radial=3, sigma=(0.5, 1.0), cutoff=4.0)


class TestBasisSpecValidation:
    """Reject malformed specs at construction time, not at use time."""

    def test_negative_max_l_rejected(self):
        from salted_ft.basis import BasisSpec

        with pytest.raises(ValueError, match=r"max_l"):
            BasisSpec(max_l=-1, n_radial=4, sigma=(0.5, 1.0, 2.0, 4.0), cutoff=4.0)

    def test_zero_n_radial_rejected(self):
        from salted_ft.basis import BasisSpec

        with pytest.raises(ValueError, match=r"n_radial"):
            BasisSpec(max_l=4, n_radial=0, sigma=(), cutoff=4.0)

    def test_negative_sigma_rejected(self):
        """sigma is a Gaussian width; nonpositive widths are nonphysical."""
        from salted_ft.basis import BasisSpec

        with pytest.raises(ValueError, match=r"sigma"):
            BasisSpec(max_l=2, n_radial=2, sigma=(0.5, -1.0), cutoff=4.0)

    def test_nonpositive_cutoff_rejected(self):
        from salted_ft.basis import BasisSpec

        with pytest.raises(ValueError, match=r"cutoff"):
            BasisSpec(max_l=2, n_radial=2, sigma=(0.5, 1.0), cutoff=0.0)


class TestBasisSpecShapes:
    """Shape helpers for downstream tensor allocation."""

    def test_n_angular_components_per_radial(self):
        """(max_l + 1)^2 real spherical harmonic components per radial channel."""
        from salted_ft.basis import BasisSpec

        # l=0,1,2,3,4 -> 1+3+5+7+9 = 25 angular components per radial channel
        assert BasisSpec().n_angular_components == 25

    def test_total_coeffs_shape(self):
        """coeffs tensor shape for a structure: (n_atoms, n_coeffs_per_atom)."""
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        assert spec.total_coeffs_shape(n_atoms=5) == (5, 100)
        assert spec.total_coeffs_shape(n_atoms=1) == (1, 100)


class TestBasisSpecImmutable:
    """BasisSpec must be hashable + immutable so it can key caches / metric runs."""

    def test_is_hashable(self):
        from salted_ft.basis import BasisSpec

        # Two specs with identical fields hash to the same value.
        a = BasisSpec()
        b = BasisSpec()
        assert hash(a) == hash(b)
        assert a == b

    def test_mutation_rejected(self):
        """Frozen dataclass — assigning to a field raises FrozenInstanceError."""
        from dataclasses import FrozenInstanceError

        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        with pytest.raises(FrozenInstanceError):
            spec.max_l = 6  # type: ignore[misc]
