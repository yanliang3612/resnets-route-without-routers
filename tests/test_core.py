from __future__ import annotations

import unittest

import torch

from resnet_routes.resnet18.mobius import (
    enumerate_masks,
    mobius_coefficients as mobius_coefficients_resnet18,
    mobius_reconstruct,
)
from resnet_routes.resnet18.reconstruction import (
    mobius_reconstruction_error,
    per_image_metrics,
    reconstruction_mobius,
)
from resnet_routes.resnet18.spectrum import (
    derived_metrics,
    order_energies_scalar,
    order_indices,
)
from resnet_routes.resnet34.mobius import (
    inverse_mobius_transform,
    mobius_coefficients as mobius_coefficients_resnet34,
)


def known_coefficients() -> torch.Tensor:
    return torch.tensor(
        [[
            [0.5, -1.0], [1.0, 0.25], [-2.0, 0.5], [3.0, 1.5],
            [0.75, -0.5], [-1.25, 2.0], [0.2, -0.8], [4.0, 0.1],
        ]],
        dtype=torch.float64,
    )


def subset_sums(coefficients: torch.Tensor) -> torch.Tensor:
    values = torch.zeros_like(coefficients)
    for subset in range(coefficients.shape[1]):
        terms = [term for term in range(coefficients.shape[1]) if term & subset == term]
        values[:, subset] = coefficients[:, terms].sum(dim=1)
    return values


class MobiusTests(unittest.TestCase):
    def test_mask_enumeration_uses_lsb_order(self) -> None:
        expected = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(enumerate_masks(3, device="cpu"), expected)

    def test_both_transforms_recover_known_interactions(self) -> None:
        expected = known_coefficients()
        responses = subset_sums(expected)
        actual18 = mobius_coefficients_resnet18(responses)
        actual34 = mobius_coefficients_resnet34(responses, dtype=torch.float64)
        torch.testing.assert_close(actual18, expected)
        torch.testing.assert_close(actual34, expected)
        torch.testing.assert_close(mobius_reconstruct(actual18), responses[:, -1])
        torch.testing.assert_close(inverse_mobius_transform(actual34), responses)

    def test_resnet34_rejects_non_power_of_two_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            mobius_coefficients_resnet34(torch.zeros(1, 6, 2))


class MetricTests(unittest.TestCase):
    def test_scalar_order_spectrum(self) -> None:
        orders = order_indices(2)
        energy = order_energies_scalar(
            torch.tensor([[10.0, 1.0, 2.0, 3.0]]), orders
        )
        torch.testing.assert_close(energy, torch.tensor([[100.0, 5.0, 9.0]]))
        metrics = derived_metrics(energy)
        torch.testing.assert_close(
            metrics["E_tilde"], torch.tensor([[5.0 / 14.0, 9.0 / 14.0]])
        )
        torch.testing.assert_close(metrics["kappa"], torch.tensor([23.0 / 14.0]))

    def test_exact_reconstruction_metrics(self) -> None:
        full = torch.tensor([[4.0, 2.0]])
        coefficients = torch.tensor(
            [[[1.0, 0.0], [1.0, 1.0], [-1.0, 3.0], [3.0, -2.0]]]
        )
        reconstructed = reconstruction_mobius(coefficients)
        torch.testing.assert_close(reconstructed, full)
        torch.testing.assert_close(
            mobius_reconstruction_error(full, reconstructed), torch.zeros(1)
        )
        metrics = per_image_metrics(full, reconstructed, full.argmax(dim=-1))
        torch.testing.assert_close(metrics["rel_err"], torch.zeros(1))
        torch.testing.assert_close(metrics["top1_agree"], torch.ones(1))


if __name__ == "__main__":
    unittest.main()
