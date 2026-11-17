"""CPU-only operator invariants for proposed T-V2TV modules.

This is a toy tensor contract, NOT a working Wan hook and NOT video-quality
evidence. Requires PyTorch; run: python -m unittest discover -s tests -v
"""
import math
import unittest

import torch
from torch import nn


class LowRankTemporalWrite(nn.Module):
    """Trainable event-to-frame residual: [B,F,P,D] + A[B,F,E] U[B,E,d]."""

    def __init__(self, event_dim: int, hidden_dim: int, rank: int) -> None:
        super().__init__()
        if min(event_dim, hidden_dim, rank) < 1:
            raise ValueError("All feature dimensions must be positive.")
        self.in_proj = nn.Parameter(
            torch.randn(event_dim, rank, dtype=torch.double)
            / math.sqrt(event_dim)
        )
        # One and only one last projection is initialized to zero.
        self.out_proj = nn.Parameter(
            torch.zeros(rank, hidden_dim, dtype=torch.double)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        events: torch.Tensor,
        frame_event: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 4 or events.ndim != 3 or frame_event.ndim != 3:
            raise ValueError("Expected [B,F,P,D], [B,E,d], [B,F,E].")
        b, f, _, d = hidden.shape
        if events.shape[0] != b or frame_event.shape != (
            b, f, events.shape[1]
        ):
            raise ValueError("Batch/frame/event axes disagree.")
        if d != self.out_proj.shape[1]:
            raise ValueError("Hidden feature dimension disagrees.")
        if events.shape[1] == 0:
            return hidden
        event_deltas = events @ self.in_proj @ self.out_proj
        frame_deltas = torch.einsum(
            "bfe,bed->bfd", frame_event, event_deltas
        )
        return hidden + frame_deltas[:, :, None, :]


class TestOperatorInvariants(unittest.TestCase):
    def test_zero_init_preserves_video_hidden(self):
        torch.manual_seed(7)
        m = LowRankTemporalWrite(4, 6, 2)
        h = torch.randn(2, 3, 5, 6, dtype=torch.double)
        events = torch.randn(2, 2, 4, dtype=torch.double)
        a = torch.softmax(
            torch.randn(2, 3, 2, dtype=torch.double), dim=-1
        )
        torch.testing.assert_close(m(h, events, a), h, rtol=0, atol=0)

    def test_initial_gradient_to_output_projection_and_later_to_events(self):
        torch.manual_seed(12)
        m = LowRankTemporalWrite(3, 5, 2)
        h = torch.randn(1, 3, 2, 5, dtype=torch.double)
        events = torch.randn(
            1, 2, 3, dtype=torch.double, requires_grad=True
        )
        a = torch.softmax(
            torch.randn(1, 3, 2, dtype=torch.double), dim=-1
        )
        loss = m(h, events, a).square().sum()
        loss.backward()
        self.assertGreater(m.out_proj.grad.norm().item(), 1e-10)
        # With final projection exactly zero, the initial U gradient is zero.
        self.assertEqual(events.grad.norm().item(), 0.0)
        with torch.no_grad():
            m.out_proj.add_(0.01 * m.out_proj.grad)
        m.zero_grad(set_to_none=True)
        events.grad = None
        loss_after_step = m(h, events, a).square().sum()
        loss_after_step.backward()
        self.assertGreater(events.grad.norm().item(), 1e-10)

    def test_time_routing_is_more_expressive_than_global_mix(self):
        m = LowRankTemporalWrite(2, 2, 2)
        with torch.no_grad():
            m.in_proj.copy_(torch.eye(2, dtype=torch.double))
            m.out_proj.copy_(torch.eye(2, dtype=torch.double))
        h = torch.zeros(1, 2, 1, 2, dtype=torch.double)
        events = torch.eye(2, dtype=torch.double)[None]
        routed = torch.eye(2, dtype=torch.double)[None]
        global_mix = torch.full((1, 2, 2), 0.5, dtype=torch.double)
        r = m(h, events, routed)[:, :, 0]
        g = m(h, events, global_mix)[:, :, 0]
        torch.testing.assert_close(r[0], torch.eye(2, dtype=torch.double))
        torch.testing.assert_close(g[0, 0], g[0, 1])
        self.assertFalse(torch.allclose(r[0, 0], r[0, 1]))

    def test_time_swapping_rows_swaps_outputs(self):
        m = LowRankTemporalWrite(2, 2, 2)
        with torch.no_grad():
            m.in_proj.copy_(torch.eye(2, dtype=torch.double))
            m.out_proj.copy_(torch.eye(2, dtype=torch.double))
        h = torch.zeros(1, 3, 2, 2, dtype=torch.double)
        u = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.double)
        a = torch.tensor(
            [[[1., 0.], [0.2, 0.8], [0., 1.]]], dtype=torch.double
        )
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(
            m(h, u, a[:, order])[:, :, 0],
            m(h, u, a)[:, order, 0],
        )

    def test_no_events_is_exact_identity(self):
        m = LowRankTemporalWrite(4, 6, 2)
        h = torch.randn(2, 3, 2, 6, dtype=torch.double)
        u = torch.empty(2, 0, 4, dtype=torch.double)
        a = torch.empty(2, 3, 0, dtype=torch.double)
        self.assertIs(m(h, u, a), h)

    def test_gradient_vs_finite_difference(self):
        torch.manual_seed(4)
        m = LowRankTemporalWrite(3, 4, 2)
        with torch.no_grad():
            m.out_proj.copy_(
                torch.randn_like(m.out_proj) * 0.2
            )
        h = torch.randn(1, 2, 2, 4, dtype=torch.double)
        events = torch.randn(
            1, 2, 3, dtype=torch.double, requires_grad=True
        )
        a = torch.softmax(
            torch.randn(1, 2, 2, dtype=torch.double), dim=-1
        )
        direction = torch.randn_like(events)
        scalar = m(h, events, a).square().sum()
        grad, = torch.autograd.grad(scalar, events)
        analytical = (grad * direction).sum()
        eps = 1e-6
        plus = m(h, events.detach() + eps * direction, a)
        minus = m(h, events.detach() - eps * direction, a)
        numerical = (plus.square().sum() - minus.square().sum()) / (2 * eps)
        torch.testing.assert_close(
            analytical, numerical, rtol=1e-6, atol=1e-6
        )

    def test_latent_temporal_grid_not_pixel_frames(self):
        pixel_frames = 49
        vae_time_stride = 4
        f_z = (pixel_frames - 1) // vae_time_stride + 1
        self.assertEqual(f_z, 13)
        h_prime, w_prime = 480 // (8 * 2), 832 // (8 * 2)
        self.assertEqual(f_z * h_prime * w_prime, 20280)
        # Frame-major flatten/unflatten preserves each latent time index.
        x = torch.arange(13 * 2 * 3).reshape(1, 13 * 2 * 3, 1)
        times = x.reshape(1, 13, 6, 1)
        self.assertEqual(times[0, 4, 0, 0].item(), 24)


if __name__ == "__main__":
    unittest.main()
