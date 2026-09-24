"""Checks that hidden raw samples stay hidden and only the student is optimized."""

import unittest

import torch

from ecg_experiment.lead_innovation import (
    LeadMultiscaleEncoder, LeadSSL, content_targets, lead_patch_mask,
)


class LeadInnovationTest(unittest.TestCase):
    def test_static_lead_time_pattern_has_zero_targets(self):
        torch.manual_seed(9)
        constant_position_pattern = torch.randn(1, 8, 25, 96).repeat(4, 1, 1, 1)
        ordinary, innovation, centered, residual = content_targets(constant_position_pattern)
        self.assertTrue(torch.equal(ordinary, torch.zeros_like(ordinary)))
        self.assertTrue(torch.equal(innovation, torch.zeros_like(innovation)))
        self.assertTrue(torch.equal(centered, torch.zeros_like(centered)))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_mask_visibility_and_gradient_flow(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        signal = torch.randn(3, 8, 1000)
        mask = lead_patch_mask(3, signal.device)
        self.assertEqual(mask.shape, (3, 8, 25))
        self.assertTrue(torch.all(mask.sum(dim=(1, 2)) == 35))

        encoder = LeadMultiscaleEncoder().eval()
        changed = signal.clone()
        changed[mask.repeat_interleave(40, dim=-1)] = 1000.0
        with torch.no_grad():
            self.assertTrue(torch.allclose(encoder.tokens(signal, mask),
                                           encoder.tokens(changed, mask)))
            self.assertEqual(encoder(signal).shape, (3, 192))

        model = LeadSSL(0.5)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=1e-3)
        before = model.encoder.morphology.weight.detach().clone()
        loss, _ = model(signal)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.encoder.morphology.weight))
        self.assertTrue(torch.isfinite(model.encoder.morphology.weight).all())


if __name__ == "__main__":
    unittest.main()
