import sys
from unittest.mock import MagicMock

# Mock dependencies before imports
import unittest.mock as mock
if "timm" in sys.modules:
    del sys.modules["timm"]
sys.modules["timm"] = mock.MagicMock()

sys.modules["timm"].create_model.return_value.feature_info.channels.return_value = [32, 32, 64, 96, 960]
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.ops"] = MagicMock()

import unittest
import torch
from neuro_pilot.nn.modules.attention import CommandGate, VLFusion

class TestCommandGate(unittest.TestCase):
    def test_gate_shape(self):
        gate = CommandGate(embed_dim=128)
        x = torch.randn(2, 128, 28, 28) # [B, C, H, W]
        out = gate(x)
        self.assertEqual(out.shape, (2, 1, 1))
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))

    def test_fusion_gating_logic(self):
        # Create a VLFusion module
        fusion = VLFusion(c1=32, c2=32, num_heads=2)

        # Manually set gate weights to output ~0 (suppress)
        # We can't easily force untrained weights, so we'll mock the gate

        x = torch.randn(1, 32, 10, 10)
        lang = torch.randn(1, 1, 32)

        # 1. Run with normal gate
        out = fusion(x, lang)
        out_normal, _gate_out = out["feats"], out["gate_score"]
        self.assertEqual(out_normal.shape, x.shape)

        # 2. Mock gate to return 0 (Closed Gate)
        original_gate_forward = fusion.gate.forward
        fusion.gate.forward = lambda feature: torch.zeros(1, 1, 1)

        out_closed_dict = fusion(x, lang)
        out_closed = out_closed_dict["feats"]

        # If gate is 0, output should be exactly input (after norm)
        # Because: x_flat = norm(x_flat + 0 * attn) = norm(x_flat)
        # Note: LayerNorm makes it not exactly equal to input if input wasn't normalized.
        # But we can check that it's different from the "normal" run which has random attention added.

        self.assertFalse(torch.allclose(out_normal, out_closed), "Gating should change the output")

        # Restore
        fusion.gate.forward = original_gate_forward

if __name__ == '__main__':
    unittest.main()
