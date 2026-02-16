import torch
import torch.nn as nn
import torch.optim as optim
import unittest

class ToyGatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision encoder
        self.vision_encoder = nn.Linear(10, 2)
        # Command encoder
        self.cmd_encoder = nn.Embedding(4, 2)

        # GATE: Initialize to bias towards 0.5 (Neutral)
        self.gate_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

        # HEAD: Simple predictor
        self.head = nn.Linear(2, 2)

    def forward(self, x_vision, cmd_idx):
        v_feat = self.vision_encoder(x_vision) # [B, 2]
        c_feat = self.cmd_encoder(cmd_idx)     # [B, 2]
        gate = self.gate_net(v_feat)           # [B, 1]

        # CRITICAL: If gate is 0, we rely ONLY on vision.
        fused = v_feat + gate * c_feat
        return self.head(fused), gate

class TestSparsityLearning(unittest.TestCase):
    def test_self_supervised_gating(self):
        print("\n=== Toy Experiment: Learning to be Lazy (Sparsity) ===")
        torch.manual_seed(42)

        model = ToyGatedModel()
        optimizer = optim.Adam(model.parameters(), lr=0.01) # Lower LR for stability

        # DATA GENERATION
        # Scenario 1: Curve (Vision Sufficient)
        # Vision [1, 0...] -> Target [1, 1]. Command is random noise.
        # We make vision very strong predictor here.
        vision_curve = torch.randn(100, 10) * 2
        # Target defined by a hidden rule from vision, e.g. vision[0:2]
        target_curve = vision_curve[:, :2]
        cmd_curve = torch.randint(0, 4, (100,))

        # Scenario 2: Intersection (Ambiguous Vision)
        # Vision is almost zero (noise).
        # Target depends entirely on Command.
        vision_inter = torch.randn(100, 10) * 0.01 # Weak vision
        cmd_inter = torch.randint(0, 4, (100,))
        # Perfect command mapping
        # SCALE TARGETS UP: Make the error of missing the turn HUGE.
        target_inter = torch.zeros(100, 2)
        for i, c in enumerate(cmd_inter):
            if c == 0: target_inter[i] = torch.tensor([0.0, 5.0]) # Go Straight long
            elif c == 1: target_inter[i] = torch.tensor([-5.0, 0.0]) # Turn Left long
            elif c == 2: target_inter[i] = torch.tensor([5.0, 0.0]) # Turn Right long
            elif c == 3: target_inter[i] = torch.tensor([0.0, -5.0]) # Back

        print("Training Phase...")
        for epoch in range(1000): # More epochs
            optimizer.zero_grad()

            # Forward Curve
            pred_c, gate_c = model(vision_curve, cmd_curve)
            loss_traj_c = (pred_c - target_curve).pow(2).mean()

            # Forward Inter
            pred_i, gate_i = model(vision_inter, cmd_inter)
            loss_traj_i = (pred_i - target_inter).pow(2).mean()

            # Total Loss
            # Key: lambda must be small enough that reducing L_traj (Inter) > Cost of opening Gate.
            # L_traj error starts high (~1.0). Gate cost max is 1.0.
            # lambda=0.05 means opening gate costs 0.05. Reducing error by 0.5 is worth it.
            l_sparsity = (gate_c.mean() + gate_i.mean()) / 2
            loss = (loss_traj_c + loss_traj_i) + 0.05 * l_sparsity

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                 print(f"Epoch {epoch}: Gate_Curve={gate_c.mean().item():.3f}, Gate_Inter={gate_i.mean().item():.3f}")

        final_gate_curve = gate_c.mean().item()
        final_gate_inter = gate_i.mean().item()

        print("\nFinal Results:")
        print(f"Gate on Curve: {final_gate_curve:.3f}")
        print(f"Gate on Intersection: {final_gate_inter:.3f}")

        self.assertLess(final_gate_curve, 0.2)
        self.assertGreater(final_gate_inter, 0.8)

        print(">>> SUCCESS: Model effectively learned WHEN to listen to commands without explicit labels!")

if __name__ == '__main__':
    unittest.main()
