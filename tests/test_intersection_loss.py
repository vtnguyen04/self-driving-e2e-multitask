import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.losses import EnhancedBezierLoss

def test_intersection_loss():
    print("=== Testing EnhancedBezierLoss (Precision & Intersection) ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize Loss
    loss_fn = EnhancedBezierLoss(
        num_eval_points=20,
        endpoint_weight=1.0, # Keep endpoint weight low for base test to see other components
        cte_weight=5.0,
        heading_weight=1.0,
        anchor_weight=10.0,
        device=device
    )

    # Config
    B = 1
    N = 20

    # --- Case 1: Perfect Match ---
    print("\n[Case 1: Perfect Match]")
    # Create a straight line trajectory
    gt_waypoints = torch.zeros(B, N, 2).to(device)
    for i in range(N):
        gt_waypoints[0, i, 0] = i * 0.1 # x moves
        gt_waypoints[0, i, 1] = 0.5     # y constant

    # Mock predictions (perfect match)
    # We need to reverse-engineer control points that produce this line?
    # Or just mock the internal 'pred_path' if we can?
    # EnhancedBezierLoss computes pred_path from control_points.
    # To get a straight line from (0, 0.5) to (1.9, 0.5), we need 4 collinear control points.
    p0 = torch.tensor([0.0, 0.5]).to(device)
    p1 = torch.tensor([0.6, 0.5]).to(device)
    p2 = torch.tensor([1.3, 0.5]).to(device)
    p3 = torch.tensor([1.9, 0.5]).to(device)

    control_points = torch.stack([p0, p1, p2, p3]).unsqueeze(0).to(device) # [1, 4, 2]

    dummy_pred = {'control_points': control_points}
    dummy_target = {'waypoints': gt_waypoints}

    # We expect near zero loss
    loss = loss_fn(dummy_pred, dummy_target)
    print(f"Loss (Perfect): {loss.item():.6f}")
    assert loss.item() < 0.1, "Perfect match should have near-zero loss"

    # --- Case 2: Lateral Offset (CTE) ---
    print("\n[Case 2: Cross-Track Error (Lateral Offset)]")
    # Shift prediction laterally by 0.1
    p0_off = p0 + torch.tensor([0.0, 0.1]).to(device)
    p1_off = p1 + torch.tensor([0.0, 0.1]).to(device)
    p2_off = p2 + torch.tensor([0.0, 0.1]).to(device)
    p3_off = p3 + torch.tensor([0.0, 0.1]).to(device)

    cp_off = torch.stack([p0_off, p1_off, p2_off, p3_off]).unsqueeze(0).to(device)
    pred_off = {'control_points': cp_off}

    loss_cte = loss_fn(pred_off, dummy_target)
    print(f"Loss (Lateral Offset): {loss_cte.item():.6f}")

    # --- Case 3: Heading Mismatch ---
    print("\n[Case 3: Heading Mismatch]")
    # Target goes straight East.
    # Pred starts at same point but veers North East (45 deg).
    # p0 same, others rotated.
    p1_rot = p0 + torch.tensor([0.6, 0.6]) * 0.707
    p2_rot = p0 + torch.tensor([1.3, 1.3]) * 0.707
    p3_rot = p0 + torch.tensor([1.9, 1.9]) * 0.707

    cp_rot = torch.stack([p0, p1_rot, p2_rot, p3_rot]).unsqueeze(0).to(device)
    pred_rot = {'control_points': cp_rot}

    loss_head = loss_fn(pred_rot, dummy_target)
    print(f"Loss (Heading Mismatch): {loss_head.item():.6f}")

    # --- Case 4: Endpoint Anchor ---
    print("\n[Case 4: Endpoint Deviation]")
    # Pred matches perfectly until last point deviation
    # Ideally change only p3 significantly?
    # Benzier is global, so changing P3 affects curve ending.
    p3_bad = p3 + torch.tensor([0.2, 0.2]) # Significant jump
    cp_bad = torch.stack([p0, p1, p2, p3_bad]).unsqueeze(0).to(device)
    pred_bad = {'control_points': cp_bad}

    loss_anchor = loss_fn(pred_bad, dummy_target)
    print(f"Loss (Endpoint Deviation): {loss_anchor.item():.6f}")

    # Comparison
    print("\n--- Summary ---")
    print(f"Perfect: {loss.item():.4f}")
    print(f"Lateral (CTE High): {loss_cte.item():.4f}")
    print(f"Heading (Direction): {loss_head.item():.4f}")
    print(f"Anchor (End): {loss_anchor.item():.4f}")

    # Assertions
    if loss_cte.item() > loss.item() * 10:
        print("PASS: Lateral Offset triggers significant penalty (CTE works).")
    else:
        print("FAIL: Lateral Offset penalty too low.")

    if loss_anchor.item() > loss.item():
        print("PASS: Endpoint deviation detected.")

if __name__ == "__main__":
    test_intersection_loss()
