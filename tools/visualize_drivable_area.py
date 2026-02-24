import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_heatmap(coords, H, W, base_sigma=3.0, drivable_area_sigma=15.0):
    """
    Generate ground truth heatmap from waypoint coordinates.
    Allows toggling between a thin trajectory and a thick drivable area.
    """
    B, K, _ = coords.shape
    device = coords.device
    
    # Standard trajectory sigma vs. Drivable Area sigma
    sigma_thin = max(H, W) / 160.0 * base_sigma
    sigma_thick = max(H, W) / 160.0 * drivable_area_sigma
    
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, H, W, 2)
    
    # Scale coordinates to absolute pixel values
    pts = (coords + 1) / 2 * torch.tensor([W, H], device=device).view(1, 1, 2)

    heatmap_thin = torch.zeros((B, 1, H, W), device=device)
    heatmap_thick = torch.zeros((B, 1, H, W), device=device)
    
    for i in range(K - 1):
        p1 = pts[:, i:i+1, :].view(B, 1, 1, 2)
        p2 = pts[:, i+1:i+2, :].view(B, 1, 1, 2)
        v = p2 - p1
        w = grid - p1
        
        t = torch.clamp(torch.sum(w * v, dim=-1) / (torch.sum(v * v, dim=-1) + 1e-6), 0.0, 1.0)
        projection = p1 + t.unsqueeze(-1) * v
        dist_sq = torch.sum((grid - projection) ** 2, dim=-1)
        
        # Calculate thin segment
        segment_thin = torch.exp(-dist_sq / (2 * sigma_thin ** 2))
        heatmap_thin = torch.maximum(heatmap_thin, segment_thin.unsqueeze(1))
        
        # Calculate thick segment (drivable area proxy)
        segment_thick = torch.exp(-dist_sq / (2 * sigma_thick ** 2))
        heatmap_thick = torch.maximum(heatmap_thick, segment_thick.unsqueeze(1))
        
    return heatmap_thin, heatmap_thick

def visualize():
    print("Generating sample heatmaps...")
    H, W = 160, 160 # Common aspect for downsampled features or full res
    
    # Create a dummy curved path (normalized coordinates -1 to 1)
    # E.g., a car turning left
    x = np.linspace(-0.2, -0.8, 10)
    y = np.linspace(1.0, 0.2, 10)
    
    coords = np.stack([x, y], axis=-1)
    coords = torch.from_numpy(coords).float().unsqueeze(0) # [1, 10, 2]
    
    # Generate Heatmaps
    heatmap_thin, heatmap_thick = generate_heatmap(coords, H, W, base_sigma=3.0, drivable_area_sigma=20.0)
    
    h_thin = heatmap_thin[0, 0].cpu().numpy()
    h_thick = heatmap_thick[0, 0].cpu().numpy()
    
    # Colorize using OpenCV
    def colorize(hm):
        hm_norm = (hm * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        return colorized

    vis_thin = colorize(h_thin)
    vis_thick = colorize(h_thick)
    
    # Plotting
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(vis_thin, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Current: Trajectory Heatmap (Sigma=3.0)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(vis_thick, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Proposed: Drivable Area Heatmap (Sigma=20.0)')
    axes[1].axis('off')
    
    out_path = "heatmap_drivable_area_compare.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    visualize()
