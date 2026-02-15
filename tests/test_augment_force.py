import cv2
import numpy as np
import torch
import albumentations as A
import os

def test_force_augmentation():
    print("\n=== Visualizing FORCED Augmentation (30 deg Rotation) ===")

    # 1. Load Real Image
    real_img_path = "neuro_pilot/data/images/video2/video2_frame_000500.jpg"
    if os.path.exists(real_img_path):
        img = cv2.imread(real_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
    else:
        print("Real image not found, cannot proceed.")
        return

    # 2. Define Waypoints (Straight Line)
    waypoints = np.array([
        [112, 100],  # Top
        [112, 140],
        [112, 180],
        [112, 220]   # Bottom
    ], dtype=np.float32)

    # 3. Define BBox EXACTLY around the 2nd Waypoint [112, 140]
    # Box size 30x30. TopLeft = 112-15, 140-15 = 97, 125
    bboxes = [[97, 125, 30, 30]] # [x, y, w, h]
    categories = [1]

    # 4. Define FORCE Pipeline (Deterministic)
    transform = A.Compose([
        A.Affine(rotate=(-30, -30), p=1.0), # Rotate 30 degrees Clockwise
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
       keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    labels = {
        "image": img,
        "keypoints": waypoints,
        "bboxes": bboxes,
        "category_ids": categories
    }

    # 5. Apply
    augmented = transform(**labels)

    # 6. Visualize Side-by-Side
    def draw_viz(img_in, kpts_in, bboxes_in, title):
        viz = img_in.copy()
        # Draw Waypoints
        for pt in kpts_in:
            cv2.circle(viz, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1) # Red (in RGB)

        # Draw BBox
        for box in bboxes_in:
            x, y, w, h = box
            cv2.rectangle(viz, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2) # Green

        return viz

    viz_orig = draw_viz(img, waypoints, bboxes, "Original")
    viz_aug = draw_viz(augmented["image"], augmented["keypoints"], augmented["bboxes"], "Augmented")

    # Stack Horizontally
    combined = np.hstack((viz_orig, viz_aug))

    # Convert to BGR for saving
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    out_path = "augmentation_force_viz.jpg"
    cv2.imwrite(out_path, combined)
    print(f"Saved side-by-side visualization to {out_path}")
    print("Check artifacts folder.")

if __name__ == "__main__":
    test_force_augmentation()
