import cv2
import numpy as np
import albumentations as A
import os
import sys

def test_force_augmentation():
    print(f"\nDEBUG: Running from {__file__}", file=sys.stderr)
    print("\n=== Visualizing FORCED Augmentation (30 deg Rotation) ===")

    # 1. Load Real Image or Fallback to Dummy
    real_img_path = "neuro_pilot/data/images/video2/video2_frame_000500.jpg"
    img = None
    if os.path.exists(real_img_path):
        img = cv2.imread(real_img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

    if img is None:
        print("Real image not found or failed, using dummy.")
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.putText(img, "DUMMY", (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Final check for Albumentations
    assert isinstance(img, np.ndarray), f"Image must be numpy array, got {type(img)}"
    assert img.dtype == np.uint8, f"Image must be uint8, got {img.dtype}"

    # 2. Define Waypoints (Straight Line)
    waypoints = np.array([
        [112, 100],  # Top
        [112, 140],
        [112, 180],
        [112, 220]   # Bottom
    ], dtype=np.float32)

    # 3. Define BBox EXACTLY around the 2nd Waypoint [112, 140]
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

    # 6. Visualize
    def draw_viz(img_in, kpts_in, bboxes_in):
        viz = img_in.copy()
        for pt in kpts_in:
            cv2.circle(viz, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)
        for box in bboxes_in:
            x, y, w, h = box
            cv2.rectangle(viz, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        return viz

    viz_orig = draw_viz(img, waypoints, bboxes)
    aug_img = augmented["image"]
    aug_kpts = augmented["keypoints"]
    aug_bboxes = augmented["bboxes"]

    viz_aug = draw_viz(aug_img, aug_kpts, aug_bboxes)
    combined = np.hstack((viz_orig, viz_aug))
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    out_path = "augmentation_force_viz.jpg"
    cv2.imwrite(out_path, combined)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    test_force_augmentation()
