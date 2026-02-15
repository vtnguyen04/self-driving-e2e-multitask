import cv2
import numpy as np
import albumentations as A
import sqlite3
import json
import os
from pathlib import Path

def test_force_augmentation_db():
    print("\n=== Visualizing FORCED Augmentation (Real DB Data) ===")

    # 1. Connect to DB and fetch a valid sample
    db_path = "neuro_pilot/data/dataset.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found!")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get a sample with waypoints
    c.execute("SELECT image_path, data FROM samples WHERE is_labeled=1 AND data LIKE '%waypoints%' LIMIT 1")
    row = c.fetchone()
    conn.close()

    if not row:
        print("No labeled samples found in DB.")
        return

    img_path_db = row['image_path']
    data = json.loads(row['data'])

    print(f"Loaded sample: {img_path_db}")

    # 2. Load Image
    # Handle path differences
    if os.path.exists(img_path_db):
        real_img_path = img_path_db
    elif os.path.exists(os.path.join("neuro_pilot", img_path_db)): # Try prefix
        real_img_path = os.path.join("neuro_pilot", img_path_db)
    elif os.path.exists(img_path_db.replace("/e2e/data", "/e2e/neuro_pilot/data")): # Try fixing path
        real_img_path = img_path_db.replace("/e2e/data", "/e2e/neuro_pilot/data")
    else:
        # Fallback search
        filename = os.path.basename(img_path_db)
        # Try finding it
        for root, dirs, files in os.walk("neuro_pilot/data"):
            if filename in files:
                real_img_path = os.path.join(root, filename)
                break
        else:
            print(f"Could not locate image {img_path_db}")
            return

    img = cv2.imread(real_img_path)
    if img is None:
        print(f"Failed to read image {real_img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) # Resize to match training

    # 3. Extract Labels (Waypoints & BBoxes)
    waypoints = np.array(data.get('waypoints', []), dtype=np.float32)
    bboxes = np.array(data.get('bboxes', []), dtype=np.float32)
    categories = data.get('categories', [])

    if len(waypoints) == 0:
        print("Sample has no waypoints.")
        return

    # 4. Define FORCE Pipeline (Deterministic)
    transform = A.Compose([
        A.Affine(rotate=(-30, -30), p=1.0), # Rotate 30 degrees Clockwise
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
       keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    # Convert bboxes to list for Albumentations if empty
    if len(bboxes) == 0:
        bboxes = []
    else:
        # DB usually stores [x, y, w, h] (Labeler format) - likely needs scaling if image was resized
        # BUT wait, the DB stores coordinates relative to ORIGINAL image size (usually 640x480 or similar).
        # We initialized image as 224x224. We MUST scale the labels!
        # Assuming original is 224x224? Or should we check?
        # NeuroPilotDataset resizes image to 224x224.
        # Let's assume the labels in DB are for 224x224 if they came from the labeler that outputs 224?
        # Actually, let's look at neuro_pilot_dataset_v2.py lines 167-176.
        # It clips/uses coordinates directly relative to 224x224.
        # "Bboxes are stored in 224x224 pixel space (from Labeler)"
        # So we assume they are already 224-scale.
        pass

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
            cv2.circle(viz, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1) # Red

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

    out_path = "augmentation_force_viz_db.jpg"
    cv2.imwrite(out_path, combined)
    print(f"Saved DB visualization to {out_path}")

if __name__ == "__main__":
    test_force_augmentation_db()
