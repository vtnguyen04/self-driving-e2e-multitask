import cv2
import numpy as np
import torch
import unittest
import albumentations as A
import matplotlib.pyplot as plt
import os
from neuro_pilot.data.augment import StandardAugmentor
from neuro_pilot.utils.instance import Instances

class TestAugmentationViz(unittest.TestCase):
    def test_visualize_augmentation(self):
        print("\n=== Visualizing Augmentation Consistency on REAL Image ===")

        # 1. Load Real Image or Fallback
        real_img_path = "neuro_pilot/data/images/video2/video2_frame_000500.jpg"
        if os.path.exists(real_img_path):
            print(f"Loading real image from {real_img_path}")
            img = cv2.imread(real_img_path)
            # Resize early to match training dims
            img = cv2.resize(img, (224, 224))
            # Augmentor expects RGB for Albumentations (usually) but cv2 reads BGR.
            # StandardAugmentor handles conversion internally?
            # neuro_pilot_dataset_v2.py converts to RGB before passing to StandardAugmentor.
            # StandardAugmentor calls Albumentations which assumes image is consistent with what you invoke it with.
            # Let's convert to RGB here to be safe and consistent with dataset class.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print("Real image not found, using dummy.")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.line(img, (50, 50), (170, 170), (255, 255, 255), 5)

        # 2. Define Waypoints (Simulating a straight lane from bottom to top-center)
        # Image is 224x224. Bottom center=(112, 224). Horizon=(112, 100).
        waypoints = np.array([
            [112, 220],
            [112, 180],
            [112, 140],
            [112, 100]
        ], dtype=np.float32)

        # 3. Define BBox (e.g. a car ahead)
        bboxes = [[102, 90, 20, 20]] # Center top
        categories = [1]

        # 4. Initialize Augmentor
        augmentor = StandardAugmentor(training=True, imgsz=224)

        labels = {
            "img": img,
            "waypoints": waypoints,
            "bboxes": bboxes,
            "cls": categories
        }

        # Run Augmentation
        print("Original Waypoints:", waypoints.tolist())

        aug_labels = augmentor(labels)

        img_aug = aug_labels["img"] # Normalize & Tensor

        # Denormalize image for viz
        # Image was Normalized in StandardAugmentor: (x - mean) / std
        # Reverse: x * std + mean
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        if isinstance(img_aug, torch.Tensor):
            img_aug = img_aug.permute(1, 2, 0).numpy()

        img_viz = (img_aug * std + mean) * 255.0
        img_viz = np.clip(img_viz, 0, 255).astype(np.uint8)

        # Convert back to BGR for OpenCV saving
        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR)

        # Draw Augmented Waypoints
        wp_aug = aug_labels["waypoints"]
        print("Augmented Waypoints:", wp_aug.tolist())

        for pt in wp_aug:
            cv2.circle(img_viz, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

        # Draw Augmented BBox
        if len(aug_labels["bboxes"]) > 0:
            bbox = aug_labels["bboxes"][0]
            x, y, w, h = bbox
            cv2.rectangle(img_viz, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

        output_path = "augmentation_test_viz_real.jpg"
        cv2.imwrite(output_path, img_viz)
        print(f"Saved visualization to {output_path}")

        # Since we are using real images, automatic validation of alignment is hard without ground truth.
        # We rely on visual inspection.
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()
