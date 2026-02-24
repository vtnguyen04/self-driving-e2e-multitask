import cv2
import numpy as np
import torch
import unittest
import os
from neuro_pilot.data.augment import StandardAugmentor

class TestAugmentationViz(unittest.TestCase):
    def test_visualize_augmentation(self):
        print("\n=== Visualizing Augmentation Consistency on REAL Image ===")

        # 1. Load Real Image or Fallback
        real_img_path = "neuro_pilot/data/images/video2/video2_frame_000500.jpg"
        img = None
        if os.path.exists(real_img_path):
            img = cv2.imread(real_img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            print("Real image not found, using dummy.")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.line(img, (50, 50), (170, 170), (255, 255, 255), 5)
            cv2.putText(img, "DUMMY VIZ", (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. Define labels
        waypoints = np.array([[112, 220], [112, 180], [112, 140], [112, 100]], dtype=np.float32)
        bboxes = [[102, 90, 122, 110]] # x1, y1, x2, y2
        categories = [1]

        # 3. Augment
        augmentor = StandardAugmentor(training=True, imgsz=224)
        labels = {
            "img": img,
            "waypoints": waypoints,
            "bboxes": bboxes,
            "cls": categories
        }

        aug_labels = augmentor(labels)

        # 4. Verify
        self.assertIsInstance(aug_labels['img'], torch.Tensor)
        output_path = "augmentation_test_viz_real.jpg"
        # Visualization logic (simplified)
        img_aug = aug_labels["img"].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_viz = (img_aug * std + mean) * 255.0
        img_viz = np.clip(img_viz, 0, 255).astype(np.uint8)
        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, img_viz)
        self.assertTrue(os.path.exists(output_path))
        print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    unittest.main()
