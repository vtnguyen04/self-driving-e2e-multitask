import cv2
import numpy as np
import torch
import os
import albumentations as A
from neuro_pilot.data.augment import StandardAugmentor

def test_force_augmentation_db():
    print("\n=== Visualizing FORCED Augmentation from DB Sample ===")

    # 1. Mock DB data (Simplified)
    data = {
        'img_path': 'neuro_pilot/data/images/video2/video2_frame_000500.jpg',
        'waypoints': [[112, 100], [112, 140], [112, 180], [112, 220]],
        'bboxes': [[97, 125, 30, 30]],
        'categories': [1]
    }

    # 2. Find Image
    img_path = data['img_path']
    real_img_path = None
    if os.path.exists(img_path):
        real_img_path = img_path

    img = None
    if real_img_path:
        img = cv2.imread(real_img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

    if img is None:
        print("Image not found, using dummy for DB test.")
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.putText(img, "DB DUMMY", (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 3. Extract Labels
    waypoints = np.array(data['waypoints'], dtype=np.float32)
    bboxes = np.array(data['bboxes'], dtype=np.float32)
    cls = data['categories']

    # 4. Augmentor
    augmentor = StandardAugmentor(training=True, imgsz=224)

    # 5. Apply
    labels = {
        "img": img,
        "waypoints": waypoints,
        "bboxes": bboxes,
        "cls": cls
    }

    aug_labels = augmentor(labels)

    # 6. Verify and Visualize
    assert isinstance(aug_labels['img'], torch.Tensor)
    print("Augmentation from DB successful.")

if __name__ == "__main__":
    test_force_augmentation_db()
