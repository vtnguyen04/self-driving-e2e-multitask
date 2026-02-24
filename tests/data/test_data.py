import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from neuro_pilot.data.neuro_pilot_dataset_v2 import NeuroPilotDataset, custom_collate_fn, Sample
from neuro_pilot.data.augment import StandardAugmentor

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_sample = Sample(
            image_path="dummy.jpg",
            bboxes=[[10, 10, 40, 40]],
            categories=[1],
            waypoints=[[0.0, 0.0]] * 10,
            command=0
        )

    def test_dataset_getitem(self):
        samples = [self.dummy_sample]
        with patch('cv2.imread', return_value=self.dummy_img):
            transform = StandardAugmentor(training=False, imgsz=224)
            # Mock transform to return standardized dict
            with patch.object(transform, 'transform') as mock_t:
                mock_t.return_value = {
                    'image': np.zeros((224, 224, 3), dtype=np.uint8),
                    'bboxes': [[10, 10, 40, 40]],
                    'category_ids': [1],
                    'keypoints': [[0.0, 0.0]] * 10
                }
                ds = NeuroPilotDataset(samples=samples, transform=transform)
                data = ds[0]
                self.assertIn('image', data)
                self.assertEqual(data['command_idx'], 0)
                self.assertIn('curvature', data)
                self.assertIn('categories', data)
                self.assertIn('heatmap', data)

    def test_robustness_injection(self):
        # Sample with command 0 (FOLLOW_LANE) should trigger robustness injection
        samples = [self.dummy_sample]
        # Patch random.random at the top level or wherever it's used
        with patch('random.random', return_value=0.1):
            ds = NeuroPilotDataset(samples=samples, split='train')
            ds.close_mosaic() # Triggers robustness injection
            self.assertEqual(len(ds.samples), 2)
            self.assertIn(ds.samples[1].command, [1, 2])

    def test_mosaic(self):
        from neuro_pilot.data.augment import Mosaic
        dataset = MagicMock()
        dataset.get_image_and_label.return_value = {
            "img": np.zeros((224, 224, 3), dtype=np.uint8),
            "bboxes": np.array([[10, 10, 50, 50]]),
            "cls": np.array([1]),
            "waypoints": np.zeros((10, 2))
        }
        labels = {
            "img": np.zeros((224, 224, 3), dtype=np.uint8),
            "bboxes": np.array([[10.0, 10.0, 50.0, 50.0]]),
            "cls": np.array([1]),
            "waypoints": np.zeros((10, 2))
        }
        mosaic = Mosaic(dataset, imgsz=224, p=1.0)
        with patch('random.randint', return_value=0), \
             patch('random.uniform', return_value=112):
            result = mosaic(labels)
            self.assertEqual(result["img"].shape, (448, 448, 3))
            self.assertIn("cls", result) # Changed from categories to cls

    def test_collate_fn(self):
        batch = [{
            'image': torch.zeros(3, 224, 224),
            'image_path': 'dummy.jpg',
            'command': torch.zeros(4),
            'command_idx': 0,
            'waypoints': torch.zeros(10, 2),
            'curvature': torch.tensor(0.1),
            'heatmap': torch.zeros(56, 56),
            'bboxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            'categories': torch.tensor([1])
        }]
        collated = custom_collate_fn(batch)
        self.assertEqual(collated['image'].shape, (1, 3, 224, 224))
        self.assertEqual(collated['command_idx'].shape, (1,))
        self.assertEqual(collated['curvature'].shape, (1,))
        self.assertIn('cls', collated)
        self.assertIn('batch_idx', collated)

if __name__ == '__main__':
    unittest.main()
