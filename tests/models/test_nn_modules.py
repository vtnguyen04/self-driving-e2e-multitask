import unittest
import torch
from neuro_pilot.nn.modules import (
    Conv, Bottleneck, C3k2, Proto, C2PSA, Detect, Segment, HeatmapHead, TrajectoryHead, SelectFeature, TimmBackbone, ClassificationHead
)

class TestNNModules(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.in_ch = 64
        self.out_ch = 128
        self.img_size = 32 # small size for fast testing

    def test_conv(self):
        print("Testing Conv...")
        m = Conv(self.in_ch, self.out_ch, k=3, s=1, p=None).to(self.device)
        x = torch.randn(self.batch_size, self.in_ch, self.img_size, self.img_size).to(self.device)
        y = m(x)
        self.assertEqual(y.shape, (self.batch_size, self.out_ch, self.img_size, self.img_size))

    def test_bottleneck(self):
        print("Testing Bottleneck...")
        m = Bottleneck(self.in_ch, self.in_ch, shortcut=True).to(self.device)
        x = torch.randn(self.batch_size, self.in_ch, self.img_size, self.img_size).to(self.device)
        y = m(x)
        self.assertEqual(y.shape, (self.batch_size, self.in_ch, self.img_size, self.img_size))

    def test_c3k2(self):
        print("Testing C3k2...")
        m = C3k2(self.in_ch, self.out_ch, n=1, shortcut=True, c3k=True).to(self.device)
        x = torch.randn(self.batch_size, self.in_ch, self.img_size, self.img_size).to(self.device)
        y = m(x)
        self.assertEqual(y.shape, (self.batch_size, self.out_ch, self.img_size, self.img_size))

    def test_c2psa(self):
        print("Testing C2PSA...")
        # C2PSA requires c1 == c2 in the current implementation
        m = C2PSA(self.in_ch, self.in_ch, n=1).to(self.device)
        x = torch.randn(self.batch_size, self.in_ch, self.img_size, self.img_size).to(self.device)
        y = m(x)
        self.assertEqual(y.shape, (self.batch_size, self.in_ch, self.img_size, self.img_size))

    def test_proto(self):
        print("Testing Proto (Production Grade)...")
        nm = 32
        m = Proto(self.in_ch, c_=64, c2=nm).to(self.device)
        x = torch.randn(self.batch_size, self.in_ch, self.img_size, self.img_size).to(self.device)
        y = m(x)
        # Proto upsamples by 2
        self.assertEqual(y.shape, (self.batch_size, nm, self.img_size * 2, self.img_size * 2))

    def test_detect(self):
        print("Testing Detect...")
        nc = 14
        ch = (64, 128, 256)
        m = Detect(nc=nc, ch=ch).to(self.device)
        x = [
            torch.randn(self.batch_size, 64, 80, 80).to(self.device),
            torch.randn(self.batch_size, 128, 40, 40).to(self.device),
            torch.randn(self.batch_size, 256, 20, 20).to(self.device)
        ]
        y = m(x)
        # Standard neuro_pilot Detect returns a dict
        if isinstance(y, dict) and 'one2many' in y:
             y = y['one2many']

        self.assertIn('boxes', y)
        self.assertIn('scores', y)

    def test_segment(self):
        print("Testing Segment...")
        nc, nm = 14, 32
        ch = (64, 128, 256)
        m = Segment(nc=nc, nm=nm, npr=256, ch=ch, end2end=True).to(self.device)
        x = [
            torch.randn(self.batch_size, 64, 80, 80).to(self.device),
            torch.randn(self.batch_size, 128, 40, 40).to(self.device),
            torch.randn(self.batch_size, 256, 20, 20).to(self.device)
        ]
        y = m(x)

        if isinstance(y, dict) and 'one2many' in y:
             y = y['one2many']

        self.assertIn('boxes', y)
        self.assertIn('scores', y)
        self.assertIn('mask_coefficient', y)
        self.assertIn('proto', y)

        # Verify shape of mask_coefficient: (batch, nm, all_anchors)
        # Anchors: (80*80) + (40*40) + (20*20) = 6400 + 1600 + 400 = 8400
        self.assertEqual(y['mask_coefficient'].shape[1], nm)
        self.assertEqual(y['proto'].shape, (self.batch_size, nm, 80*2, 80*2))

    def test_heatmap_head(self):
        print("Testing HeatmapHead...")
        # HeatmapHead expects [p3, c2], where p3 is upsampled by 2 to match c2
        m = HeatmapHead(ch_in=[64, 128], ch_out=1).to(self.device)
        x = [
            torch.randn(self.batch_size, 64, 8, 8).to(self.device),  # p3 (stride 8)
            torch.randn(self.batch_size, 128, 16, 16).to(self.device) # c2 (stride 4)
        ]
        y = m(x)
        self.assertIn('heatmap', y)
        # Should be upsampled by 4x from c2 (16x16 -> 64x64)
        expected_shape = (x[1].shape[2] * 4, x[1].shape[3] * 4)
        self.assertEqual(y['heatmap'].shape[2:], expected_shape)

    def test_trajectory_head(self):
        print("Testing TrajectoryHead...")
        m = TrajectoryHead(ch_in=128, num_waypoints=10).to(self.device)
        x = torch.randn(self.batch_size, 128, 8, 8).to(self.device)
        y = m(x, cmd_idx=torch.zeros(self.batch_size, dtype=torch.long).to(self.device))
        self.assertIn('waypoints', y)
        self.assertEqual(y['waypoints'].shape, (self.batch_size, 10, 2))

    def test_select_feature(self):
        print("Testing SelectFeature...")
        m = SelectFeature(index=1)
        # Timm backbone outputs dict of channels
        x = {0: torch.randn(1, 1), 1: torch.randn(1, 32)}
        y = m(x)
        self.assertEqual(y.shape[1], 32)

    def test_timm_backbone(self):
        print("Testing TimmBackbone...")
        m = TimmBackbone(model_name='mobilenetv4_conv_small_050', pretrained=False).to(self.device)
        x = torch.randn(self.batch_size, 3, 224, 224).to(self.device)
        y = m(x)
        # Features only mode returns a list of tensors
        self.assertIsInstance(y, (dict, list, tuple))

    def test_classification_head(self):
        print("Testing ClassificationHead...")
        m = ClassificationHead(ch=128, nc=10).to(self.device)
        x = torch.randn(self.batch_size, 128, 7, 7).to(self.device)
        y = m(x)
        self.assertIn('classes', y)
        self.assertEqual(y['classes'].shape, (self.batch_size, 10))

if __name__ == '__main__':
    unittest.main()
