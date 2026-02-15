
import unittest
from neuro_pilot.cfg.schema import AppConfig, BackboneConfig, HeadConfig

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = AppConfig()
        self.assertEqual(cfg.project_name, "neuro_pilot_e2e")
        self.assertEqual(cfg.head.num_classes, 14)
        self.assertEqual(cfg.backbone.pretrained, True)

    def test_custom_config(self):
        cfg = AppConfig(
            project_name="custom_project",
            head=HeadConfig(num_classes=5)
        )
        self.assertEqual(cfg.project_name, "custom_project")
        self.assertEqual(cfg.head.num_classes, 5)
        # Check default persistence
        self.assertEqual(cfg.backbone.name, "mobilenetv4_conv_small.e2400_r224_in1k")

    def test_validation_error(self):
        # Pydantic < 2.0 or > 2.0? Assuming basic validation
        # E.g. providing string for int
        with self.assertRaises(Exception):
            AppConfig(data={"batch_size": "invalid"})

if __name__ == '__main__':
    unittest.main()
