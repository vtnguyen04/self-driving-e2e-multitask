
import unittest
import subprocess
import os
import sys

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = f"{os.getcwd()}:{self.env.get('PYTHONPATH', '')}"
        self.python = sys.executable

    def test_cli_help(self):
        result = subprocess.run([self.python, "neuro_pilot/entrypoint.py", "--help"],
                                capture_output=True, text=True, env=self.env)
        self.assertEqual(result.returncode, 0)
        self.assertIn("NeuroPilot CLI", result.stdout)

    def test_cli_benchmark(self):
        # Run a quick benchmark (shortened)
        # Pointing to config to avoid loading huge weights
        cmd = [self.python, "neuro_pilot/entrypoint.py", "benchmark",
               "--model", "neuro_pilot/cfg/models/yolo_style.yaml",
               "--imgsz", "32", "--batch", "1"]

        # Force CPU
        env = self.env.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Benchmark Results", result.stdout)

    def test_cli_export_dry_run(self):
        # We don't want to actually export a full model in tests if possible,
        # but let's check if the subcommand is recognized.
        cmd = [self.python, "neuro_pilot/entrypoint.py", "export", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=self.env)
        self.assertEqual(result.returncode, 0)
        self.assertIn("export", result.stdout.lower())

if __name__ == '__main__':
    unittest.main()
