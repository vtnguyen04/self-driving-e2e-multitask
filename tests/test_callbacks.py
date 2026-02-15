
import unittest
from unittest.mock import MagicMock
from pathlib import Path
import shutil
import tempfile
from neuro_pilot.engine.callbacks import CallbackList, CheckpointCallback, LoggingCallback

class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_callback_list_trigger(self):
        cb1 = MagicMock()
        cb2 = MagicMock()
        cbl = CallbackList([cb1, cb2])

        trainer = MagicMock()
        cbl.on_epoch_start(trainer)

        cb1.on_epoch_start.assert_called_once_with(trainer)
        cb2.on_epoch_start.assert_called_once_with(trainer)

    def test_checkpoint_callback_logic(self):
        # Mock Trainer
        trainer = MagicMock()
        trainer.epoch = 1
        trainer.val_loss = 0.5

        # Mock Config
        cfg = MagicMock()
        cfg.trainer.checkpoint_top_k = 2

        cb = CheckpointCallback(self.test_dir, cfg)

        # Epoch 1: Loss 0.5 (Best)
        cb.on_val_end(trainer)
        trainer.save_checkpoint.assert_called()
        self.assertEqual(cb.best_loss, 0.5)
        self.assertEqual(len(cb.top_k), 1)

        # Epoch 2: Loss 0.3 (New Best)
        trainer.epoch = 2
        trainer.val_loss = 0.3
        cb.on_val_end(trainer)
        self.assertEqual(cb.best_loss, 0.3)
        self.assertEqual(len(cb.top_k), 2)

        # Epoch 3: Loss 0.6 (Worse than 0.5 and 0.3, but top_k=2)
        # Should replace 0.5? No, top_k should keep lowest.
        # Current top_k: [(0.3, 2, path), (0.5, 1, path)]
        # New: 0.6.
        # 0.6 > 0.5 (worst). Should NOT be added.
        trainer.epoch = 3
        trainer.val_loss = 0.6
        cb.on_val_end(trainer)
        self.assertEqual(len(cb.top_k), 2)
        self.assertEqual(cb.top_k[-1][0], 0.5) # Worst kept is 0.5

    def test_logging_callback(self):
        logger = MagicMock()
        cb = LoggingCallback(logger)
        trainer = MagicMock()
        trainer.batch_metrics = {'loss': 0.1}
        trainer.batch_size = 32

        cb.on_batch_end(trainer)
        logger.log_batch.assert_called_with({'loss': 0.1})

if __name__ == '__main__':
    unittest.main()
