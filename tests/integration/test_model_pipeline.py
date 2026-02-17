
import unittest
import torch
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.data import prepare_dataloaders

class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        self.model_cfg_path = "neuro_pilot/cfg/models/yolo_all_tasks.yaml"
        self.overrides = {
            "data": {
                "dataset_yaml": "data_v1/data.yaml",
                "batch_size": 4,
                "image_size": 640,
                "num_workers": 0,
                "augment": {"mosaic": 0.0}
            },
            "trainer": {
                "max_epochs": 1,
                "use_amp": False 
            }
        }
        self.model = NeuroPilot(self.model_cfg_path, **self.overrides)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_forward_backward_pass(self):
        """
        Kiểm tra toàn bộ luồng từ Dataloader -> Model -> Loss -> Backward.
        Đảm bảo dữ liệu chuẩn bị trong data_v1 khớp hoàn toàn với kiến trúc mô hình.
        """
        print("\n--- Testing Model Pipeline with Real Data ---")
        
        # 1. Load Data
        train_loader, _ = prepare_dataloaders(self.model.cfg_obj)
        batch = next(iter(train_loader))
        
        # Chuyển batch sang device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        # 2. Forward Pass
        print(f"Executing forward pass on {self.device}...")
        # Lấy raw predictions từ model underlying
        preds = self.model.model(batch['image'], cmd=batch['command'])
        
        self.assertIsNotNone(preds, "Model returned None for predictions")
        
        # 3. Loss Calculation
        print("Building criterion and calculating Loss components...")
        self.model.task_wrapper.build_criterion() # Ensure criterion is built
        loss_dict = self.model.task_wrapper.criterion.advanced(preds, batch)
        total_loss = loss_dict['total']
        
        # Monitor the Gate!
        if 'gate_score' in preds:
            gate_val = preds['gate_score'].mean().item()
            print(f"🔍 Current Gate Score (Importance of Command): {gate_val:.4f}")
            if gate_val > 0.8: print("   -> Model is heavily RELYING on the Command.")
            elif gate_val < 0.2: print("   -> Model is ignoring the Command (Vision dominant).")
            else: print("   -> Model is balancing Vision and Command.")

        print(f"✅ Total Loss: {total_loss.item():.4f}")
        print(f"✅ Loss components: {loss_dict}")
        
        self.assertFalse(torch.isnan(total_loss), "Loss is NaN!")
        self.assertGreater(total_loss.item(), 0, "Loss should be greater than 0")

        # 4. Backward Pass
        print("Executing backward pass...")
        total_loss.backward()
        print("✅ Backward pass successful.")

if __name__ == "__main__":
    unittest.main()
