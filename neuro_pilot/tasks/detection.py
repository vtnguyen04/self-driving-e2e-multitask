from neuro_pilot.engine.task import BaseTask, TaskRegistry
import torch.nn as nn
import torch
from neuro_pilot.nn.modules import Detect

@TaskRegistry.register("detect")
class DetectionTask(BaseTask):
    def build_model(self) -> nn.Module:
        if self.backbone:
            nc = self.cfg.head.num_classes
            det_head = Detect(nc=nc, ch=(128, 128, 128)) # Assuming 128 neck dim
            det_head.stride = torch.tensor([8., 16., 32.])
            self.model = det_head
            return det_head
        raise NotImplementedError

    def build_criterion(self) -> nn.Module:
        from neuro_pilot.utils.losses import DetectionLossAtomic
        # Pass both backbone (for features) and head (to avoid redundant init)
        return DetectionLossAtomic(self.backbone, self.model, self.cfg)

    def get_validator(self):
        from neuro_pilot.utils.metrics import DetectionMetric
        # Pass the HEAD to the metric, not the whole model, or handle inside
        # DetectionMetric expects 'model_head' to wrap for 'decoder'
        # The 'self.model' in the task is the Head.
        return DetectionMetric(self.cfg, 'cuda', self.model)

    def get_trainer(self):
        """
        Return the trainer for this task.
        For atomic tasks used in composition, this might not be used directly
        if the CompositeTrainer handles the loop.
        """
        return None
