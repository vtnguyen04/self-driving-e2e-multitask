from neuro_pilot.engine.task import BaseTask, TaskRegistry
import torch.nn as nn
from neuro_pilot.nn.modules import TrajectoryHead

@TaskRegistry.register("trajectory")
class TrajectoryTask(BaseTask):
    def build_model(self) -> nn.Module:
        # If backbone provided, return Head
        if self.backbone:
            # Assuming neck_dim (128) is the output of the backbone/neck passed to the head
            self.model = TrajectoryHead(c1=128, num_waypoints=self.cfg.head.num_waypoints)
            return self.model
        else:
            raise NotImplementedError("Standalone TrajectoryTask not yet supported (needs backbone)")

    def build_criterion(self) -> nn.Module:
        # Atomic Trajectory Loss
        from neuro_pilot.utils.losses import TrajectoryLossAtomic
        return TrajectoryLossAtomic(self.cfg)

    def get_trainer(self): pass
    def get_validator(self):
        from neuro_pilot.utils.metrics import TrajectoryMetric
        return TrajectoryMetric()

@TaskRegistry.register("heatmap")
class HeatmapTask(BaseTask):
    def build_model(self) -> nn.Module:
        if self.backbone:
             c2 = getattr(self.backbone, 'c2_dim', 512) # fallback to common resnet dim
             # HeatmapHead expects [p3, c2], so c1 should be a list/tuple of [p3_dim, c2_dim]
             # Assuming neck_dim (128) for p3 and c2 from backbone
             self.model = HeatmapHead(c1=[128, c2], ch_out=1)
             return self.model
        raise NotImplementedError

    def build_criterion(self) -> nn.Module:
         from neuro_pilot.utils.losses import HeatmapLossAtomic
         return HeatmapLossAtomic(self.cfg)

    def get_trainer(self): pass
    def get_validator(self):
        # Heatmap usually just logs loss, but we could add Dice/IoU if needed.
        # For now, let's return None or a dummy metric
        return None

# Helper imports
from neuro_pilot.nn.modules import HeatmapHead
