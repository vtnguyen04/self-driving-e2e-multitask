from neuro_pilot.engine.task import BaseTask, TaskRegistry
import torch.nn as nn
from neuro_pilot.nn.modules import TrajectoryHead

@TaskRegistry.register("trajectory")
class TrajectoryTask(BaseTask):
    def build_model(self) -> nn.Module:
        if self.backbone:
            self.model = TrajectoryHead(c1=128, num_waypoints=self.cfg.head.num_waypoints)
            return self.model
        else:
            raise NotImplementedError("Standalone TrajectoryTask not yet supported (needs backbone)")

    def build_criterion(self) -> nn.Module:
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
             c2 = getattr(self.backbone, 'c2_dim', 512)
             self.model = HeatmapHead(c1=[128, c2], ch_out=1)
             return self.model
        raise NotImplementedError

    def build_criterion(self) -> nn.Module:
         from neuro_pilot.utils.losses import HeatmapLossAtomic
         return HeatmapLossAtomic(self.cfg)

    def get_trainer(self): pass
    def get_validator(self):
        return None

from neuro_pilot.nn.modules import HeatmapHead
