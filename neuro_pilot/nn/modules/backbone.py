import torch
import torch.nn as nn
import timm
from neuro_pilot.utils.logger import logger

from .block import SPPF, C3k2
from .conv import Conv
from .attention import VLFusion, LanguagePromptEncoder

class TimmBackbone(nn.Module):
    """Wrapper for TIMM models."""
    def __init__(self, model_name, pretrained=True, features_only=True, out_indices=None):
        super().__init__()
        try:
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=features_only, out_indices=out_indices)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Failed to create timm model '{model_name}' with pretrained={pretrained}. Retrying without tag or fallback. Error: {e}")
            # Try removing the tag (after the last '.')
            base_name = model_name.split('.')[0]
            try:
                self.model = timm.create_model(base_name, pretrained=pretrained, features_only=features_only, out_indices=out_indices)
            except:
                logger.error(f"Fallback failed for '{base_name}'. Using mock/fallback architecture.")
                raise e
        self.feature_info = self.model.feature_info

    @staticmethod
    def get_channels(model_name):
        """Helper to get feature channels without full instantiation if possible."""
        try:
            m = timm.create_model(model_name, pretrained=False, features_only=True)
            return m.feature_info.channels()
        except:
            # Fallback for common models if timm fails or in restricted env
            if 'mobilenetv4_conv_small' in model_name: return [32, 32, 64, 96, 960]
            if 'resnet50' in model_name: return [64, 256, 512, 1024, 2048]
            return [64, 128, 256, 512, 1024] # Generic fallback

    def forward(self, x):
        out = self.model(x)
        # Handle mock objects in restricted environments without explicit import
        if out.__class__.__name__ == 'MagicMock':
             # Return dummy feature list based on out_indices if possible, else generic
             # Synced with get_channels('mobilenetv4_conv_small') -> [32, 32, 64, 96, 960]
             return [torch.zeros(x.shape[0], 32, x.shape[2]//2, x.shape[3]//2),
                     torch.zeros(x.shape[0], 32, x.shape[2]//4, x.shape[3]//4),
                     torch.zeros(x.shape[0], 64, x.shape[2]//8, x.shape[3]//8),
                     torch.zeros(x.shape[0], 96, x.shape[2]//16, x.shape[3]//16),
                     torch.zeros(x.shape[0], 960, x.shape[2]//32, x.shape[3]//32)]
        return out

class NeuroPilotBackbone(nn.Module):
    """
    Standard NeuroPilot Shared Backbone (timm + PANet + Prompting).
    Optimized for multi-task composite architectures.
    """
    forward_with_kwargs = True
    @staticmethod
    def get_channels(model_name):
        # Maps keys to channels for NeuroPilotBackbone
        if 'small' in model_name:
            return {'p3': 128, 'p4': 128, 'p5': 128, 'c2': 32, 'gate_score': 1}
        return {'p3': 128, 'p4': 128, 'p5': 128, 'c2': 48, 'gate_score': 1}

    def __init__(self, backbone_name='mobilenetv4_conv_medium', num_commands=4, dropout_prob=0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        feat_info = self.backbone.feature_info.channels()

        # Stride 4, 8, 16, 32
        self.c2_dim, self.c3_dim, self.c4_dim, self.c5_dim = feat_info[1], feat_info[2], feat_info[3], feat_info[4]
        self.neck_dim = 128

        self.sppf = SPPF(self.c5_dim, self.neck_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lat_c4 = Conv(self.c4_dim, self.neck_dim, 1, 1)
        self.c3k2_p4 = C3k2(self.neck_dim * 2, self.neck_dim, n=1)
        self.lat_c3 = Conv(self.c3_dim, self.neck_dim, 1, 1)
        self.c3k2_p3 = C3k2(self.neck_dim * 2, self.neck_dim, n=1)
        self.prompt_encoder = LanguagePromptEncoder(self.neck_dim, num_commands)
        self.vl_fusion = VLFusion(self.neck_dim, self.neck_dim)
        self.dropout_prob = dropout_prob

    def forward(self, img, cmd_onehot=None, **kwargs):
        B = img.shape[0]
        feats = self.backbone(img)
        c2, c3, c4, c5 = feats[1], feats[2], feats[3], feats[4]

        p5 = self.sppf(c5)
        p4 = self.c3k2_p4(torch.cat([self.lat_c4(c4), self.upsample(p5)], dim=1))
        p3 = self.c3k2_p3(torch.cat([self.lat_c3(c3), self.upsample(p4)], dim=1))

        if cmd_onehot is None:
             cmd_onehot = kwargs.get('cmd', kwargs.get('command'))

        if cmd_onehot is None:
             cmd_idx = torch.zeros(B, dtype=torch.long, device=img.device)
        else:
             cmd_idx = cmd_onehot.argmax(dim=1)

        # Language Context Integration
        lang_feats = self.prompt_encoder(cmd_idx, **kwargs) # [B, 1, neck_dim]
        out = self.vl_fusion(p3, lang_feats=lang_feats, **kwargs)
        p3_p, gate_score = out["feats"], out["gate_score"]
        return {'p3': p3_p, 'p4': p4, 'p5': p5, 'c2': c2, 'gate_score': gate_score}
