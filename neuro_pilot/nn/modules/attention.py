import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Simple Attention Gate for multi-modal feature fusion."""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class CommandGate(nn.Module):
    """
    Context-Adaptive Gating Mechanism.
    Predicts a scalar importance weight [0, 1] for the command based on vision features.

    Logic:
    - If scene is simple, Gate -> 0.
    - If scene is complex, Gate -> 1.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Vision features [B, C, H, W]
        Returns:
            Tensor: [B, 1, 1] gating weight
        """
        B, C, H, W = x.shape
        try:
            with torch.amp.autocast(device_type=x.device.type if hasattr(x.device, 'type') else 'cuda', enabled=False):
                x_f32 = x.float()
                x_f32 = torch.clamp(torch.nan_to_num(x_f32, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
                x_gap = self.gap(x_f32).view(B, C)

                dtype = self.fc[0].weight.dtype
                gate = self.fc(x_gap.to(dtype)).view(B, 1, 1)

                gate = torch.clamp(torch.nan_to_num(gate, nan=0.5), 0.0, 1.0)
                return gate.to(x.dtype)
        except:
             x_gap = self.gap(x).view(B, C)
             dtype = self.fc[0].weight.dtype
             gate = self.fc(x_gap.to(dtype)).view(B, 1, 1)
             return gate

class VLFusion(nn.Module):
    """Vision-Language Fusion module using Cross-Attention and Context Gating.
    Inspired by YOLO-World and SAM.
    """
    forward_with_kwargs = True
    def __init__(self, c1, c2, num_heads=4):
        super().__init__()
        self.q = nn.Linear(c1, c1)
        self.k = nn.Linear(c2, c1)
        self.v = nn.Linear(c2, c1)
        self.mha = nn.MultiheadAttention(c1, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(c1)

        self.gate = CommandGate(c1)

        self.resid_gain = nn.Parameter(torch.ones(1))

    def forward(self, x, lang_feats=None, **kwargs):
        """
        Args:
            x (Tensor): Vision features [B, C, H, W]
            lang_feats (Tensor): Language features [B, L, C] (if from previous layer)
        """
        if isinstance(x, list):
            vision = x[0]
            if len(x) > 1 and lang_feats is None: lang_feats = x[1]
        else:
            vision = x

        if lang_feats is None and 'lang_feats' in kwargs:
            lang_feats = kwargs['lang_feats']

        B, C, H, W = vision.shape
        x_flat = vision.flatten(2).permute(0, 2, 1)

        gate_score = self.gate(vision)

        dtype = self.q.weight.dtype
        x_flat = x_flat.to(dtype)
        lang_feats = lang_feats.to(dtype)

        attn_out, _ = self.mha(self.q(x_flat), self.k(lang_feats), self.v(lang_feats))

        x_flat = self.norm(x_flat + self.resid_gain * gate_score * attn_out)
        x_flat = torch.nan_to_num(x_flat, nan=0.0)

        vision = x_flat.permute(0, 2, 1).reshape(B, C, H, W).to(x.dtype if hasattr(x, 'dtype') else torch.float32)
        return {"feats": vision, "gate_score": gate_score}

class CFRBridge(nn.Module):
    """
    Causal Feature Router Bridge.
    Enforces asymmetric causal graph: Perception -> Planning.
    Takes Planning features as Query, and Perception features as Key/Value.
    CRITICAL: Applies stop_gradient to Perception features to prevent
    Planning task from corrupting the Perception backbone.
    """
    forward_with_kwargs = True
    def __init__(self, c_plan, c_percept, num_heads=4):
        super().__init__()
        self.q = nn.Linear(c_plan, c_plan)
        self.k = nn.Linear(c_percept, c_plan)
        self.v = nn.Linear(c_percept, c_plan)

        self.mha = nn.MultiheadAttention(c_plan, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(c_plan)
        self.resid_gain = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        """
        x: List containing [Planning_Features, Perception_Features]
        """
        if isinstance(x, list) and len(x) >= 2:
            feat_plan = x[0]
            feat_percept = x[1]
        else:
            raise ValueError("CFRBridge requires a list of [Planning_Features, Perception_Features]")

        B, C_p, H_p, W_p = feat_plan.shape
        _, C_d, H_d, W_d = feat_percept.shape

        plan_flat = feat_plan.flatten(2).permute(0, 2, 1)
        percept_flat = feat_percept.flatten(2).permute(0, 2, 1)

        percept_causal = percept_flat.detach()

        dtype = self.q.weight.dtype
        plan_flat = plan_flat.to(dtype)
        percept_causal = percept_causal.to(dtype)

        q = self.q(plan_flat)
        k = self.k(percept_causal)
        v = self.v(percept_causal)

        attn_out, _ = self.mha(q, k, v)

        out_flat = self.norm(plan_flat + self.resid_gain * attn_out)
        out = out_flat.permute(0, 2, 1).reshape(B, C_p, H_p, W_p)

        return out

class LanguagePromptEncoder(nn.Module):
    """Semantic mapping for commands using cached CLIP embeddings with synonym support."""
    forward_with_kwargs = True
    def __init__(self, embed_dim=128, num_prompts=10, mode='embedding', clip_dim=512):
        super().__init__()
        self.mode = mode
        if mode == 'clip':
            max_synonyms = 5
            self.register_buffer('cached_embeds', torch.randn(num_prompts, max_synonyms, clip_dim))
            self.max_synonyms = max_synonyms

            self.projector = nn.Sequential(
                nn.Linear(clip_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.embedding = nn.Embedding(num_prompts, embed_dim)
            self.prompt_map = {
                0: "go straight on drivable area",
                1: "turn left at intersection",
                2: "turn right at intersection",
                3: "stop and wait at line"
            }

    def forward(self, x, indices=None, **kwargs):
        """
        Args:
            x (Tensor): Vision features or Command indices
            indices (Tensor): [B] batch of prompt indices (if from kwargs)
        Returns:
            Tensor: [B, 1, embed_dim]
        """
        if indices is None:
            if 'cmd' in kwargs:
                indices = kwargs['cmd']
            elif 'command_idx' in kwargs:
                indices = kwargs['command_idx']
            elif 'cmd_onehot' in kwargs:
                indices = kwargs['cmd_onehot'].argmax(dim=1)
            elif isinstance(x, torch.Tensor) and x.dtype in {torch.long, torch.int}:
                indices = x
            else:
                B = x.shape[0] if hasattr(x, 'shape') else 1
                indices = torch.zeros(B, dtype=torch.long, device=getattr(x, 'device', 'cpu'))

        if torch.is_tensor(indices) and indices.dim() > 1:
            if indices.shape[-1] == 4:
                indices = indices.argmax(dim=-1)
            else:
                indices = indices.view(-1)
        if self.mode == 'clip':
            if self.training:
                syn_idx = torch.randint(0, self.max_synonyms, (indices.shape[0],), device=indices.device)
            else:
                syn_idx = torch.zeros_like(indices)

            raw_embeds = self.cached_embeds[indices, syn_idx]

            x = raw_embeds.to(self.projector[0].weight.dtype)
            for layer in self.projector:
                x = layer(x)
                if torch.is_floating_point(x):
                    x = x.to(self.projector[-1].weight.dtype)
        else:
            x = self.embedding(indices)

        return x.unsqueeze(1)
