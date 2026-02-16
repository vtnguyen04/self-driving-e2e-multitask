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
        # Interpolate g to match x size if needed
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
    - If scene is simple (straight road), Gate -> 0. Command is suppressed.
    - If scene is complex (intersection), Gate -> 1. Command is active.
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
        x_gap = self.gap(x).view(B, C)
        gate = self.fc(x_gap).view(B, 1, 1)
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

        # Context Gate to filter command relevance
        self.gate = CommandGate(c1)

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
        x_flat = vision.flatten(2).permute(0, 2, 1) # [B, HW, C]

        # 1. Calculate Context Relevance (Gate)
        gate_score = self.gate(vision) # [B, 1, 1]

        # 2. Cross-Attention
        attn_out, _ = self.mha(self.q(x_flat), self.k(lang_feats), self.v(lang_feats))

        # 3. Gated Residual Connection
        x_flat = self.norm(x_flat + gate_score * attn_out)

        vision = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return {"feats": vision, "gate_score": gate_score}

class LanguagePromptEncoder(nn.Module): # Renamed back for compatibility
    """Semantic mapping for commands using cached CLIP embeddings with synonym support."""
    forward_with_kwargs = True
    def __init__(self, embed_dim=128, num_prompts=10, mode='embedding', clip_dim=512): # Added mode and clip_dim
        super().__init__()
        self.mode = mode # Store mode
        if mode == 'clip':
            # Frozen semantic anchors with Synonym Augmentation
            # We store [num_prompts, max_synonyms, clip_dim]
            # For 4 commands, we simulate synonyms to force the model to learn the "semantic cluster"
            # rather than a single point.
            # Example:
            # 0 (Straight): "go straight", "forward", "keep lane"
            # 1 (Left): "turn left", "take a left", "steer left"
            max_synonyms = 5
            self.register_buffer('cached_embeds', torch.randn(num_prompts, max_synonyms, clip_dim))
            self.max_synonyms = max_synonyms

            # Learnable projection to align CLIP space with Vision space
            self.projector = nn.Sequential(
                nn.Linear(clip_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.embedding = nn.Embedding(num_prompts, embed_dim)
            # Dictionary mapping for semantic clarity (kept for non-CLIP mode, or could be removed if CLIP is default)
            self.prompt_map = {
                0: "go straight on drivable area",
                1: "turn left at intersection",
                2: "turn right at intersection",
                3: "stop and wait at line"
            }


    def forward(self, x, indices=None, **kwargs):
        """
        Args:
            x (Tensor): Ignored (from previous layer)
            indices (Tensor): [B] batch of prompt indices (if from kwargs)
        Returns:
            Tensor: [B, 1, embed_dim]
        """
        if indices is None:
            if 'cmd_onehot' in kwargs:
                indices = kwargs['cmd_onehot'].argmax(dim=1)
            elif 'command_idx' in kwargs:
                indices = kwargs['command_idx']
            else:
                B = x.shape[0] if hasattr(x, 'shape') else 1
                indices = torch.zeros(B, dtype=torch.long, device=getattr(x, 'device', 'cpu'))
        if self.mode == 'clip':
            # 1. Select Synonyms (Training Augmentation)
            if self.training:
                # Randomly pick a synonym index for each sample in batch
                syn_idx = torch.randint(0, self.max_synonyms, (indices.shape[0],), device=indices.device)
            else:
                # Inference: Use canonical prompt (index 0) or average
                syn_idx = torch.zeros_like(indices)

            # Gather semantics: [B, clip_dim]
            # Advanced indexing to pick specific synonym for each batch item
            raw_embeds = self.cached_embeds[indices, syn_idx]

            # 2. Project to vision dimension
            x = self.projector(raw_embeds) # [B, embed_dim]
        else:
            x = self.embedding(indices) # [B, embed_dim]

        return x.unsqueeze(1)
