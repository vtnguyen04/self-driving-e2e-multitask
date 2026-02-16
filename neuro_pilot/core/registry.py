import logging
from typing import Dict, Type

logger = logging.getLogger(__name__)

class Registry:
    """
    Central Registry for NeuroPilot modules.
    Enables OCP (Open-Closed Principle) by allowing registration of new
    Backbones, Heads, and Losses without modifying core code.
    """
    _BACKBONES: Dict[str, Type] = {}
    _HEADS: Dict[str, Type] = {}
    _LOSSES: Dict[str, Type] = {}
    _NECKS: Dict[str, Type] = {}

    @classmethod
    def register_backbone(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            if key in cls._BACKBONES:
                logger.warning(f"Backbone {key} already registered. Overwriting.")
            cls._BACKBONES[key] = obj
            return obj
        return decorator

    @classmethod
    def register_head(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            cls._HEADS[key] = obj
            return obj
        return decorator

    @classmethod
    def register_loss(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            cls._LOSSES[key] = obj
            return obj
        return decorator

    @classmethod
    def register_neck(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            cls._NECKS[key] = obj
            return obj
        return decorator

    @classmethod
    def get_backbone(cls, name: str) -> Type:
        return cls._BACKBONES.get(name)

    @classmethod
    def get_head(cls, name: str) -> Type:
        return cls._HEADS.get(name)

    @classmethod
    def get_loss(cls, name: str) -> Type:
        return cls._LOSSES.get(name)

    @classmethod
    def get_neck(cls, name: str) -> Type:
        return cls._NECKS.get(name)

    @classmethod
    def get(cls, name: str) -> Type:
        """General lookup across all categories."""
        if name in cls._BACKBONES: return cls._BACKBONES[name]
        if name in cls._HEADS: return cls._HEADS[name]
        if name in cls._NECKS: return cls._NECKS[name]
        if name in cls._LOSSES: return cls._LOSSES[name]
        return None

# Alias for ease of use
register_backbone = Registry.register_backbone
register_head = Registry.register_head
register_loss = Registry.register_loss
register_neck = Registry.register_neck
