import torch.nn as nn
from neuro_pilot.core.registry import Registry, register_backbone, register_head

def test_registry_decorators():
    # 1. Test Backbone Registration
    @register_backbone("TestBackbone")
    class TestBackbone(nn.Module):
        pass

    assert "TestBackbone" in Registry._BACKBONES
    assert Registry.get_backbone("TestBackbone") == TestBackbone
    assert Registry.get("TestBackbone") == TestBackbone

    # 2. Test Head Registration
    @register_head("TestHead")
    class TestHead(nn.Module):
        pass

    assert "TestHead" in Registry._HEADS
    assert Registry.get_head("TestHead") == TestHead
    assert Registry.get("TestHead") == TestHead

def test_registry_lookup():
    # Test fallback
    assert Registry.get("NonExistent") is None

def test_duplicate_registration_warning(caplog):
    @register_backbone("DuplicateBackbone")
    class B1(nn.Module): pass

    @register_backbone("DuplicateBackbone")
    class B2(nn.Module): pass

    assert "already registered" in caplog.text
    # Should overwrite
    assert Registry.get("DuplicateBackbone") == B2
