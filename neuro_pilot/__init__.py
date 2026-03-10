# NeuroPilot: Unified E2E Autonomous Driving Framework
# Copyright (c) 2026 Vo Thanh Nguyen. All rights reserved.

__version__ = "1.0.0"

from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.engine.results import Results
from neuro_pilot.utils.checks import check_version

# Export common classes for easy access: from neuro_pilot import NeuroPilot
__all__ = ["NeuroPilot", "Results", "check_version"]
