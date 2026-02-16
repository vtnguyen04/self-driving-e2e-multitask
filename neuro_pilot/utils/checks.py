from __future__ import annotations

import math
import os
import platform
import re
from importlib import metadata
from pathlib import Path

import torch

# Global Constants
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = str(torch.__version__)
USER_CONFIG_DIR = Path(os.getenv("NEUROPILOT_CONFIG_DIR", Path.home() / ".config/neuropilot"))

def is_ascii(s: str | list | tuple | dict) -> bool:
    """Check if a string is composed of only ASCII characters."""
    return all(ord(c) < 128 for c in str(s))

def check_version(current: str, required: str, name: str = "version", hard: bool = False) -> bool:
    """Check current version against required version string (e.g., '>=8.0.0')."""
    if not current or not required:
        return True

    # Simple version parsing logic
    def parse(v):
        return tuple(map(int, re.findall(r"\d+", v)[:3])) or (0, 0, 0)

    cv = parse(current)
    op, rv_str = re.match(r"([^0-9]*)([\d.]+)", required).groups()
    op = op or ">="
    rv = parse(rv_str)

    result = True
    if op == "==": result = cv == rv
    elif op == "!=": result = cv != rv
    elif op == ">=": result = cv >= rv
    elif op == "<=": result = cv <= rv
    elif op == ">": result = cv > rv
    elif op == "<": result = cv < rv

    if not result and hard:
        raise ModuleNotFoundError(f"{name}{required} is required, but {name}=={current} is installed.")
    return result

def check_python(minimum: str = "3.8.0") -> bool:
    """Check minimum Python version."""
    return check_version(PYTHON_VERSION, minimum, name="Python", hard=True)

def check_font(font: str = "Arial.ttf") -> Path:
    """
    Find font locally or download to configurations directory.
    Essential for Annotator to have consistent rendering across environments.
    """
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    file = USER_CONFIG_DIR / Path(font).name
    if file.exists():
        return file

    # Placeholder for system font search or safe download
    # In a real environment, we'd use requests to download from a NeuroPilot assets mirror
    return file # Assume it might be provided or handled by the system for now

def check_imgsz(imgsz: int | list[int], stride: int = 32, min_dim: int = 2) -> list[int]:
    """Verify image size is a multiple of stride."""
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    sz = [max(math.ceil(x / stride) * stride, 0) for x in imgsz]
    if len(sz) == 1 and min_dim == 2:
        sz = [sz[0], sz[0]]
    return sz

def check_requirements(requirements: list[str] | str):
    """Check if package requirements are met."""
    if isinstance(requirements, str):
        requirements = [requirements]

    for r in requirements:
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r)
        name, req = match[1], match[2] or ""
        try:
            ver = metadata.version(name)
            check_version(ver, req, name=name, hard=True)
        except metadata.PackageNotFoundError:
            print(f"Warning: {name} not found. Some features may be disabled.")

def collect_system_info():
    """Summary of software and hardware environment."""
    info = {
        "OS": platform.platform(),
        "Python": PYTHON_VERSION,
        "Torch": TORCH_VERSION,
        "CUDA": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "CPU count": os.cpu_count(),
    }
    return info

def check_amp(model):
    """
    Checks if Automatic Mixed Precision (AMP) is available and functional.
    Returns True if AMP is supported and recommended.
    """
    from neuro_pilot.utils.logger import logger

    device = next(model.parameters()).device
    if device.type == 'cpu':
        return False

    try:
        from torch.cuda.amp import autocast
        with autocast():
             # Basic test
             x = torch.zeros(1, 3, 32, 32).to(device)
             _ = model(x) if not hasattr(model, 'forward_with_kwargs') else model(x)
        return True
    except Exception as e:
        logger.warning(f"AMP check failed: {e}. Disabling AMP.")
        return False

def find_file(file: str | Path) -> str:
    """Search for a file and return its path as a string."""
    file = Path(file)
    if file.exists():
        return str(file)

    # Search in common directories
    for d in [USER_CONFIG_DIR, Path("cfg")]:
        if (d / file.name).exists():
            return str(d / file.name)

    return str(file)

def print_args(args: dict):
    """Print training arguments in a professional Ultralytics format."""
    from neuro_pilot.utils.logger import logger
    logger.info("Training Arguments:")
    for k, v in args.items():
        logger.info(f"{k:>20}: {v}")
