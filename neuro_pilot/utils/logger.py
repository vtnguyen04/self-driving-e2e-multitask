import sys
import platform
import torch
from loguru import logger
from pathlib import Path

def set_logger(name="NeuroPilot", save_dir=None, endpoint=None):
    """
    Configure Loguru logger with custom format mimicking Ultralytics.
    """
    logger.remove()

    fmt = "<level>{message}</level>"

    logger.add(sys.stderr, format=fmt, level="INFO", colorize=True)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_file = save_dir / "neuropilot.log"
        logger.add(log_file, format=fmt, level="DEBUG", rotation="10 MB")

    return logger

def log_system_info():
    """Logs system information similar to Ultralytics startup."""
    try:
        os_info = f"{platform.system()} {platform.release()}"
        python_info = f"Python-{platform.python_version()}"

        torch_info = f"torch-{torch.__version__}"
        cuda_available = torch.cuda.is_available()
        cuda_info = "CUDA:? (Unknown)"
        if cuda_available:
            devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            cuda_info = f"CUDA:{torch.version.cuda} ({', '.join(devices)})"
        else:
            cuda_info = "CPU"

        logger.info(f"System: {os_info}")
        logger.info(f"Environment: {python_info} {torch_info} {cuda_info}")

    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")

def colorstr(*args):
    """
    Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code.
    Example:
        colorstr('blue', 'hello world')
        colorstr('bold', 'hello world')
        colorstr('blue', 'bold', 'hello world')
    """
    *colors_list, string = args if len(args) > 1 else ("blue", "bold", args[0])
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in colors_list) + f"{string}" + colors["end"]

set_logger()

__all__ = ["logger", "set_logger", "log_system_info", "colorstr"]
