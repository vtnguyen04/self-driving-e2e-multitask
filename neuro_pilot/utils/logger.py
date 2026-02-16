import os
import sys
import logging
from loguru import logger
from pathlib import Path

def set_logger(name="NeuroPilot", save_dir=None, endpoint=None):
    """
    Configure Loguru logger with custom format mimicking Ultralytics.
    """
    # Remove default handler
    logger.remove()

    # Define Format
    # Ultralytics style: "2024-01-01 12:00:00 [INFO] NeuroPilot: Message"
    # But usually just: "Ultralytics YOLOv8.1.0 🚀 Python-3.10.12 torch-2.1.0..."
    # We want "NeuroPilot 🚀 ... [INFO] Message"

    # Custom format
    # <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>NeuroPilot 🚀</cyan> - <level>{message}</level>"

    # Add Console Handler
    logger.add(sys.stderr, format=fmt, level="INFO", colorize=True)

    # Add File Handler if save_dir
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_file = save_dir / "neuropilot.log"
        logger.add(log_file, format=fmt, level="DEBUG", rotation="10 MB")

    return logger

# Create a default instance
# To match 'logging.getLogger(__name__)' usage, we can just expose 'logger'
# But 'logger' is a singleton in loguru.
# We can just export it.

__all__ = ["logger", "set_logger"]
