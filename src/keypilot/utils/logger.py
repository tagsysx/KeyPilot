"""
Logging utilities for KeyPilot project.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> None:
    """
    Set up logger with file and console handlers.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional custom log file name
        rotation: Log file rotation size
        retention: How long to keep log files
        
    Raises:
        ValueError: If log_level is invalid
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")
    
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level.upper(),
        colorize=True,
    )
    
    # Add file handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    if log_file is None:
        log_file = "keypilot_{time:YYYY-MM-DD}.log"
    
    logger.add(
        log_path / log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level.upper(),
        rotation=rotation,
        retention=retention,
        compression="zip",
    )
    
    logger.info(f"Logger initialized with level {log_level}")

