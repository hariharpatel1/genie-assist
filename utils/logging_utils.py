"""
Logging utilities for the onboarding agent.
"""
import logging
import os
from datetime import datetime
from typing import Optional

from config.settings import settings

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    module_name: str = "onboarding_agent"
):
    """
    Set up logging configuration.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses a default path.
        module_name: Module name for the logger
    
    Returns:
        Configured logger
    """
    level = log_level or settings.LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    if not log_file:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/{module_name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Create and return module-specific logger
    logger = logging.getLogger(module_name)
    logger.setLevel(numeric_level)
    
    return logger


class LoggingContext:
    """
    Context manager for temporarily changing log level.
    """
    
    def __init__(self, logger, level=None):
        """
        Initialize the context manager.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.level = level
        self.old_level = logger.level
    
    def __enter__(self):
        """Set temporary log level."""
        if self.level is not None:
            self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log level."""
        self.logger.setLevel(self.old_level)