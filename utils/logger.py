"""
Custom logging setup for the project.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, output_dir: Path):
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory for log files
    
    Returns:
        Logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / "logs" / f"{experiment_name}_{timestamp}.log"
    return setup_logger(experiment_name, str(log_file))

