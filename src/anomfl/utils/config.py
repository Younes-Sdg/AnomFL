"""
Configuration management for AnomFL.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# Alias for backward compatibility
get_config = load_config


def get_data_path() -> str:
    """Get the data directory path."""
    return os.path.join(os.getcwd(), "data")


def get_output_path() -> str:
    """Get the output directory path."""
    return os.path.join(os.getcwd(), "outputs")


def get_models_path() -> str:
    """Get the models directory path."""
    return os.path.join(os.getcwd(), "models")


def get_logs_path() -> str:
    """Get the logs directory path."""
    return os.path.join(os.getcwd(), "logs")


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        get_data_path(),
        get_output_path(),
        get_models_path(),
        get_logs_path(),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True) 