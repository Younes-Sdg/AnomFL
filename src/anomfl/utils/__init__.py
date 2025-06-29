"""
Utility functions for AnomFL.

This module contains common functionality used across the project:
- Configuration management
- Logging setup
- Data utilities
- Path management
- Fixed scaling utilities
"""

from .config import load_config, get_data_path, get_output_path
from .logging import setup_logging, get_logger
from .fixed_scaling import (
    FixedScaler, 
    get_aircraft_scaler, 
    scale_aircraft_data, 
    inverse_scale_aircraft_data,
    AIRCRAFT_SENSOR_RANGES,
    AIRCRAFT_SENSOR_RANGES_EXTENDED
)

__all__ = [
    "load_config",
    "get_data_path", 
    "get_output_path",
    "setup_logging",
    "get_logger",
    "FixedScaler",
    "get_aircraft_scaler", 
    "scale_aircraft_data", 
    "inverse_scale_aircraft_data",
    "AIRCRAFT_SENSOR_RANGES",
    "AIRCRAFT_SENSOR_RANGES_EXTENDED"
] 