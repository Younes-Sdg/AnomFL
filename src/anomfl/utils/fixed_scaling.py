"""
Fixed scaling utilities for aircraft sensor data.
These scaling functions use predefined ranges and don't depend on the data being scaled.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class FixedScaler:
    """
    A scaler with fixed parameters that don't depend on the data being scaled.
    This ensures consistent scaling across all experiments and better anomaly detection.
    """
    
    def __init__(self, feature_ranges: Dict[str, Dict[str, float]]):
        """
        Initialize with fixed ranges for each feature.
        
        Args:
            feature_ranges: Dictionary mapping feature names to their min/max ranges
                Example: {
                    'engine_rpm': {'min': 800, 'max': 3000},
                    'fuel_flow': {'min': 200, 'max': 800},
                    ...
                }
        """
        self.feature_ranges = feature_ranges
        self.fitted = True  # Always fitted since ranges are fixed
    
    def fit(self, X):
        """No-op since ranges are fixed. Included for sklearn compatibility."""
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fixed ranges.
        
        Args:
            X: Input data (numpy array or pandas DataFrame)
            
        Returns:
            Scaled data as numpy array
        """
        if isinstance(X, pd.DataFrame):
            return self._transform_dataframe(X)
        else:
            return self._transform_array(X)
    
    def _transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Transform pandas DataFrame using fixed ranges."""
        result = df.copy()
        
        for feature in df.columns:
            if feature in self.feature_ranges:
                min_val = self.feature_ranges[feature]['min']
                max_val = self.feature_ranges[feature]['max']
                result[feature] = (df[feature] - min_val) / (max_val - min_val)
        
        return result.values
    
    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform numpy array using fixed ranges."""
        # For array input, assume features are in the order of feature_ranges keys
        result = X.copy()
        feature_names = list(self.feature_ranges.keys())
        
        for i, feature in enumerate(feature_names):
            if i < X.shape[1]:
                min_val = self.feature_ranges[feature]['min']
                max_val = self.feature_ranges[feature]['max']
                result[:, i] = (X[:, i] - min_val) / (max_val - min_val)
        
        return result
    
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            X: Scaled data (numpy array or pandas DataFrame)
            
        Returns:
            Original scale data as numpy array
        """
        if isinstance(X, pd.DataFrame):
            return self._inverse_transform_dataframe(X)
        else:
            return self._inverse_transform_array(X)
    
    def _inverse_transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Inverse transform pandas DataFrame."""
        result = df.copy()
        
        for feature in df.columns:
            if feature in self.feature_ranges:
                min_val = self.feature_ranges[feature]['min']
                max_val = self.feature_ranges[feature]['max']
                result[feature] = df[feature] * (max_val - min_val) + min_val
        
        return result.values
    
    def _inverse_transform_array(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform numpy array."""
        result = X.copy()
        feature_names = list(self.feature_ranges.keys())
        
        for i, feature in enumerate(feature_names):
            if i < X.shape[1]:
                min_val = self.feature_ranges[feature]['min']
                max_val = self.feature_ranges[feature]['max']
                result[:, i] = X[:, i] * (max_val - min_val) + min_val
        
        return result


# Predefined aircraft sensor ranges based on the data generator
AIRCRAFT_SENSOR_RANGES = {
    'engine_rpm': {'min': 800, 'max': 3000},
    'fuel_flow': {'min': 200, 'max': 800},
    'engine_temperature': {'min': 400, 'max': 900},
    'vibration_level': {'min': 0.1, 'max': 3.0}
}

# Extended ranges that include anomalous values
AIRCRAFT_SENSOR_RANGES_EXTENDED = {
    'engine_rpm': {'min': 500, 'max': 3000},  # Anomalies can go down to 500
    'fuel_flow': {'min': 100, 'max': 800},    # Anomalies can go down to 100
    'engine_temperature': {'min': 400, 'max': 1100},  # Anomalies can go up to 1100
    'vibration_level': {'min': 0.1, 'max': 5.0}      # Anomalies can go up to 5.0
}


def get_aircraft_scaler(include_anomalies: bool = False) -> FixedScaler:
    """
    Get a fixed scaler for aircraft sensor data.
    
    Args:
        include_anomalies: If True, use extended ranges that include anomalous values.
                          If False, use normal operating ranges only.
    
    Returns:
        FixedScaler instance configured for aircraft data
    """
    ranges = AIRCRAFT_SENSOR_RANGES_EXTENDED if include_anomalies else AIRCRAFT_SENSOR_RANGES
    return FixedScaler(ranges)


def scale_aircraft_data(data: Union[np.ndarray, pd.DataFrame], 
                       include_anomalies: bool = False) -> np.ndarray:
    """
    Convenience function to scale aircraft data using fixed ranges.
    
    Args:
        data: Input data (numpy array or pandas DataFrame)
        include_anomalies: Whether to use extended ranges that include anomalies
    
    Returns:
        Scaled data as numpy array
    """
    scaler = get_aircraft_scaler(include_anomalies)
    return scaler.transform(data)


def inverse_scale_aircraft_data(data: Union[np.ndarray, pd.DataFrame], 
                               include_anomalies: bool = False) -> np.ndarray:
    """
    Convenience function to inverse scale aircraft data.
    
    Args:
        data: Scaled data (numpy array or pandas DataFrame)
        include_anomalies: Whether the data was scaled with extended ranges
    
    Returns:
        Original scale data as numpy array
    """
    scaler = get_aircraft_scaler(include_anomalies)
    return scaler.inverse_transform(data) 