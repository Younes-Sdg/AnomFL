"""
Mathematical validation utilities for AnomFL.
Ensures consistency in calculations across the codebase.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


def validate_mse_calculation(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Validates and computes MSE calculation consistently.
    
    Args:
        predictions: Model predictions (batch_size, features)
        targets: True values (batch_size, features)
        
    Returns:
        MSE per sample (batch_size,)
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
    
    # Compute MSE: mean across features, not across batch
    mse = torch.mean((targets - predictions)**2, dim=1)
    return mse


def validate_fedavg_weights(weights_list: List[Dict[str, torch.Tensor]], 
                          client_data_sizes: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    """
    Validates FedAvg weight aggregation.
    
    Args:
        weights_list: List of client model weights
        client_data_sizes: Optional list of client data sizes for weighted averaging
        
    Returns:
        Averaged weights
    """
    if not weights_list:
        raise ValueError("Empty weights list")
    
    # Validate all models have same structure
    first_keys = set(weights_list[0].keys())
    for i, weights in enumerate(weights_list[1:], 1):
        if set(weights.keys()) != first_keys:
            raise ValueError(f"Model {i} has different structure than model 0")
    
    # Calculate weights
    if client_data_sizes and len(client_data_sizes) == len(weights_list):
        total_size = sum(client_data_sizes)
        weights = [size / total_size for size in client_data_sizes]
    else:
        weights = [1.0 / len(weights_list)] * len(weights_list)
    
    # Validate weights sum to 1
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"Weights don't sum to 1: {sum(weights)}")
    
    # Perform weighted averaging
    avg_weights = {}
    for key in first_keys:
        avg_weights[key] = sum(w * weights[i][key] for i, w in enumerate(weights))
    
    return avg_weights


def validate_threshold_calculation(errors: np.ndarray, method: str = "3sigma") -> float:
    """
    Validates anomaly threshold calculation.
    
    Args:
        errors: Array of prediction errors
        method: Threshold method ("3sigma", "percentile", "iqr")
        
    Returns:
        Calculated threshold
    """
    if len(errors) == 0:
        raise ValueError("Empty errors array")
    
    if method == "3sigma":
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        threshold = mean_error + 3 * std_error
    elif method == "percentile":
        threshold = np.percentile(errors, 95)  # 95th percentile
    elif method == "iqr":
        q1, q3 = np.percentile(errors, [25, 75])
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return threshold


def validate_data_scaling(data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Validates data scaling consistency.
    
    Args:
        data: Input data
        scaler: Fitted scaler
        
    Returns:
        Scaled data
    """
    if data.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Data features {data.shape[1]} don't match scaler features {scaler.n_features_in_}")
    
    scaled_data = scaler.transform(data)
    
    # Validate scaling bounds
    if np.any(scaled_data < 0) or np.any(scaled_data > 1):
        print("Warning: Scaled data outside [0,1] range")
    
    return scaled_data


def validate_model_consistency(model1: torch.nn.Module, model2: torch.nn.Module) -> bool:
    """
    Validates that two models have the same architecture.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        True if models are consistent
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    if set(state1.keys()) != set(state2.keys()):
        return False
    
    for key in state1.keys():
        if state1[key].shape != state2[key].shape:
            return False
    
    return True


def validate_linear_regression_formula(weights: torch.Tensor, bias: torch.Tensor, 
                                     inputs: torch.Tensor, predictions: torch.Tensor) -> bool:
    """
    Validates linear regression formula: y = Wx + b
    
    Args:
        weights: Model weights
        bias: Model bias
        inputs: Input features
        predictions: Model predictions
        
    Returns:
        True if formula is correct
    """
    expected = torch.matmul(inputs, weights.T) + bias
    return torch.allclose(predictions, expected, atol=1e-6)


def validate_anomaly_detection_metrics(errors: np.ndarray, threshold: float, 
                                     true_anomalies: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Validates anomaly detection metrics.
    
    Args:
        errors: Prediction errors
        threshold: Anomaly threshold
        true_anomalies: Optional true anomaly labels
        
    Returns:
        Dictionary of metrics
    """
    detected_anomalies = errors > threshold
    
    metrics = {
        'anomaly_rate': np.mean(detected_anomalies),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors)
    }
    
    if true_anomalies is not None:
        if len(true_anomalies) != len(errors):
            raise ValueError("Length mismatch between errors and true anomalies")
        
        tp = np.sum(detected_anomalies & true_anomalies)
        fp = np.sum(detected_anomalies & ~true_anomalies)
        tn = np.sum(~detected_anomalies & ~true_anomalies)
        fn = np.sum(~detected_anomalies & true_anomalies)
        
        metrics.update({
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        })
    
    return metrics 