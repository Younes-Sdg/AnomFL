"""
AnomFL: Federated Learning for Aircraft Anomaly Detection

A research project implementing federated learning for detecting anomalies
in aircraft engine performance data using autoencoders and FedAvg algorithm.
"""

__version__ = "0.1.0"
__author__ = "AnomFL Team"
__email__ = "contact@anomfl.com"

# Main package imports for easy access
from .anomfl.data_generation.aircraft_data_generator import Aircraft, Fleet
from .anomfl.autoencoders.autoencoder import Autoencoder
from .anomfl.autoencoders.linear_regression import LinearRegression
from .anomfl.federated.centralized_server import FederatedClient, CentralizedServer
from .anomfl.federated.linear_regression_server import LinearRegressionFederatedClient, LinearRegressionCentralizedServer

__all__ = [
    "Aircraft",
    "Fleet", 
    "Autoencoder",
    "LinearRegression",
    "FederatedClient",
    "CentralizedServer",
    "LinearRegressionFederatedClient",
    "LinearRegressionCentralizedServer",
] 