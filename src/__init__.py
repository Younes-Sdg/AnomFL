"""
AnomFL: Federated Learning for Aircraft Anomaly Detection

A research project implementing federated learning for detecting anomalies
in aircraft engine performance data using autoencoders and FedAvg algorithm.
"""

__version__ = "0.1.0"
__author__ = "AnomFL Team"
__email__ = "contact@anomfl.com"

# Main package imports for easy access
from .data_generation.aircraft_data_generator import Aircraft, Fleet
from .autoencoders.autoencoder import Autoencoder
from .federated.centralized_server import FederatedClient, CentralizedServer

__all__ = [
    "Aircraft",
    "Fleet", 
    "Autoencoder",
    "FederatedClient",
    "CentralizedServer",
] 