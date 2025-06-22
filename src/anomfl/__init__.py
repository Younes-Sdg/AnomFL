"""
AnomFL - Federated Learning for Aircraft Anomaly Detection
"""

from .data_generation.aircraft_data_generator import Aircraft, Fleet
from .autoencoders.autoencoder import Autoencoder
from .federated.centralized_server import FederatedClient, CentralizedServer

__version__ = "0.1.0"

__all__ = [
    "Aircraft",
    "Fleet", 
    "Autoencoder",
    "FederatedClient",
    "CentralizedServer",
] 