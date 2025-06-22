import torch
import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple
from ..autoencoders.autoencoder import Autoencoder


class FederatedClient:
    """
    Represents a single client (e.g., an aircraft) in the federated system.
    It holds a path to its local data and handles its own local training.
    """
    def __init__(self, client_id: int, file_path: str):
        self.id = client_id
        self.file_path = file_path
        self.model = None # The model will be sent by the server

    def set_model(self, model):
        """Receives a copy of the global model from the server."""
        self.model = copy.deepcopy(model)

    def train(self, num_epochs: int, lr: float):
        """Trains the local model on its local data."""
        if self.model is None:
            raise ValueError("Model has not been set by the server.")
        
        # Each client loads and trains on its own data file
        self.model.train_autoencoder(
            file_paths=[self.file_path],
            num_epochs=num_epochs,
            lr=lr
        )

    def get_weights(self):
        """Returns a copy of the local model's weights."""
        if self.model is None:
            return None
        return copy.deepcopy(self.model.state_dict())

    def reconstruction_error(self, data_tensor: torch.Tensor):
        """Calculates reconstruction error for a given tensor."""
        if self.model is None:
            raise ValueError("Model has not been set by the server.")
        return self.model.reconstruction_error(data_tensor)


class CentralizedServer:
    """
    Orchestrates the Federated Averaging (FedAvg) process.
    """
    def __init__(self, clients: List[FederatedClient], model: Autoencoder):
        self.clients = clients
        self.global_model = copy.deepcopy(model)

    @staticmethod
    def _average_weights(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Averages the weights from a list of client models."""
        if not weights_list:
            return {}
        
        # Get the keys from the first model's state_dict
        avg_weights = copy.deepcopy(weights_list[0])
        
        # Sum the weights for each layer
        for key in avg_weights:
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            
            # Divide by the number of clients to get the average
            avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
            
        return avg_weights

    def train(self, rounds: int, local_epochs: int, lr: float):
        """Manages the federated training rounds."""
        for r in range(rounds):
            print(f"\nROUND {r+1}/{rounds}")
            
            # 1. Distribute the current global model to all clients
            for client in self.clients:
                client.set_model(self.global_model)

            # 2. Train each client locally and collect their updated weights
            weights_collected = []
            for client in self.clients:
                print(f"  -> Training Client {client.id}...")
                client.train(num_epochs=local_epochs, lr=lr)
                weights_collected.append(client.get_weights())

            # 3. Aggregate the collected weights to update the global model
            new_global_weights = self._average_weights(weights_collected)
            if new_global_weights:
                self.global_model.load_state_dict(new_global_weights)

        print("\nFederated training complete.")

    def evaluate_client(self, client_id: int):
        """
        Calculates reconstruction errors and an anomaly threshold for a specific client.
        The threshold is based on the 3-sigma rule applied to that client's data.
        """
        target_client = next((c for c in self.clients if c.id == client_id), None)
        if not target_client:
            raise ValueError(f"Client with ID {client_id} not found.")
            
        # Load the client's data to calculate errors
        df = pd.read_csv(target_client.file_path)
        sensor_names = ['engine_rpm', 'fuel_flow', 'engine_temperature', 'vibration_level']
        
        # This is a simplification. In a real scenario, the scaler should be
        # fitted only on normal data and reused.
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[sensor_names])
        data_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # Calculate errors using the final global model
        errors = self.global_model.reconstruction_error(data_tensor).numpy()
        
        # Calculate threshold (3-sigma rule)
        threshold = np.mean(errors) + 3 * np.std(errors)
        
        return errors, threshold

    def benchmark_all_clients(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame (mean, max, std) for each client,
        sorted by mean error descending.
        """
        records = []
        for c in self.clients:
            errors, _ = self.evaluate_client(c.id)
            records.append({
                "client_id": c.id,
                "mean_mse": errors.mean(),
                "max_mse": errors.max(),
                "std_mse": errors.std(),
                "n_points": len(errors)
            })
        df = pd.DataFrame(records).sort_values("mean_mse", ascending=False)
        return df.reset_index(drop=True)
