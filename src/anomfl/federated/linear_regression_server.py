import torch
import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional
from ..autoencoders.linear_regression import LinearRegression


class LinearRegressionFederatedClient:
    """
    Represents a single client (e.g., an aircraft) in the federated system for linear regression.
    It holds a path to its local data and handles its own local training.
    """
    def __init__(self, client_id: int, file_path: str, target_sensor='engine_temperature'):
        self.id = client_id
        self.file_path = file_path
        self.target_sensor = target_sensor
        self.model = None # The model will be sent by the server

    def set_model(self, model):
        """Receives a copy of the global model from the server."""
        self.model = copy.deepcopy(model)

    def train(self, num_epochs: int, lr: float):
        """Trains the local model on its local data."""
        if self.model is None:
            raise ValueError("Model has not been set by the server.")
        
        # Each client loads and trains on its own data file
        self.model.train_linear_regression(
            file_paths=[self.file_path],
            target_sensor=self.target_sensor,
            num_epochs=num_epochs,
            lr=lr
        )

    def get_weights(self):
        """Returns a copy of the local model's weights."""
        if self.model is None:
            return None
        return copy.deepcopy(self.model.state_dict())

    def prediction_error(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        """Calculates prediction error for given input and target tensors."""
        if self.model is None:
            raise ValueError("Model has not been set by the server.")
        return self.model.prediction_error(input_tensor, target_tensor)


class LinearRegressionCentralizedServer:
    """
    Orchestrates the Federated Averaging (FedAvg) process for linear regression models.
    """
    def __init__(self, clients: List[LinearRegressionFederatedClient], model: LinearRegression):
        self.clients = clients
        self.global_model = copy.deepcopy(model)

    @staticmethod
    def _average_weights(weights_list: List[Dict[str, torch.Tensor]], client_data_sizes: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Averages the weights from a list of client models using FedAvg algorithm."""
        if not weights_list:
            return {}
        
        # If client data sizes are provided, use weighted averaging (FedAvg)
        if client_data_sizes and len(client_data_sizes) == len(weights_list):
            total_data_size = sum(client_data_sizes)
            weights = [size / total_data_size for size in client_data_sizes]
        else:
            # Equal weighting if data sizes not provided
            weights = [1.0 / len(weights_list)] * len(weights_list)
        
        # Get the keys from the first model's state_dict
        avg_weights = copy.deepcopy(weights_list[0])
        
        # Initialize with weighted first model
        for key in avg_weights:
            avg_weights[key] = weights[0] * weights_list[0][key]
        
        # Add weighted contributions from other models
        for i in range(1, len(weights_list)):
            for key in avg_weights:
                avg_weights[key] += weights[i] * weights_list[i][key]
            
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
        Calculates prediction errors and an anomaly threshold for a specific client.
        The threshold is based on the 3-sigma rule applied to that client's data.
        """
        target_client = next((c for c in self.clients if c.id == client_id), None)
        if not target_client:
            raise ValueError(f"Client with ID {client_id} not found.")
            
        # Load the client's data to calculate errors
        df = pd.read_csv(target_client.file_path)
        sensor_names = ['engine_rpm', 'fuel_flow', 'engine_temperature', 'vibration_level']
        
        # Separate target sensor from input sensors
        input_sensors = [s for s in sensor_names if s != target_client.target_sensor]
        
        # Prepare input and target data
        input_data = df[input_sensors]
        target_data = df[[target_client.target_sensor]]
        
        # Scale the data consistently
        input_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        scaled_input = input_scaler.fit_transform(input_data)
        scaled_target = target_scaler.fit_transform(target_data)
        
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        target_tensor = torch.tensor(scaled_target, dtype=torch.float32)

        # Calculate errors using the final global model
        errors = self.global_model.prediction_error(input_tensor, target_tensor).numpy()
        
        # Calculate threshold using 3-sigma rule (more robust than mean + 3*std)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        threshold = mean_error + 3 * std_error
        
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