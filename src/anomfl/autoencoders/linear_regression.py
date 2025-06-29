import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class LinearRegression(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        """
        Linear Regression model for flight data anomaly detection.
        Predicts one sensor value from the other sensors.
        :param input_dim: Number of input features (sensors to use for prediction).
        :param output_dim: Number of output features (sensor to predict).
        """
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

    def train_on_tensor(self, train_tensor, target_tensor, num_epochs=10, lr=1e-3, batch_size=32):
        """Trains the linear regression directly on PyTorch tensors."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        train_dataset = TensorDataset(train_tensor, target_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        print("Starting training...")
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    def train_linear_regression(self, file_paths, target_sensor='engine_temperature', num_epochs=10, lr=1e-3, batch_size=32, input_scaler=None, target_scaler=None):
        """Method to train from a list of CSV files (used by federated clients)."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        train_loader, _, _ = self.load_data_from_csv(
            file_paths, target_sensor, batch_size=batch_size,
            input_scaler=input_scaler, target_scaler=target_scaler
        )
        
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Print loss every few epochs for clients
            if (epoch + 1) % 2 == 0:
                print(f'    Epoch [{epoch+1}/{num_epochs}], Client Loss: {loss.item():.6f}')

    def prediction_error(self, x, y_true):
        """Calculate prediction error for anomaly detection."""
        self.eval()
        with torch.no_grad():
            y_pred = self(x)
            # Ensure consistent MSE calculation: mean across features, not across batch
            return torch.mean((y_true - y_pred)**2, dim=1)

    @staticmethod
    def load_data_from_csv(file_paths, target_sensor='engine_temperature', batch_size=32, sensor_names=None, input_scaler=None, target_scaler=None):
        if sensor_names is None:
            sensor_names = ['engine_rpm', 'fuel_flow', 'engine_temperature', 'vibration_level']
        all_dfs = [pd.read_csv(fp) for fp in file_paths]
        full_df = pd.concat(all_dfs, ignore_index=True)
        input_sensors = [s for s in sensor_names if s != target_sensor]
        input_data = full_df[input_sensors]
        target_data = full_df[[target_sensor]]
        # Use provided scalers if available, otherwise fit new ones
        if input_scaler is not None:
            scaled_input = input_scaler.transform(input_data)
        else:
            input_scaler = MinMaxScaler()
            scaled_input = input_scaler.fit_transform(input_data)
        if target_scaler is not None:
            scaled_target = target_scaler.transform(target_data)
        else:
            target_scaler = MinMaxScaler()
            scaled_target = target_scaler.fit_transform(target_data)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        target_tensor = torch.tensor(scaled_target, dtype=torch.float32)
        dataset = TensorDataset(input_tensor, target_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader, input_scaler, target_scaler 