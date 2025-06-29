import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class AutoencoderNoScaler(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, latent_dim=2):
        """
        Autoencoder for flight data anomaly detection without scaling.
        :param input_dim: Number of input features (sensors).
        :param hidden_dim: Dimension of the hidden layer.
        :param latent_dim: Dimension of the compressed representation.
        """
        super(AutoencoderNoScaler, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
            # Note: No Sigmoid activation - we want to reconstruct raw values
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_on_tensor(self, train_tensor, num_epochs=10, lr=1e-3, batch_size=32):
        """Trains the autoencoder directly on a PyTorch tensor."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        print("Starting training...")
        for epoch in range(num_epochs):
            for data in train_loader:
                inputs = data[0]
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    def train_autoencoder(self, file_paths, num_epochs=10, lr=1e-3, batch_size=32):
        """Method to train from a list of CSV files (used by federated clients) - NO SCALER."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        train_loader = self.load_data_from_csv(file_paths, batch_size=batch_size)
        
        for epoch in range(num_epochs):
            for inputs, _ in train_loader:
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # This is the new line for federated client statistics
            if (epoch + 1) % 2 == 0: # Print loss every few epochs for clients
                 print(f'    Epoch [{epoch+1}/{num_epochs}], Client Loss: {loss.item():.6f}')

    def reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            x_recon = self(x)
            return torch.mean((x - x_recon)**2, dim=1)

    @staticmethod
    def load_data_from_csv(file_paths, batch_size=32, sensor_names=None):
        if sensor_names is None:
            sensor_names = ['engine_rpm', 'fuel_flow', 'engine_temperature', 'vibration_level']
            
        all_dfs = [pd.read_csv(fp) for fp in file_paths]
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # NO SCALER - Use raw data directly
        data = full_df[sensor_names].values
        
        tensor = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor, tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True) 