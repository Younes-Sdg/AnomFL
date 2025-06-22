import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        """
        Autoencoder pour la détection d'anomalies dans les données de vol.
        :param input_dim: Nombre de caractéristiques en entrée (4 capteurs).
        :param latent_dim: Dimension de la représentation compressée.
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )


    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, z):
        with torch.no_grad():
            return self.decoder(z)

    def load_data_from_csv(self, file_paths: list[str], batch_size: int = 32) -> DataLoader:
        """
        Charge et normalise les données CSV (MinMax scaling) pour entraînement.
        :param file_paths: liste de chemins vers des fichiers CSV
        :param batch_size: taille du batch
        :return: DataLoader PyTorch
        """
        all_dfs = [pd.read_csv(f) for f in file_paths]
        df = pd.concat(all_dfs, ignore_index=True)
        
        features = ['engine_rpm', 'fuel_flow', 'engine_temperature', 'vibration_level']
        data = df[features].values.astype(np.float32)

        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)

        tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def train_autoencoder(self, file_paths: list[str], num_epochs: int = 100, lr: float = 1e-3):
        """
        Entraîne l'autoencodeur localement à partir de fichiers CSV.
        :param file_paths: liste de fichiers CSV
        :param num_epochs: nombre d'époques
        :param lr: taux d'apprentissage
        """
        loader = self.load_data_from_csv(file_paths, self.batch_size)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in loader:
                data = batch[0]
                optimizer.zero_grad()
                output = self(data)
                loss = loss_fn(output, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.size(0)
            avg_loss = total_loss / len(loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
