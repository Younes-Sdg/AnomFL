import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

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

    def load_data_from_csv(self, file_paths):
        """
        Charge et normalise les données CSV (MinMax scaling) pour entraînement.
        :param file_paths: liste de chemins vers des fichiers CSV
        :return: DataLoader PyTorch
        """
        all_data = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            features = df[['engine_rpm', 'fuel_flow', 'engine_temperature', 'vibration_level']]
            normalized = (features - features.min()) / (features.max() - features.min())
            all_data.append(torch.tensor(normalized.values, dtype=torch.float32))
        full_dataset = torch.cat(all_data, dim=0)
        return DataLoader(TensorDataset(full_dataset), batch_size=32, shuffle=True)

    def train_autoencoder(self, file_paths, num_epochs=10, lr=1e-3):
        """
        Entraîne l'autoencodeur localement à partir de fichiers CSV.
        :param file_paths: liste de fichiers CSV
        :param num_epochs: nombre d'époques
        :param lr: taux d'apprentissage
        """
        train_loader = self.load_data_from_csv(file_paths)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                data = batch[0]
                optimizer.zero_grad()
                output = self(data)
                loss = loss_fn(output, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
