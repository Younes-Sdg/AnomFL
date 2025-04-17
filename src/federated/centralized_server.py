import torch
import copy
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from autoencoders.autoencoder import Autoencoder


class FederatedClient:
    """
    Client fédéré (un avion ou une flotte).
    Possède un autoencodeur et ses fichiers CSV locaux.
    """
    def __init__(self, client_id: int, file_paths: List[str]):
        self.id = client_id
        self.file_paths = file_paths
        self.model = Autoencoder()

    # ---------- Entraînement local ----------
    def train(self, epochs: int = 1, lr: float = 1e-3):
        print(f"Client {self.id} - Entraînement local...")
        self.model.train_autoencoder(self.file_paths, num_epochs=epochs, lr=lr)

    # ---------- Gestion des poids ----------
    def get_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, weights):
        self.model.load_state_dict(copy.deepcopy(weights))

    # ---------- Évaluation ----------
    def reconstruction_errors(self) -> np.ndarray:
        """
        Retourne l'erreur MSE point-à-point sur TOUTES ses données.
        """
        all_errors = []
        for csv_path in self.file_paths:
            df = pd.read_csv(csv_path)
            feats = df[['engine_rpm', 'fuel_flow',
                        'engine_temperature', 'vibration_level']]
            norm = (feats - feats.min()) / (feats.max() - feats.min())
            tensor = torch.tensor(norm.values, dtype=torch.float32)
            errs = self.model.reconstruction_error(tensor).detach().numpy()
            all_errors.append(errs)
        return np.concatenate(all_errors)


class CentralizedServer:
    """
    Orchestrateur FedAvg + outils de détection d'anomalies.
    """
    def __init__(self, clients: List[FederatedClient]):
        self.clients = clients
        self.global_model = Autoencoder()

    # ---------- Agrégation FedAvg ----------
    @staticmethod
    def _average(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        avg = copy.deepcopy(weights_list[0])
        for key in avg:
            for i in range(1, len(weights_list)):
                avg[key] += weights_list[i][key]
            avg[key] /= len(weights_list)
        return avg

    # ---------- Entraînement fédéré ----------
    def train(self, rounds: int = 5, local_epochs: int = 1, lr: float = 1e-3):
        for r in range(rounds):
            print(f"\n📡 ROUND {r+1}/{rounds}")
            weights_collected = []

            # 1) Distribuer le modèle global
            for c in self.clients:
                c.set_weights(self.global_model.state_dict())

            # 2) Entraînement local + collecte des poids
            for c in self.clients:
                c.train(epochs=local_epochs, lr=lr)
                weights_collected.append(c.get_weights())

            # 3) Agrégation
            self.global_model.load_state_dict(self._average(weights_collected))

        print("\n✅ Entraînement fédéré terminé.")

    # ---------- Évaluation / Détection ----------
    def evaluate_client(self, client_id: int) -> Tuple[np.ndarray, float, float]:
        """
        Retourne (erreurs, seuil, proportion_anomalies) pour un client.
        Seuil = mu + 3*sigma (3σ-rule).
        """
        client = self.clients[client_id]
        client.set_weights(self.global_model.state_dict())  # sync dernier modèle
        errs = client.reconstruction_errors()
        mu, sigma = errs.mean(), errs.std()
        threshold = mu + 3 * sigma
        prop_anom = (errs > threshold).mean()
        return errs, threshold, prop_anom

    def detect_anomalies(self, client_id: int, k: float = 3.0) -> np.ndarray:
        """
        Indices où l'erreur > mu + k*sigma.
        """
        errs = self.clients[client_id].reconstruction_errors()
        mu, sigma = errs.mean(), errs.std()
        return np.where(errs > mu + k * sigma)[0]

    def benchmark_all_clients(self) -> pd.DataFrame:
        """
        Retourne un DataFrame récapitulatif (mean, max, std) pour chaque client
        et les classe par erreur moyenne décroissante.
        """
        records = []
        for c in self.clients:
            c.set_weights(self.global_model.state_dict())
            errs = c.reconstruction_errors()
            records.append({
                "client_id": c.id,
                "mean_mse": errs.mean(),
                "max_mse": errs.max(),
                "std_mse": errs.std(),
                "n_points": len(errs)
            })
        df = pd.DataFrame(records).sort_values("mean_mse", ascending=False)
        return df.reset_index(drop=True)
