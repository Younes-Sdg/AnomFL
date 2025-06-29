"""
Tests for the federated learning module.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from src.anomfl.federated.centralized_server import FederatedClient, CentralizedServer


class TestFederatedClient:
    """Test cases for the FederatedClient class."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
            f.write("1,1,2025-01-01 00:00:00,2000,500,700,1.0\n")
            temp_file = f.name
        
        try:
            client = FederatedClient(client_id=0, file_paths=[temp_file])
            assert client.id == 0
            assert client.file_paths == [temp_file]
            assert client.model is not None
        finally:
            os.unlink(temp_file)
    
    def test_get_set_weights(self):
        """Test weight management."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
            f.write("1,1,2025-01-01 00:00:00,2000,500,700,1.0\n")
            temp_file = f.name
        
        try:
            client = FederatedClient(client_id=0, file_paths=[temp_file])
            
            # Get weights
            weights = client.get_weights()
            assert isinstance(weights, dict)
            assert len(weights) > 0
            
            # Set weights
            client.set_weights(weights)
            
        finally:
            os.unlink(temp_file)
    
    def test_reconstruction_errors(self):
        """Test reconstruction error calculation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
            for i in range(10):
                f.write(f"1,1,2025-01-01 00:{i:02d}:00,{2000+i},{500+i},{700+i},{1.0+i*0.1}\n")
            temp_file = f.name
        
        try:
            client = FederatedClient(client_id=0, file_paths=[temp_file])
            
            # Calculate reconstruction errors
            errors = client.reconstruction_errors()
            assert isinstance(errors, np.ndarray)
            assert len(errors) == 10
            assert np.all(errors >= 0)
            
        finally:
            os.unlink(temp_file)


class TestCentralizedServer:
    """Test cases for the CentralizedServer class."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        # Create test clients
        clients = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
                f.write(f"1,{i+1},2025-01-01 00:00:00,2000,500,700,1.0\n")
                temp_file = f.name
            
            client = FederatedClient(client_id=i, file_paths=[temp_file])
            clients.append(client)
        
        try:
            server = CentralizedServer(clients)
            assert len(server.clients) == 3
            assert server.global_model is not None
            
        finally:
            # Clean up
            for client in clients:
                os.unlink(client.file_paths[0])
    
    def test_weight_averaging(self):
        """Test FedAvg weight averaging."""
        # Create test clients
        clients = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
                f.write(f"1,{i+1},2025-01-01 00:00:00,2000,500,700,1.0\n")
                temp_file = f.name
            
            client = FederatedClient(client_id=i, file_paths=[temp_file])
            clients.append(client)
        
        try:
            server = CentralizedServer(clients)
            
            # Get weights from clients
            weights_list = [client.get_weights() for client in clients]
            
            # Test averaging
            avg_weights = server._average(weights_list)
            assert isinstance(avg_weights, dict)
            assert len(avg_weights) > 0
            
        finally:
            # Clean up
            for client in clients:
                os.unlink(client.file_paths[0])
    
    def test_evaluate_client(self):
        """Test client evaluation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
            for i in range(10):
                f.write(f"1,1,2025-01-01 00:{i:02d}:00,{2000+i},{500+i},{700+i},{1.0+i*0.1}\n")
            temp_file = f.name
        
        try:
            client = FederatedClient(client_id=0, file_paths=[temp_file])
            server = CentralizedServer([client])
            
            # Evaluate client
            errors, threshold, prop_anom = server.evaluate_client(client_id=0)
            
            assert isinstance(errors, np.ndarray)
            assert isinstance(threshold, float)
            assert isinstance(prop_anom, float)
            assert 0 <= prop_anom <= 1
            
        finally:
            os.unlink(temp_file)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
            for i in range(10):
                f.write(f"1,1,2025-01-01 00:{i:02d}:00,{2000+i},{500+i},{700+i},{1.0+i*0.1}\n")
            temp_file = f.name
        
        try:
            client = FederatedClient(client_id=0, file_paths=[temp_file])
            server = CentralizedServer([client])
            
            # Detect anomalies
            anomalies = server.detect_anomalies(client_id=0, k=3.0)
            
            assert isinstance(anomalies, np.ndarray)
            assert len(anomalies) >= 0  # Could be 0 if no anomalies detected
            
        finally:
            os.unlink(temp_file)
    
    def test_benchmark_all_clients(self):
        """Test client benchmarking."""
        # Create test clients
        clients = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
                for j in range(5):
                    f.write(f"1,{i+1},2025-01-01 00:{j:02d}:00,{2000+j},{500+j},{700+j},{1.0+j*0.1}\n")
                temp_file = f.name
            
            client = FederatedClient(client_id=i, file_paths=[temp_file])
            clients.append(client)
        
        try:
            server = CentralizedServer(clients)
            
            # Benchmark all clients
            results = server.benchmark_all_clients()
            
            assert isinstance(results, pd.DataFrame)
            assert len(results) == 2  # 2 clients
            assert "client_id" in results.columns
            assert "mean_mse" in results.columns
            
        finally:
            # Clean up
            for client in clients:
                os.unlink(client.file_paths[0]) 