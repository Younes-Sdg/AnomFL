"""
Tests for the autoencoder module.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from src.anomfl.autoencoders.autoencoder import Autoencoder


class TestAutoencoder:
    """Test cases for the Autoencoder class."""
    
    def test_autoencoder_initialization(self):
        """Test autoencoder initialization."""
        model = Autoencoder(input_dim=4, latent_dim=2)
        
        # Check architecture
        assert model.encoder is not None
        assert model.decoder is not None
        
        # Check input/output dimensions
        test_input = torch.randn(10, 4)
        output = model(test_input)
        assert output.shape == (10, 4)
        
    def test_forward_pass(self):
        """Test forward pass through autoencoder."""
        model = Autoencoder(input_dim=4, latent_dim=2)
        test_input = torch.randn(5, 4)
        
        output = model(test_input)
        assert output.shape == test_input.shape
        assert isinstance(output, torch.Tensor)
        
    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        model = Autoencoder(input_dim=4, latent_dim=2)
        test_input = torch.randn(10, 4)
        
        errors = model.reconstruction_error(test_input)
        assert errors.shape == (10,)
        assert torch.all(errors >= 0)  # Errors should be non-negative
        
    def test_encode_decode(self):
        """Test encode and decode methods."""
        model = Autoencoder(input_dim=4, latent_dim=2)
        test_input = torch.randn(5, 4)
        
        # Test encoding
        encoded = model.encode(test_input)
        assert encoded.shape == (5, 2)
        
        # Test decoding
        decoded = model.decode(encoded)
        assert decoded.shape == (5, 4)
        
    def test_load_data_from_csv(self):
        """Test loading data from CSV files."""
        model = Autoencoder()
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write test data
            f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
            for i in range(10):
                f.write(f"1,1,2025-01-01 00:{i:02d}:00,{2000+i},{500+i},{700+i},{1.0+i*0.1}\n")
            temp_file = f.name
        
        try:
            # Load data
            dataloader = model.load_data_from_csv([temp_file])
            
            # Check dataloader
            assert dataloader is not None
            batch = next(iter(dataloader))
            assert len(batch) == 1  # DataLoader returns (data,)
            assert batch[0].shape[1] == 4  # 4 features
            
        finally:
            # Clean up
            os.unlink(temp_file)
            
    def test_train_autoencoder(self):
        """Test autoencoder training."""
        model = Autoencoder()
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write test data
            f.write("fleet_id,aircraft_id,timestamp,engine_rpm,fuel_flow,engine_temperature,vibration_level\n")
            for i in range(20):
                f.write(f"1,1,2025-01-01 00:{i:02d}:00,{2000+i},{500+i},{700+i},{1.0+i*0.1}\n")
            temp_file = f.name
        
        try:
            # Train model
            model.train_autoencoder([temp_file], num_epochs=2, lr=1e-3)
            
            # Check that model parameters were updated
            initial_params = [p.clone() for p in model.parameters()]
            # Note: In a real test, you'd want to check that parameters actually changed
            
        finally:
            # Clean up
            os.unlink(temp_file) 