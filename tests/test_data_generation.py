"""
Tests for the data generation module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.anomfl.data_generation.aircraft_data_generator import Aircraft, Fleet


class TestAircraft:
    """Test cases for the Aircraft class."""
    
    def test_aircraft_initialization(self):
        """Test aircraft initialization."""
        aircraft = Aircraft(fleet_id=1, id=1)
        assert aircraft.fleet_id == 1
        assert aircraft.id == 1
        assert aircraft.engine_rpm is None
        assert aircraft.fuel_flow is None
        
    def test_data_generation(self):
        """Test data generation."""
        aircraft = Aircraft(fleet_id=1, id=1)
        aircraft.generate_data(num_points=10)
        
        assert len(aircraft.engine_rpm) == 10
        assert len(aircraft.fuel_flow) == 10
        assert len(aircraft.engine_temperature) == 10
        assert len(aircraft.vibration_level) == 10
        assert len(aircraft.timestamps) == 10
        
    def test_anomaly_injection(self):
        """Test anomaly injection."""
        aircraft = Aircraft(fleet_id=1, id=1)
        aircraft.generate_data(num_points=20)
        
        # Inject anomaly
        success = aircraft.inject_events(duration=5)
        assert success is True
        assert aircraft.anomaly_info is not None
        assert aircraft.anomaly_info["fleet_id"] == 1
        assert aircraft.anomaly_info["aircraft_id"] == 1
        
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        aircraft = Aircraft(fleet_id=1, id=1)
        aircraft.generate_data(num_points=5)
        
        df = aircraft.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == [
            'fleet_id', 'aircraft_id', 'timestamp', 
            'engine_rpm', 'fuel_flow', 'engine_temperature', 'vibration_level'
        ]


class TestFleet:
    """Test cases for the Fleet class."""
    
    def test_fleet_initialization(self):
        """Test fleet initialization."""
        fleet = Fleet(fleet_id=1, num_aircraft=3)
        assert fleet.fleet_id == 1
        assert fleet.num_aircraft == 3
        assert len(fleet.aircraft_list) == 0
        
    def test_fleet_data_generation(self):
        """Test fleet data generation."""
        fleet = Fleet(
            fleet_id=1, 
            num_aircraft=3, 
            num_anomalous=1,
            num_points=10
        )
        fleet.generate_fleet_data()
        
        assert len(fleet.aircraft_list) == 3
        assert len(fleet.anomaly_records) == 1
        
        # Check that each aircraft has data
        for aircraft in fleet.aircraft_list:
            assert len(aircraft.engine_rpm) == 10
            assert len(aircraft.timestamps) == 10 