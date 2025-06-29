#!/usr/bin/env python3
"""
Generate aircraft data for linear regression experiments.
Temperature will be the target variable, and anomalies will be injected only in temperature.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from anomfl.data_generation.aircraft_data_generator import Fleet
import pandas as pd

def generate_linear_regression_data():
    """Generate data for linear regression experiments with temperature anomalies."""
    
    print("Generating aircraft data for linear regression experiments...")
    print("Target variable: engine_temperature")
    print("Anomalies: Only in temperature")
    
    # Create fleet with temperature-only anomalies
    # Force aircraft 5 to be anomalous by setting num_anomalous=1 and ensuring proper selection
    fleet = Fleet(
        fleet_id=1,
        num_aircraft=5,  # 5 aircraft total
        num_anomalous=1,  # 1 anomalous aircraft
        num_points=200,
        interval_minutes=5,
        anomaly_duration=10,  # 10 points of anomaly
        anomaly_features=['engine_temperature']  # Only temperature anomalies
    )
    
    # Generate and save data
    fleet.generate_fleet_data(output_dir="temp_data")
    
    print("\nData generation complete!")
    print("Files saved in temp_data/fleet_1/")
    print("\nFile structure:")
    print("- aircraft_1_1_normal.csv (normal)")
    print("- aircraft_1_2_normal.csv (normal)")
    print("- aircraft_1_3_normal.csv (normal)")
    print("- aircraft_1_4_normal.csv (normal)")
    print("- aircraft_1_5_anomalous.csv (temperature anomalies)")
    
    # List the generated files
    print("\nGenerated files:")
    for i in range(1, 6):
        suffix = "_anomalous" if i == 5 else "_normal"
        filename = f"temp_data/fleet_1/aircraft_1_{i}{suffix}.csv"
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} (missing)")
    
    # Check which aircraft actually has anomalies
    print("\nChecking anomaly distribution:")
    for i in range(1, 6):
        normal_file = f"temp_data/fleet_1/aircraft_1_{i}_normal.csv"
        anomalous_file = f"temp_data/fleet_1/aircraft_1_{i}_anomalous.csv"
        
        if os.path.exists(anomalous_file):
            print(f"  Aircraft {i}: ANOMALOUS")
        elif os.path.exists(normal_file):
            print(f"  Aircraft {i}: normal")
        else:
            print(f"  Aircraft {i}: missing")

if __name__ == "__main__":
    generate_linear_regression_data() 