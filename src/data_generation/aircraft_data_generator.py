import numpy as np
import pandas as pd
import os
import random
import copy

class Aircraft:
    def __init__(self, fleet_id, id, start_time=None):
        self.fleet_id = fleet_id
        self.id = id
        self.engine_rpm = None
        self.fuel_flow = None
        self.engine_temperature = None
        self.vibration_level = None
        self.timestamps = None
        self.anomaly_info = None  # Store anomaly information
        
        # Ensure start time is the same for all aircraft with a margin of ±5 minutes
        base_time = pd.Timestamp("2025-01-01 00:00:00")
        margin = random.randint(-5, 5)  # Random offset in minutes
        self.start_time = base_time + pd.Timedelta(minutes=margin) if start_time is None else start_time

    def generate_data(self, num_points=100, interval_minutes=5, inject_anomalies=False):
        """Simulate normal engine data for the aircraft with a time-series structure."""
        self.timestamps = [self.start_time + pd.Timedelta(minutes=i * interval_minutes) for i in range(num_points)]

        self.engine_rpm = np.random.normal(2000, 100, num_points)  # RPM
        self.fuel_flow = np.random.normal(500, 50, num_points)  # kg/hr
        self.engine_temperature = np.random.normal(700, 30, num_points)  # Celsius
        self.vibration_level = np.random.normal(1.0, 0.1, num_points)  # G-force
        
        if inject_anomalies:
            self.inject_events(duration=5)

    def inject_events(self, duration=5):
        """Introduce a single anomaly affecting multiple consecutive time points."""
        num_points = len(self.engine_rpm)
        if num_points <= duration:
            return  
        
        idx = random.randint(0, num_points - duration)
        event_type = random.choice(["engine_stall", "fuel_leak", "turbulence", "overheat"])
        
        for i in range(duration):  # Apply anomaly over multiple time steps
            timestamp = self.timestamps[idx + i]
            if event_type == "engine_stall":
                self.engine_rpm[idx + i] = np.random.uniform(500, 1000)  # Sudden drop
            elif event_type == "fuel_leak":
                self.fuel_flow[idx + i] = np.random.uniform(100, 200)  # Sharp decrease
            elif event_type == "turbulence":
                self.vibration_level[idx + i] = np.random.uniform(2.0, 5.0)  # High vibration
            elif event_type == "overheat":
                self.engine_temperature[idx + i] = np.random.uniform(900, 1100)  # Temperature spike
        
        start_time = self.timestamps[idx]
        end_time = self.timestamps[idx + duration - 1]
        self.anomaly_info = {
            "fleet_id": self.fleet_id,
            "aircraft_id": self.id,
            "anomaly_type": event_type,
            "start_time": start_time,
            "end_time": end_time
        }
        print(f"Anomaly '{event_type}' added from time {start_time} to {end_time} for aircraft {self.fleet_id}-{self.id}")

    def to_dataframe(self):
        """Convert the aircraft data into a pandas DataFrame with a time-series index."""
        data = {
            'fleet_id': self.fleet_id,
            'aircraft_id': self.id,
            'timestamp': self.timestamps,
            'engine_rpm': self.engine_rpm,
            'fuel_flow': self.fuel_flow,
            'engine_temperature': self.engine_temperature,
            'vibration_level': self.vibration_level
        }
        return pd.DataFrame(data)

    def save_dataset(self, output_dir="../data/aircraft_data", suffix=""):
        """Save the aircraft's dataset to a CSV file inside a fleet-specific folder."""
        fleet_dir = os.path.join(output_dir, f"fleet_{self.fleet_id}")
        if not os.path.exists(fleet_dir):
            os.makedirs(fleet_dir)
        
        filename = f"aircraft_{self.fleet_id}_{self.id}{suffix}.csv"
        filepath = os.path.join(fleet_dir, filename)
        self.to_dataframe().to_csv(filepath, index=False)
        print(f"Dataset saved as {filepath}")

    def copy(self):
        """Return a deep copy of the aircraft object."""
        return copy.deepcopy(self)


class Fleet:
    def __init__(self, fleet_id, num_aircraft=10, num_anomalous=3, num_points=200, interval_minutes=5):
        self.fleet_id = fleet_id
        self.num_aircraft = num_aircraft
        self.num_anomalous = num_anomalous
        self.num_points = num_points
        self.interval_minutes = interval_minutes
        self.aircraft_list = []
        self.anomaly_records = []

    def generate_fleet_data(self):
        """Generate data for an entire fleet of aircraft."""
        anomalous_indices = set(random.sample(range(self.num_aircraft), self.num_anomalous))
        
        for i in range(self.num_aircraft):
            aircraft = Aircraft(fleet_id=self.fleet_id, id=i+1)
            inject_anomaly = i in anomalous_indices
            aircraft.generate_data(num_points=self.num_points, interval_minutes=self.interval_minutes, inject_anomalies=inject_anomaly)
            aircraft.save_dataset(suffix="_anomalous" if inject_anomaly else "_normal")
            self.aircraft_list.append(aircraft)
            print(f"Aircraft {self.fleet_id}-{i+1} dataset generated ({'Anomalous' if inject_anomaly else 'Normal'}) in fleet {self.fleet_id}")
            
            if inject_anomaly and aircraft.anomaly_info:
                self.anomaly_records.append(aircraft.anomaly_info)
        
        # Save the anomaly summary dataset
        self.save_anomaly_summary()
    
    def save_anomaly_summary(self, output_dir="../data/aircraft_data"):
        """Save a CSV file containing all anomalies in the fleet."""
        fleet_dir = os.path.join(output_dir, f"fleet_{self.fleet_id}")
        if not os.path.exists(fleet_dir):
            os.makedirs(fleet_dir)
        
        if self.anomaly_records:
            anomaly_df = pd.DataFrame(self.anomaly_records)
            anomaly_df.to_csv(os.path.join(fleet_dir, f"fleet_{self.fleet_id}_anomalies.csv"), index=False)
            print(f"Anomaly summary saved for fleet {self.fleet_id}")


