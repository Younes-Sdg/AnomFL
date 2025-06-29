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
        self.anomaly_info = None

        base_time = pd.Timestamp("2025-01-01 00:00:00")
        self.start_time = start_time if start_time is not None else base_time

    def generate_data(self, num_points=100, interval_minutes=1):
        """Generate realistic aircraft sensor data with correlations."""
        self.timestamps = [self.start_time + pd.Timedelta(minutes=i * interval_minutes) for i in range(num_points)]
        
        # Generate base engine RPM with realistic variations
        self.engine_rpm = np.random.normal(2000, 100, num_points) ### 2000/300 (3*sigma)
        
        # Fuel flow correlates with engine RPM (higher RPM = higher fuel consumption)
        fuel_flow_base = 500 + 0.1 * (self.engine_rpm - 2000)  # Base correlation -500/90
        self.fuel_flow = fuel_flow_base + np.random.normal(0, 20, num_points)
        
        # Engine temperature correlates with both RPM and fuel flow
        temp_base = 700 + 0.05 * (self.engine_rpm - 2000) + 0.1 * (self.fuel_flow - 500)
        self.engine_temperature = temp_base + np.random.normal(0, 15, num_points)
        
        # Vibration level correlates with RPM (higher RPM = more vibration)
        vib_base = 1.0 + 0.0001 * (self.engine_rpm - 2000)
        self.vibration_level = vib_base + np.random.normal(0, 0.05, num_points)
        
        # Ensure all values are physically realistic
        self.engine_rpm = np.clip(self.engine_rpm, 800, 3000)
        self.fuel_flow = np.clip(self.fuel_flow, 200, 800)
        self.engine_temperature = np.clip(self.engine_temperature, 400, 900)
        self.vibration_level = np.clip(self.vibration_level, 0.1, 3.0)

    def inject_events(self, duration=5, affected_features=None):
        if self.engine_rpm is None:
            print("[ERROR] Data not generated. Call generate_data() first.")
            return False

        num_points = len(self.engine_rpm)
        if num_points < duration:
            print(f"[WARNING] Not enough points to inject anomaly (duration={duration}, points={num_points})")
            return False

        idx = random.randint(0, num_points - duration)
        event_type = random.choice(["engine_stall", "fuel_leak", "turbulence", "overheat"])

        default_map = {
            "engine_stall": ["engine_rpm"],
            "fuel_leak": ["fuel_flow"],
            "turbulence": ["vibration_level"],
            "overheat": ["engine_temperature"]
        }
        features_to_inject = affected_features if affected_features is not None else default_map[event_type]

        for i in range(duration):
            t = idx + i
            if "engine_rpm" in features_to_inject and self.engine_rpm is not None:
                self.engine_rpm[t] = np.random.uniform(500, 1000)
            if "fuel_flow" in features_to_inject and self.fuel_flow is not None:
                self.fuel_flow[t] = np.random.uniform(100, 200)
            if "engine_temperature" in features_to_inject and self.engine_temperature is not None:
                self.engine_temperature[t] = np.random.uniform(900, 1100)
            if "vibration_level" in features_to_inject and self.vibration_level is not None:
                self.vibration_level[t] = np.random.uniform(2.0, 5.0)

        if self.timestamps is None:
            return False
            
        start_time = self.timestamps[idx]
        end_time = self.timestamps[idx + duration - 1]
        self.anomaly_info = {
            "fleet_id": self.fleet_id,
            "aircraft_id": self.id,
            "anomaly_type": event_type,
            "affected_features": features_to_inject,
            "start_time": start_time,
            "end_time": end_time
        }
        return True

    def to_dataframe(self):
        return pd.DataFrame({
            'fleet_id': self.fleet_id, 'aircraft_id': self.id, 'timestamp': self.timestamps,
            'engine_rpm': self.engine_rpm, 'fuel_flow': self.fuel_flow,
            'engine_temperature': self.engine_temperature, 'vibration_level': self.vibration_level
        })

    def save_dataset(self, output_dir="data", suffix=""):
        fleet_dir = os.path.join(output_dir, f"fleet_{self.fleet_id}")
        os.makedirs(fleet_dir, exist_ok=True)
        filename = f"aircraft_{self.fleet_id}_{self.id}{suffix}.csv"
        filepath = os.path.join(fleet_dir, filename)
        self.to_dataframe().to_csv(filepath, index=False)
        print(f"Dataset saved as {os.path.abspath(filepath)}")

    def copy(self):
        return copy.deepcopy(self)


class Fleet:
    def __init__(self, fleet_id, num_aircraft=10, num_anomalous=3,
                 num_points=200, interval_minutes=5, anomaly_duration=5,
                 anomaly_features=None):
        self.fleet_id = fleet_id
        self.num_aircraft = num_aircraft
        self.num_anomalous = num_anomalous
        self.num_points = num_points
        self.interval_minutes = interval_minutes
        self.anomaly_duration = anomaly_duration
        self.anomaly_features = anomaly_features
        self.aircraft_list = []
        self.anomaly_records = []

    def generate_fleet_data(self, output_dir="data"):
        """
        Generates data for all aircraft in the fleet and saves them to CSV files.
        :param output_dir: The directory where fleet data will be saved.
        """
        all_indices = list(range(self.num_aircraft))
        random.shuffle(all_indices)
        anomalous_indices = set(all_indices[:self.num_anomalous])

        shared_start_time = pd.Timestamp("2025-01-01 00:00:00")

        for i in range(self.num_aircraft):
            aircraft = Aircraft(fleet_id=self.fleet_id, id=i + 1, start_time=shared_start_time)
            aircraft.generate_data(num_points=self.num_points, interval_minutes=self.interval_minutes)

            anomaly_injected = False
            if i in anomalous_indices:
                anomaly_injected = aircraft.inject_events(
                    duration=self.anomaly_duration,
                    affected_features=self.anomaly_features
                )

            aircraft.save_dataset(output_dir=output_dir, suffix="_anomalous" if anomaly_injected else "_normal")
            self.aircraft_list.append(aircraft)
            print(f"Aircraft {self.fleet_id}-{i + 1} generated ({'Anomalous' if anomaly_injected else 'Normal'})")

            if anomaly_injected and aircraft.anomaly_info:
                self.anomaly_records.append(aircraft.anomaly_info)
        
        self.save_anomaly_summary(output_dir=output_dir)

    def save_anomaly_summary(self, output_dir="data"):
        """Saves a summary of generated anomalies to a CSV file."""
        fleet_dir = os.path.join(output_dir, f"fleet_{self.fleet_id}")
        os.makedirs(fleet_dir, exist_ok=True)
        if self.anomaly_records:
            filepath = os.path.join(fleet_dir, f"fleet_{self.fleet_id}_anomalies.csv")
            df = pd.DataFrame(self.anomaly_records)
            df.to_csv(filepath, index=False)
            print(f"Anomaly summary saved as {os.path.abspath(filepath)}")
