import numpy as np
import pandas as pd
import os

class Aircraft:
    def __init__(self, id):
        self.id = id
        self.engine_rpm = None
        self.fuel_flow = None
        self.engine_temperature = None
        self.vibration_level = None

    def generate_data(self, num_points=100):
        """Simulate normal engine data for the aircraft."""
        self.engine_rpm = np.random.normal(2000, 100, num_points)  # RPM
        self.fuel_flow = np.random.normal(500, 50, num_points)  # kg/hr
        self.engine_temperature = np.random.normal(700, 30, num_points)  # Celsius
        self.vibration_level = np.random.normal(1.0, 0.1, num_points)  # G-force

    def to_dataframe(self):
        """Convert the aircraft data into a pandas DataFrame."""
        data = {
            'engine_rpm': self.engine_rpm,
            'fuel_flow': self.fuel_flow,
            'engine_temperature': self.engine_temperature,
            'vibration_level': self.vibration_level
        }
        return pd.DataFrame(data)
