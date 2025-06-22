"""
Command-line interface for data generation.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from aircraft_data_generator import Fleet
from utils.config import load_config, ensure_directories


def main():
    """Main CLI function for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate aircraft sensor data with anomalies"
    )
    
    parser.add_argument(
        "--fleet-id", 
        type=int, 
        default=1,
        help="Fleet ID (default: 1)"
    )
    
    parser.add_argument(
        "--num-aircraft", 
        type=int, 
        default=5,
        help="Number of aircraft in fleet (default: 5)"
    )
    
    parser.add_argument(
        "--num-anomalous", 
        type=int, 
        default=1,
        help="Number of aircraft with anomalies (default: 1)"
    )
    
    parser.add_argument(
        "--num-points", 
        type=int, 
        default=200,
        help="Number of data points per aircraft (default: 200)"
    )
    
    parser.add_argument(
        "--interval-minutes", 
        type=int, 
        default=5,
        help="Time interval between data points in minutes (default: 5)"
    )
    
    parser.add_argument(
        "--anomaly-duration", 
        type=int, 
        default=50,
        help="Duration of anomaly in data points (default: 50)"
    )
    
    parser.add_argument(
        "--anomaly-features", 
        nargs="+",
        default=["engine_rpm", "fuel_flow", "engine_temperature"],
        help="Features to affect with anomalies (default: engine_rpm fuel_flow engine_temperature)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load configuration if available
    try:
        config = load_config(args.config)
        print(f"✅ Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"⚠️ Configuration file {args.config} not found, using command line arguments")
        config = {}
    
    # Ensure directories exist
    ensure_directories()
    
    # Create fleet
    fleet = Fleet(
        fleet_id=args.fleet_id,
        num_aircraft=args.num_aircraft,
        num_anomalous=args.num_anomalous,
        num_points=args.num_points,
        interval_minutes=args.interval_minutes,
        anomaly_duration=args.anomaly_duration,
        anomaly_features=args.anomaly_features
    )
    
    print(f"🚀 Generating data for fleet {args.fleet_id}...")
    print(f"   Aircraft: {args.num_aircraft}")
    print(f"   Anomalous: {args.num_anomalous}")
    print(f"   Data points: {args.num_points}")
    print(f"   Interval: {args.interval_minutes} minutes")
    
    # Generate data
    fleet.generate_fleet_data()
    
    print(f"✅ Data generation completed!")
    print(f"📁 Data saved to: data/aircraft_data/fleet_{args.fleet_id}/")


if __name__ == "__main__":
    main() 