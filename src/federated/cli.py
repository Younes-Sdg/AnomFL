"""
Command-line interface for federated learning training.
"""

import argparse
import sys
import os
import glob
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from centralized_server import CentralizedServer, FederatedClient
from utils.config import load_config, ensure_directories, get_data_path


def main():
    """Main CLI function for federated learning training."""
    parser = argparse.ArgumentParser(
        description="Train federated learning model for anomaly detection"
    )
    
    parser.add_argument(
        "--fleet-id", 
        type=int, 
        default=1,
        help="Fleet ID to train on (default: 1)"
    )
    
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=10,
        help="Number of federated learning rounds (default: 10)"
    )
    
    parser.add_argument(
        "--local-epochs", 
        type=int, 
        default=100,
        help="Number of local training epochs per round (default: 100)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-3,
        help="Learning rate (default: 1e-3)"
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
    
    # Find CSV files for the fleet
    data_path = get_data_path()
    csv_pattern = os.path.join(data_path, "aircraft_data", f"fleet_{args.fleet_id}", "aircraft_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ No CSV files found for fleet {args.fleet_id}")
        print(f"   Expected pattern: {csv_pattern}")
        print("   Please run data generation first: python -m src.data_generation.cli")
        sys.exit(1)
    
    csv_files.sort()
    print(f"📊 Found {len(csv_files)} aircraft data files")
    
    # Create federated clients
    clients = []
    for idx, csv_path in enumerate(csv_files):
        client = FederatedClient(client_id=idx, file_paths=[csv_path])
        clients.append(client)
        print(f"   Client {idx}: {os.path.basename(csv_path)}")
    
    # Create server and train
    server = CentralizedServer(clients)
    
    print(f"🚀 Starting federated learning training...")
    print(f"   Rounds: {args.rounds}")
    print(f"   Local epochs: {args.local_epochs}")
    print(f"   Learning rate: {args.learning_rate}")
    
    server.train(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.learning_rate
    )
    
    # Evaluate results
    print(f"\n📈 Training completed! Evaluating results...")
    results = server.benchmark_all_clients()
    print("\nClient Performance Summary:")
    print(results)
    
    # Detect anomalies
    print(f"\n🔍 Anomaly detection results:")
    for client_id in range(len(clients)):
        anomalies = server.detect_anomalies(client_id=client_id, k=3.0)
        print(f"   Client {client_id}: {len(anomalies)} anomalies detected")
    
    print(f"\n✅ Federated learning completed successfully!")


if __name__ == "__main__":
    main() 