"""
Basic usage example for AnomFL.

This script demonstrates the complete workflow:
1. Generate aircraft data with anomalies
2. Set up federated learning clients
3. Train the federated model
4. Detect anomalies
"""

import os
import sys
import glob
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.aircraft_data_generator import Fleet
from federated.centralized_server import CentralizedServer, FederatedClient


def main():
    """Main function demonstrating AnomFL usage."""
    
    print("🚀 Starting AnomFL Basic Usage Example")
    
    # Step 1: Generate data
    print("\n📊 Step 1: Generating aircraft data...")
    
    # Clean up existing data
    data_path = os.path.join("..", "data")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    
    # Create fleet with anomalies
    fleet = Fleet(
        fleet_id=1,
        num_aircraft=5,
        num_anomalous=1,
        num_points=200,
        interval_minutes=5,
        anomaly_duration=50,
        anomaly_features=["engine_rpm", "fuel_flow", "engine_temperature"]
    )
    
    fleet.generate_fleet_data()
    
    # Step 2: Set up federated clients
    print("\n🛩️ Step 2: Setting up federated clients...")
    
    csv_files = glob.glob(os.path.join("..", "data", "aircraft_data", "fleet_1", "aircraft_*.csv"))
    csv_files.sort()
    
    clients = []
    for idx, path in enumerate(csv_files):
        clients.append(FederatedClient(client_id=idx, file_paths=[path]))
    
    print(f"Created {len(clients)} federated clients")
    
    # Step 3: Train federated model
    print("\n🧠 Step 3: Training federated model...")
    
    server = CentralizedServer(clients)
    server.train(rounds=5, local_epochs=50, lr=1e-3)
    
    # Step 4: Evaluate and detect anomalies
    print("\n🔍 Step 4: Evaluating anomaly detection...")
    
    # Benchmark all clients
    results = server.benchmark_all_clients()
    print("\nClient Performance Summary:")
    print(results)
    
    # Detect anomalies in the anomalous client (client 4)
    anomalies = server.detect_anomalies(client_id=4, k=3.0)
    print(f"\nDetected {len(anomalies)} anomalies in client 4")
    
    # Evaluate specific client
    errs, threshold, prop_anom = server.evaluate_client(client_id=4)
    print(f"Client 4 - Threshold: {threshold:.4f}, Anomaly proportion: {prop_anom:.2%}")
    
    print("\n✅ AnomFL example completed successfully!")


if __name__ == "__main__":
    main() 