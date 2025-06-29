print("\n--- Training Federated Model ---")

# Create a client for each aircraft
clients = [
    LinearRegressionFederatedClient(
        client_id=i + 1, 
        file_path=file_paths[i],
        target_sensor=MODEL_CONFIG['target_sensor']
    )
    for i in range(DATA_CONFIG['num_aircraft'])
]

# Create a base model instance for the server
base_model = LinearRegression(
    input_dim=MODEL_CONFIG['input_dim'],
    output_dim=MODEL_CONFIG['output_dim']
)

# Instantiate the server
server = LinearRegressionCentralizedServer(clients=clients, model=base_model)

# Run the federated training process (hide output)
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Train with suppressed output
with suppress_stdout():
    server.train(
        rounds=TRAINING_CONFIG['federated_rounds'],
        local_epochs=TRAINING_CONFIG['federated_local_epochs'],
        lr=TRAINING_CONFIG['learning_rate']
    )

federated_model = server.global_model

# Calculate and print final MSE statistic
federated_model.eval()
with torch.no_grad():
    predictions = federated_model(input_tensor)
    final_loss = nn.MSELoss()(predictions, target_tensor)
print(f"\n📊 Final MSE on Normal Training Data (Federated): {final_loss.item():.8f}") 