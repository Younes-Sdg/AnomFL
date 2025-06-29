print("\n--- Generating Comparison Plots ---")

# Helper function to get prediction errors
def get_errors(model, input_tensor, target_tensor):
    with torch.no_grad():
        model.eval()
        predictions = model(input_tensor)
        return torch.mean((target_tensor - predictions)**2, dim=1).numpy()

# Calculate thresholds using normal data only for better anomaly detection
# Get normal file paths
normal_file_paths = [p for p in file_paths if "anomalous" not in p]

# Collect normal data for threshold calculation
normal_dfs = [pd.read_csv(p) for p in normal_file_paths]
normal_data_df = pd.concat(normal_dfs, ignore_index=True)
normal_input_data = normal_data_df[input_sensors]
normal_target_data = normal_data_df[[MODEL_CONFIG['target_sensor']]]

normal_scaled_input = input_scaler.transform(normal_input_data)
normal_scaled_target = target_scaler.transform(normal_target_data)

normal_input_tensor = torch.tensor(normal_scaled_input, dtype=torch.float32)
normal_target_tensor = torch.tensor(normal_scaled_target, dtype=torch.float32)

# Calculate thresholds using normal data only
fed_normal_errors = get_errors(federated_model, normal_input_tensor, normal_target_tensor)
cen_normal_errors = get_errors(centralized_model, normal_input_tensor, normal_target_tensor)

fed_threshold = np.mean(fed_normal_errors) + 3 * np.std(fed_normal_errors)
cen_threshold = np.mean(cen_normal_errors) + 3 * np.std(cen_normal_errors)

print(f"Federated threshold (normal data): {fed_threshold:.6f}")
print(f"Centralized threshold (normal data): {cen_threshold:.6f}")

# Generate a plot for each aircraft
for i, file_path in enumerate(file_paths):
    client_id = i + 1
    
    # Load the aircraft's full dataset
    client_df = pd.read_csv(file_path)
    timestamps = pd.to_datetime(client_df['timestamp'])
    
    # Prepare input and target data for this client
    client_input_data = client_df[input_sensors]
    client_target_data = client_df[[MODEL_CONFIG['target_sensor']]]
    
    # Scale data using the GLOBAL scalers
    client_input_tensor = torch.tensor(input_scaler.transform(client_input_data), dtype=torch.float32)
    client_target_tensor = torch.tensor(target_scaler.transform(client_target_data), dtype=torch.float32)

    # Get errors for both models
    fed_errors = get_errors(federated_model, client_input_tensor, client_target_tensor)
    cen_errors = get_errors(centralized_model, client_input_tensor, client_target_tensor)
    
    # Identify anomalies using the thresholds calculated from normal data
    fed_anomalies = fed_errors > fed_threshold
    cen_anomalies = cen_errors > cen_threshold

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)
    fig.suptitle(f'Aircraft {client_id}: Federated vs. Centralized Anomaly Detection', fontsize=18)

    # Federated Plot
    ax1.plot(timestamps, fed_errors, label='MSE', color='crimson')
    ax1.axhline(fed_threshold, color='navy', linestyle='--', label=f'Threshold ({fed_threshold:.4f})')
    ax1.scatter(timestamps[fed_anomalies], fed_errors[fed_anomalies], s=60, color='gold', edgecolor='black', label='Anomalous Points', zorder=5)
    ax1.set_title('Federated Model')
    ax1.set_ylabel("Prediction Error (MSE)"); ax1.set_xlabel("Time")
    ax1.grid(True, which='both', linestyle=':'); ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Centralized Plot
    ax2.plot(timestamps, cen_errors, label='MSE', color='crimson')
    ax2.axhline(cen_threshold, color='navy', linestyle='--', label=f'Threshold ({cen_threshold:.4f})')
    ax2.scatter(timestamps[cen_anomalies], cen_errors[cen_anomalies], s=60, color='gold', edgecolor='black', label='Anomalous Points', zorder=5)
    ax2.set_title('Centralized Model')
    ax2.set_xlabel("Time")
    ax2.grid(True, which='both', linestyle=':'); ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show() 