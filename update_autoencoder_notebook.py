import json
import os

def update_autoencoder_notebook():
    """Update the all_data autoencoder notebook to use fixed scaling"""
    
    notebook_path = 'autoencoder_federated_vs_centralized_analysis_all_data.ipynb'
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return
    
    print(f"🔧 Updating {notebook_path} with fixed scaling...")
    
    # Track changes
    changes_made = 0
    
    # 1. Add fixed scaling import after existing imports
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'import' in source and 'torch' in source and 'pandas' in source:
                # Add fixed scaling import after imports
                import_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# --- Fixed Scaling Import ---\n",
                        "from anomfl.utils import get_aircraft_scaler, scale_aircraft_data\n",
                        "print(\"✅ Fixed scaling utilities imported\")\n"
                    ]
                }
                notebook['cells'].insert(i + 1, import_cell)
                print("✅ Added fixed scaling import")
                changes_made += 1
                break
    
    # 2. Replace centralized training scaling
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Training Centralized Model' in source and 'MinMaxScaler' in source:
                # Replace MinMaxScaler with fixed scaling
                new_source = []
                for line in cell['source']:
                    if 'MinMaxScaler()' in line:
                        new_source.append('scaler = get_aircraft_scaler(include_anomalies=True)  # Fixed scaling\n')
                    elif 'scaler.fit_transform' in line:
                        new_source.append(line.replace('scaler.fit_transform', 'scaler.transform'))
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                print("✅ Updated centralized training scaling")
                changes_made += 1
    
    # 3. Replace plotting cell scaling
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Generating Comparison Plots' in source and 'scaler.transform' in source:
                # Replace global scaler with fixed scaling
                new_source = []
                for line in cell['source']:
                    if 'client_tensor = torch.tensor(scaler.transform' in line:
                        new_source.append('    # Scale data using fixed ranges (same as training)\n')
                        new_source.append('    scaler = get_aircraft_scaler(include_anomalies=True)\n')
                        new_source.append('    client_tensor = torch.tensor(scaler.transform(client_df[DATA_CONFIG[\'sensor_names\']]), dtype=torch.float32)\n')
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                print("✅ Updated plotting cell scaling")
                changes_made += 1
    
    # 4. Replace model performance statistics scaling
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Model Performance Statistics' in source and 'global_scaler.transform' in source:
                # Replace global scaler with fixed scaling
                new_source = []
                for line in cell['source']:
                    if 'scaled_data = global_scaler.transform(sensor_data)' in line:
                        new_source.append('        # Scale data using fixed ranges\n')
                        new_source.append('        scaler = get_aircraft_scaler(include_anomalies=True)\n')
                        new_source.append('        scaled_data = scaler.transform(sensor_data)\n')
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                print("✅ Updated model performance statistics scaling")
                changes_made += 1
    
    # Write the updated notebook
    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"✅ Successfully updated notebook with {changes_made} changes")
        print("🎯 The notebook now uses fixed scaling that doesn't depend on the data!")
        print("   This should give consistent results across experiments.")
    except Exception as e:
        print(f"❌ Error writing notebook: {e}")

if __name__ == "__main__":
    update_autoencoder_notebook() 