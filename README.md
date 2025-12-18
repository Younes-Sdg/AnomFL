# Federated Learning for Aircraft Anomaly Detection (AnomFL)

## Project Overview
This project is an implementation of **Federated Learning (FL)** for detecting anomalies in aircraft engine performance. The system simulates a fleet of aircraft, where each generates its own dataset of engine metrics, including attributes like engine RPM, fuel flow, temperature, and vibration levels. 

The system uses a **centralized federated learning architecture** where a central server orchestrates collaborative learning while preserving data privacy. Each aircraft (client) trains locally on its own data and only shares model weights with the central server, which aggregates them using the FedAvg algorithm. The project demonstrates how **autoencoders** and **federated algorithms** can be used for anomaly detection in distributed systems.

## Quick Start

### 1. Installation
```bash

# Install in development mode
pip install -e .[dev]

# Or use the Makefile
make install-dev
```

### 2. Generate Data
```bash
# Using CLI tool
python -m src.data_generation.cli --fleet-id 1 --num-aircraft 5 --num-anomalous 1

# Or using Makefile
make data
```

### 3. Train Federated Model
```bash
# Using CLI tool
python -m src.federated.cli --fleet-id 1 --rounds 10 --local-epochs 100

# Or using Makefile
make train
```

### 4. Run Complete Example
```bash
# Run the basic usage example
python examples/basic_usage.py

# Or using Makefile
make example
```

## Project Structure and File Descriptions

### Root Directory Files

#### `README.md`
This file - provides comprehensive documentation of the project structure, file purposes, and usage instructions.

#### `setup.py`
Package installation and distribution configuration. Enables `pip install -e .` for development.

#### `requirements.txt`
Lists the core Python dependencies required to run the project:
- `numpy`: Numerical computing and array operations
- `pandas`: Data manipulation and analysis
- `torch`: PyTorch deep learning framework

#### `requirements-dev.txt`
Development dependencies including testing, code quality, and documentation tools.

#### `config.yaml`
Centralized configuration file for all project settings including data generation, autoencoder, and federated learning parameters.

#### `Makefile`
Common development tasks and workflows:
- `make install-dev`: Install development dependencies
- `make test`: Run tests
- `make data`: Generate sample data
- `make train`: Run federated learning
- `make example`: Run complete example

#### `LICENSE`
Contains the project's license terms and conditions.

#### `CONTRIBUTING.md`
Guidelines for contributing to the project.

#### `PROJECT_STRUCTURE.md`
Detailed explanation of the project structure and improvements.

### Source Code (`src/`)

#### Data Generation Module (`src/data_generation/`)

##### `__init__.py`
Python package initialization file for the data generation module.

##### `aircraft_data_generator.py`
**Main data generation engine** that creates synthetic aircraft sensor data:

**Classes:**
- **`Aircraft`**: Represents a single aircraft with sensor simulation
  - Generates realistic engine data (RPM, fuel flow, temperature, vibration)
  - Injects controlled anomalies (engine stall, fuel leak, turbulence, overheat)
  - Saves data to CSV files with timestamps
  - Tracks anomaly information for evaluation

- **`Fleet`**: Manages multiple aircraft and coordinates data generation
  - Creates configurable fleet sizes
  - Controls anomaly injection across the fleet
  - Generates both normal and anomalous datasets
  - Saves anomaly summaries for analysis

##### `cli.py`
**Command-line interface** for data generation:
```bash
python -m src.data_generation.cli --fleet-id 1 --num-aircraft 5 --num-anomalous 1
```

#### Autoencoder Module (`src/autoencoders/`)

##### `__init__.py`
Python package initialization file for the autoencoder module.

##### `autoencoder.py`
**Deep learning model for anomaly detection** using autoencoders:

**Classes:**
- **`Autoencoder`**: Neural network architecture for unsupervised anomaly detection
  - **Encoder**: Compresses 4D sensor data to 2D latent representation
  - **Decoder**: Reconstructs original data from latent space
  - **Training**: Learns normal patterns from sensor data
  - **Detection**: Uses reconstruction error to identify anomalies

#### Federated Learning Module (`src/federated/`)

##### `centralized_server.py`
**Centralized federated learning orchestration and anomaly detection system**:

**Classes:**
- **`FederatedClient`**: Represents each aircraft in the federated network
  - Local autoencoder model for each aircraft
  - Local training on aircraft's own sensor data
  - Weight sharing capabilities (get/set model weights)
  - Reconstruction error computation for anomaly detection

- **`CentralizedServer`**: Orchestrates the federated learning process using FedAvg
  - **FedAvg Algorithm**: Implements Federated Averaging for weight aggregation
  - **Training Coordination**: Manages rounds of local training and global aggregation
  - **Centralized Control**: Distributes global model to all clients and collects their updates
  - **Anomaly Detection**: Evaluates reconstruction errors across all clients
  - **Benchmarking**: Provides performance metrics and anomaly analysis

##### `cli.py`
**Command-line interface** for federated learning:
```bash
python -m src.federated.cli --fleet-id 1 --rounds 10 --local-epochs 100
```

#### Utilities Module (`src/utils/`)

##### `config.py`
Configuration management utilities:
- Load configuration from YAML files
- Path management for data, outputs, models, and logs
- Directory creation utilities

##### `logging.py`
Logging setup and utilities:
- Colored console logging
- File logging with timestamps
- Configurable log levels and formats

### Data and Output Directories

#### `data/`
Centralized data directory containing generated aircraft datasets:
- `aircraft_data/fleet_1/` - Aircraft sensor data files
- `aircraft_data/fleet_1/aircraft_1_1_normal.csv` - Normal aircraft data
- `aircraft_data/fleet_1/aircraft_1_5_anomalous.csv` - Anomalous aircraft data
- `aircraft_data/fleet_1/fleet_1_anomalies.csv` - Anomaly summary

#### `outputs/`
Directory for experiment outputs and results (gitignored).

#### `models/`
Directory for saved model files (gitignored).

#### `logs/`
Directory for log files (gitignored).

### Testing (`tests/`)

#### `test_data_generation.py`
Unit tests for the data generation module:
- Aircraft class initialization and data generation
- Fleet class management and anomaly injection
- Data format validation

#### `test_autoencoders.py`
Unit tests for the autoencoder module:
- Model architecture validation
- Forward pass testing
- Reconstruction error calculation
- Data loading from CSV

#### `test_federated.py`
Unit tests for the federated learning module:
- Client initialization and weight management
- Server initialization and FedAvg implementation
- Anomaly detection evaluation

### Examples (`examples/`)

#### `basic_usage.py`
**Complete workflow example** showing the entire process:
- Data generation for a fleet of aircraft
- Federated client setup and training
- Anomaly detection evaluation
- Performance analysis

#### `centralized_example.ipynb`
**Jupyter notebook** demonstrating federated learning workflow with visualizations.

#### `autoencoder_aircraft.ipynb`
**Autoencoder experiments** and analysis notebook.

#### `data_generation_example.ipynb`
**Data generation demonstration** notebook.

## Usage Workflow

### 1. Environment Setup
```bash
# Install the package
pip install -e .[dev]

# Or use Makefile
make install-dev
```

### 2. Data Generation
```python
from src.data_generation.aircraft_data_generator import Fleet

# Create a fleet with anomalies
fleet = Fleet(fleet_id=1, num_aircraft=5, num_anomalous=1)
fleet.generate_fleet_data()
```

### 3. Federated Learning
```python
from src.federated.centralized_server import CentralizedServer, FederatedClient

# Setup federated clients
clients = [FederatedClient(client_id=i, file_paths=[csv_path]) for i, csv_path in enumerate(csv_files)]

# Train federated model
server = CentralizedServer(clients)
server.train(rounds=10, local_epochs=100)
```

### 4. Anomaly Detection
```python
# Evaluate anomaly detection performance
results = server.benchmark_all_clients()
anomalies = server.detect_anomalies(client_id=4, k=3.0)
```

## Configuration

The project uses `config.yaml` for centralized configuration:

```yaml
# Data Generation Settings
data_generation:
  default_fleet_id: 1
  default_num_aircraft: 5
  default_num_anomalous: 1
  
# Autoencoder Settings
autoencoder:
  input_dim: 4
  latent_dim: 2
  learning_rate: 0.001
  
# Federated Learning Settings
federated:
  default_rounds: 10
  default_local_epochs: 100
  anomaly_threshold_k: 3.0
```

## Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Or use Makefile
make test
make test-cov
```

### Code Quality
```bash
# Format code
black src/ tests/ examples/

# Run linting
flake8 src/ tests/ examples/

# Or use Makefile
make format
make lint
```

### Development Workflow
```bash
# Complete development setup
make dev-setup

# Generate data and train
make workflow

# Clean up
make clean
```

## Key Features

### Privacy-Preserving Learning
- Raw sensor data never leaves individual aircraft
- Only model weights are shared with the central server
- Each aircraft maintains complete control over its data
- Central server never sees actual sensor readings

### Centralized Federated Learning
- Single central server orchestrates the learning process
- Standard FedAvg (Federated Averaging) algorithm implementation
- Aircraft learn from each other's normal patterns through weight aggregation
- Global model benefits from diverse data sources across the fleet

### Professional Project Structure
- Proper Python package with `setup.py`
- Comprehensive testing framework
- Configuration management
- Command-line interfaces
- Development tools and workflows

### Realistic Simulation
- Time-series data with realistic sensor values
- Multiple anomaly types affecting different sensors
- Configurable fleet sizes and anomaly patterns

### Comprehensive Evaluation
- Reconstruction error analysis per client
- Anomaly detection benchmarking
- Performance metrics comparison across fleet

## Use Cases

1. **Aircraft Fleet Monitoring**: Detect engine issues across multiple aircraft
2. **Predictive Maintenance**: Identify potential failures before they occur
3. **Privacy-Sensitive Industries**: Healthcare, finance, defense applications
4. **Distributed IoT Systems**: Smart cities, industrial IoT, sensor networks

## Technical Requirements

- **Python 3.7+**
- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Jupyter**: Interactive development and experimentation

## Contributing

See `CONTRIBUTING.md` for detailed guidelines on:
- Development setup
- Code style and quality
- Testing requirements
- Pull request process
- Areas for contribution

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
