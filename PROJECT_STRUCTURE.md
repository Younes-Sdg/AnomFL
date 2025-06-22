# AnomFL Project Structure

This document explains the improved project structure and the reasoning behind the changes.

## Current Issues Identified

### 1. **Mixed Data Locations**
- **Problem**: Data stored in both `src/data/` and `src/Notebooks/data/`
- **Solution**: Centralize data in `data/` directory at root level
- **Benefit**: Clear separation of data from source code

### 2. **Notebooks Mixed with Source Code**
- **Problem**: Jupyter notebooks in `src/Notebooks/` mixed with core functionality
- **Solution**: Move notebooks to `examples/` directory
- **Benefit**: Clear separation of examples from core library code

### 3. **Missing Essential Files**
- **Problem**: No proper package setup, tests, or configuration
- **Solution**: Added `setup.py`, `tests/`, `config.yaml`, etc.
- **Benefit**: Professional Python package structure

### 4. **Poor Documentation**
- **Problem**: Limited documentation and contribution guidelines
- **Solution**: Added comprehensive docs and contributing guidelines
- **Benefit**: Easier onboarding for new contributors

## Improved Project Structure

```
AnomFL/
├── src/                           # Main source code (installable package)
│   ├── __init__.py               # Package initialization
│   ├── data_generation/          # Aircraft data simulation
│   │   ├── __init__.py
│   │   └── aircraft_data_generator.py
│   ├── autoencoders/             # Neural network models
│   │   ├── __init__.py
│   │   └── autoencoder.py
│   ├── federated/                # Federated learning implementation
│   │   ├── __init__.py
│   │   └── centralized_server.py
│   └── utils/                    # Utility functions (future)
│       └── __init__.py
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_autoencoders.py
│   └── test_federated.py
├── examples/                     # Usage examples and notebooks
│   ├── __init__.py
│   ├── basic_usage.py
│   ├── centralized_example.ipynb
│   ├── autoencoder_aircraft.ipynb
│   └── data_generation_example.ipynb
├── docs/                         # Documentation (future)
│   ├── api/
│   ├── tutorials/
│   └── index.md
├── data/                         # Generated data (gitignored)
│   └── aircraft_data/
├── outputs/                      # Experiment outputs (gitignored)
├── models/                       # Saved models (gitignored)
├── logs/                         # Log files (gitignored)
├── config.yaml                   # Configuration file
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
├── requirements-dev.txt          # Development dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # Project documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # License file
└── PROJECT_STRUCTURE.md          # This file
```

## Key Improvements

### 1. **Proper Python Package Structure**
- **`setup.py`**: Enables `pip install -e .` for development
- **`src/__init__.py`**: Makes src a proper package with version info
- **Entry points**: Command-line tools for data generation and training

### 2. **Configuration Management**
- **`config.yaml`**: Centralized configuration for all components
- **Environment-specific settings**: Easy to modify without code changes
- **Default values**: Sensible defaults for all parameters

### 3. **Testing Infrastructure**
- **`tests/` directory**: Dedicated space for unit tests
- **Test coverage**: Comprehensive testing of all modules
- **CI/CD ready**: Structure supports automated testing

### 4. **Examples and Documentation**
- **`examples/` directory**: Clear separation from core code
- **Multiple formats**: Both scripts and notebooks
- **Progressive complexity**: From basic to advanced examples

### 5. **Development Tools**
- **`.gitignore`**: Comprehensive exclusions for Python projects
- **Code quality tools**: Black, flake8, mypy support
- **Development dependencies**: Separate requirements for dev tools

## Migration Plan

### Phase 1: Core Structure (✅ Completed)
- [x] Create `setup.py` and package structure
- [x] Add configuration file
- [x] Create tests directory
- [x] Update `.gitignore`
- [x] Add contributing guidelines

### Phase 2: Data Reorganization
- [ ] Move data from `src/data/` to root `data/`
- [ ] Update all data paths in code
- [ ] Remove duplicate data in `src/Notebooks/data/`

### Phase 3: Examples Migration
- [ ] Move notebooks from `src/Notebooks/` to `examples/`
- [ ] Update import paths in notebooks
- [ ] Create additional example scripts

### Phase 4: Documentation
- [ ] Create `docs/` directory
- [ ] Add API documentation
- [ ] Create tutorials
- [ ] Add type hints to all functions

### Phase 5: Advanced Features
- [ ] Add logging system
- [ ] Implement CLI tools
- [ ] Add experiment tracking
- [ ] Create deployment utilities

## Benefits of New Structure

### 1. **Professional Quality**
- Follows Python packaging best practices
- Suitable for publication on PyPI
- Industry-standard project layout

### 2. **Developer Experience**
- Easy installation and development setup
- Clear separation of concerns
- Comprehensive testing and documentation

### 3. **Scalability**
- Easy to add new modules
- Configurable without code changes
- Extensible architecture

### 4. **Collaboration**
- Clear contribution guidelines
- Automated code quality checks
- Professional documentation

### 5. **Maintainability**
- Organized file structure
- Comprehensive testing
- Clear dependencies

## Usage Examples

### Development Installation
```bash
git clone <repository>
cd AnomFL
pip install -e .[dev]
```

### Running Tests
```bash
pytest tests/
pytest --cov=src tests/
```

### Running Examples
```bash
python examples/basic_usage.py
jupyter notebook examples/
```

### Command Line Tools (after setup)
```bash
anomfl-generate-data --fleet-id 1 --num-aircraft 5
anomfl-train --rounds 10 --local-epochs 100
```

This improved structure makes AnomFL a professional, maintainable, and scalable research project that can easily grow and attract contributors. 