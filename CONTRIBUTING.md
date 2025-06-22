# Contributing to AnomFL

Thank you for your interest in contributing to AnomFL! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

## Development Setup

### Code Style
We use the following tools for code quality:
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

Run them before committing:
```bash
black src/ tests/ examples/
flake8 src/ tests/ examples/
mypy src/
```

### Testing
Run tests with pytest:
```bash
pytest tests/
```

For coverage:
```bash
pytest --cov=src tests/
```

## Project Structure

```
AnomFL/
├── src/                    # Main source code
│   ├── data_generation/    # Aircraft data simulation
│   ├── autoencoders/       # Neural network models
│   ├── federated/          # Federated learning implementation
│   └── data/              # Generated datasets
├── tests/                 # Unit tests
├── examples/              # Usage examples
├── docs/                  # Documentation
├── config.yaml           # Configuration file
└── setup.py              # Package setup
```

## Areas for Contribution

### 1. **New Anomaly Types**
- Add new anomaly patterns in `aircraft_data_generator.py`
- Implement realistic sensor behavior changes
- Add corresponding tests

### 2. **Federated Learning Algorithms**
- Implement new aggregation methods (FedProx, FedNova, etc.)
- Add client selection strategies
- Implement differential privacy

### 3. **Model Architectures**
- Add different autoencoder architectures
- Implement other anomaly detection models
- Add support for different data types

### 4. **Evaluation Metrics**
- Add more comprehensive evaluation metrics
- Implement visualization tools
- Add statistical significance testing

### 5. **Real-world Integration**
- Add support for real sensor data formats
- Implement data preprocessing pipelines
- Add deployment utilities

## Pull Request Guidelines

1. **Create a feature branch** from `main`
2. **Write clear commit messages** following conventional commits
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Ensure all tests pass** before submitting
6. **Provide a clear description** of changes

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(data): add new anomaly type for engine stall`
- `fix(federated): resolve weight aggregation bug`
- `docs(readme): update installation instructions`

## Code Review Process

1. **Automated checks** must pass (tests, linting, type checking)
2. **At least one maintainer** must approve
3. **All conversations resolved** before merge
4. **Squash and merge** to maintain clean history

## Reporting Issues

When reporting issues, please include:
- **Environment details** (OS, Python version, dependencies)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages** and stack traces
- **Minimal example** if possible

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the README and docstrings first

## License

By contributing to AnomFL, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to AnomFL! 🚀 