# Contributing to GraphMind

Thank you for your interest in GraphMind! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project adheres to a standard code of conduct. Please be respectful and constructive in all interactions.

---

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GraphMind.git
   cd GraphMind
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/Jdepp007004/GraphMind.git
   ```

---

## Development Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Set up the dashboard
cd dashboard && npm install && cd ..
```

---

## Project Structure

The project is organized into these key areas:

| Directory | Purpose |
|-----------|---------|
| `src/core/` | Core graph engine, memory manager, event bus |
| `src/rl/` | Reinforcement learning environment and trainer |
| `src/agents/` | Multi-agent orchestration system |
| `src/prefetch/` | Production prefetch engine (frozen) |
| `src/android/` | Android device integration |
| `scripts/` | Benchmark and analysis scripts |
| `dashboard/` | Next.js interactive dashboard |
| `tests/` | Full test suite |
| `docs/` | Documentation |

---

## Coding Standards

### Python

- Follow **PEP 8** style guidelines
- Use **type hints** for all function signatures
- Write **docstrings** for all public classes and functions
- Maximum line length: **100 characters**

```python
def predict_next_apps(
    current_app: str,
    graph: nx.DiGraph,
    threshold: float = 0.16
) -> list[tuple[str, float]]:
    """
    Predict the next apps a user will likely open.

    Args:
        current_app: The currently active application package name.
        graph: The user's Markov behaviour graph.
        threshold: Minimum confidence score for inclusion.

    Returns:
        List of (app_name, confidence_score) tuples, sorted by score descending.
    """
    ...
```

### TypeScript (Dashboard)

- Use **TypeScript strict mode**
- Follow the existing component structure in `dashboard/components/`
- Use **Recharts** for data visualization
- Use **Framer Motion** for animations

---

## Testing

All changes must include appropriate tests.

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run a specific test file
pytest tests/test_confidence_prefetch.py -v

# Run the official benchmark (frozen — do not modify)
python scripts/run_phase11_e.py
```

### Test Guidelines

- Each module in `src/` should have corresponding tests in `tests/`
- Use `pytest` fixtures defined in `tests/conftest.py`
- Mock external dependencies (filesystem, network, ADB)
- Aim for **>80% coverage** on new code

---

## Submitting Changes

### Branch Naming

- `feat/your-feature-name` — new features
- `fix/issue-description` — bug fixes
- `docs/update-description` — documentation updates
- `refactor/what-changed` — code refactoring

### Commit Messages

Follow the **Conventional Commits** specification:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

Examples:
```
feat(agents): add confidence decay mechanism to prefetch agent
fix(core): resolve memory leak in graph engine edge cleanup
docs(readme): add dashboard screenshot and setup instructions
test(rl): add edge case tests for reward function boundary conditions
```

### Pull Request Process

1. Update the README if your changes affect public interfaces
2. Add/update tests for your changes
3. Ensure all tests pass: `pytest tests/ -v`
4. Update `docs/` if you change architecture or APIs
5. Create a pull request with a clear description

---

## ⚠️ Important: Frozen Components

The following components are **frozen** and should not be modified:

- `src/prefetch/confidence_prefetch.py` — Production prefetch engine
- `config/settings.py` — Production configuration
- `results/final_production_results.csv` — Official benchmark result

These files represent the submission state for the Samsung EnnovateX AX Hackathon 2025. Any changes to these files invalidate the official result.

---

*GraphMind — Samsung EnnovateX AX Hackathon 2025*
