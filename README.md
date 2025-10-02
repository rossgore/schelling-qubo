# Schelling Model QUBO Implementation

A quantum optimization approach to the Schelling segregation model using QUBO (Quadratic Unconstrained Binary Optimization) formulation and QAOA (Quantum Approximate Optimization Algorithm).

## Overview

This repository implements the Schelling segregation model as a quantum optimization problem. The classical agent-based model is mapped to quantum representations using different encoding schemes, allowing for quantum optimization techniques to find socially optimal arrangements.

## Features

- **Classical Schelling Model**: Traditional agent-based implementation for comparison
- **Multiple Quantum Encodings**:
  - Ising spin encoding (1 qubit per agent)
  - Basis state encoding (2 qubits per agent) 
  - One-hot encoding (3 qubits per agent)
- **QAOA Implementation**: Quantum Approximate Optimization Algorithm for finding optimal configurations
- **Resource Estimation**: Circuit depth and gate count analysis for quantum hardware feasibility

## Repository Structure

```
schelling-qubo/
├── README.md
├── requirements.txt
├── schelling/
│   ├── __init__.py
│   ├── classical.py          # Classical Schelling model
│   ├── grid_utils.py         # Grid utilities and neighbor functions
│   └── visualization.py      # Plotting and visualization
├── quantum/
│   ├── __init__.py
│   ├── encodings/
│   │   ├── __init__.py
│   │   ├── ising.py          # Ising spin encoding
│   │   ├── basis_state.py    # 2-qubit basis state encoding
│   │   └── one_hot.py        # 3-qubit one-hot encoding
│   ├── qaoa.py               # QAOA implementation
│   └── resource_estimation.py
├── examples/
│   ├── basic_simulation.py
│   ├── quantum_comparison.py
│   └── resource_analysis.py
└── tests/
    ├── test_classical.py
    ├── test_encodings.py
    └── test_qaoa.py
```

## Installation

```bash
git clone https://github.com/your-username/schelling-qubo.git
cd schelling-qubo
pip install -r requirements.txt
```

## Quick Start

### Classical Simulation

```python
from schelling.classical import SchellingModel

# Initialize model
model = SchellingModel(
    size=3,
    colors={'red': 3, 'blue': 3, 'empty': 3},
    similarity_threshold=0.5
)

# Run simulation
model.simulate(max_steps=10)
model.visualize()
```

### Quantum Optimization

```python
from quantum.encodings.basis_state import BasisStateEncoder
from quantum.qaoa import run_qaoa

# Create quantum representation
encoder = BasisStateEncoder(size=3, colors={'red': 3, 'blue': 3, 'empty': 3})
hamiltonian = encoder.build_hamiltonian()

# Run QAOA
cost, params = run_qaoa(hamiltonian, reps=5)
```

## Quantum Encodings

### 1. Ising Encoding (1 qubit per agent)
- Maps red agents to spin-up (+1) and blue agents to spin-down (-1)
- Most qubit-efficient but limited to two agent types
- Empty spaces handled through occupancy indicators

### 2. Basis State Encoding (2 qubits per agent)
- Four states: |00⟩ (empty), |01⟩ (red), |10⟩ (blue), |11⟩ (penalized)
- Balances efficiency with flexibility
- **Recommended for most applications**

### 3. One-Hot Encoding (3 qubits per agent)
- Dedicated qubit for each state: empty, red, blue
- Most intuitive but highest qubit cost
- Suitable for small grids or proof-of-concept work

## Mathematical Formulation

The Schelling model unhappiness is encoded as a Hamiltonian:

### Basis State Encoding
**Penalty Hamiltonian** (penalizes invalid |11⟩ states):
```
H_p = J ∑_i P^(i)_11 = J ∑_i (1/4)(I - Z) ⊗ (I - Z)
```

**Interaction Hamiltonian** (penalizes unlike neighbors):
```
H_int = J ∑_{⟨i,j⟩} (q_{i,R} q_{j,B} + q_{i,B} q_{j,R})
```

**Total Hamiltonian**:
```
H = H_p + H_int + H_count + H_threshold
```

## Performance and Scaling

- **Small grids (3x3)**: Suitable for NISQ devices with ~18 qubits
- **Circuit depth**: O(reps × edges) for QAOA layers  
- **Classical preprocessing**: Reduces problem size through constraint elimination
- **Hybrid approach**: Combines quantum optimization with classical post-processing

## Research Applications

This implementation supports research in:
- Quantum advantage in combinatorial optimization
- Social dynamics modeling with quantum algorithms
- NISQ algorithm development and benchmarking
- Hybrid classical-quantum optimization strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## References

- Schelling, T. C. (1971). Dynamic models of segregation. Journal of Mathematical Sociology.
- Farhi, E., & Gutmann, S. (2014). The Quantum Approximate Optimization Algorithm.
- Lucas, A. (2014). Ising formulations of many NP problems.

## License

MIT License - see LICENSE file for details.