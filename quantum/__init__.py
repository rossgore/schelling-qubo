"""
Quantum encodings and algorithms for Schelling model optimization.
"""

__version__ = "1.0.0"

# Import main quantum components
from .qaoa import run_qaoa, evaluate_qaoa_circuit
from .resource_estimation import estimate_circuit_resources

__all__ = [
    'run_qaoa',
    'evaluate_qaoa_circuit', 
    'estimate_circuit_resources'
]