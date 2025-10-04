"""
Quantum encoding schemes for the Schelling model.
"""

from .basis_state import BasisStateEncoder
from .ising import IsingEncoder
from .one_hot import OneHotEncoder

__all__ = [
    'BasisStateEncoder',
    'IsingEncoder', 
    'OneHotEncoder'
]