"""
Schelling Model QUBO Implementation

A quantum optimization approach to the classical Schelling segregation model.
"""

__version__ = "1.0.0"
__author__ = "Schelling QUBO Research Team"

# Import main classes for easy access
from .classical import SchellingModel
from .grid_utils import get_neighbors, calculate_grid_energy
from .visualization import plot_grid, plot_grid_evolution

__all__ = [
    'SchellingModel',
    'get_neighbors', 
    'calculate_grid_energy',
    'plot_grid',
    'plot_grid_evolution'
]