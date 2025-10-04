"""
Unit tests for classical Schelling model implementation.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schelling.classical import SchellingModel
from schelling.grid_utils import get_neighbors, calculate_grid_energy


class TestSchellingModel(unittest.TestCase):
    """Test cases for the classical Schelling model."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {
            'size': 3,
            'colors': {'red': 3, 'blue': 3, 'empty': 3},
            'similarity_threshold': 0.5,
            'wrapping': True,
            'random_seed': 42
        }
        self.model = SchellingModel(**self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.size, 3)
        self.assertEqual(self.model.similarity_threshold, 0.5)
        self.assertTrue(self.model.wrapping)
        self.assertEqual(self.model.grid.shape, (3, 3))
        
        # Check agent counts
        unique, counts = np.unique(self.model.grid, return_counts=True)
        count_dict = dict(zip(unique, counts))
        self.assertEqual(count_dict.get(0, 0), 3)  # Empty
        self.assertEqual(count_dict.get(1, 0), 3)  # Red
        self.assertEqual(count_dict.get(2, 0), 3)  # Blue
    
    def test_get_neighbors(self):
        """Test neighbor calculation."""
        # Test center position with wrapping
        neighbors = self.model.get_neighbors(1, 1)
        self.assertEqual(len(neighbors), 8)  # 8 neighbors for center position
        
        # Test corner position with wrapping
        neighbors = self.model.get_neighbors(0, 0)
        self.assertEqual(len(neighbors), 8)  # Still 8 neighbors with wrapping
        
        # Test non-wrapping model
        non_wrap_model = SchellingModel(
            size=3, 
            colors={'red': 3, 'blue': 3, 'empty': 3},
            wrapping=False,
            random_seed=42
        )
        neighbors = non_wrap_model.get_neighbors(0, 0)
        self.assertEqual(len(neighbors), 3)  # Only 3 neighbors for corner without wrapping
    
    def test_calculate_satisfaction(self):
        """Test satisfaction calculation."""
        # Create known grid configuration
        test_grid = np.array([
            [1, 1, 0],  # Two reds and empty
            [2, 2, 0],  # Two blues and empty
            [0, 0, 0]   # All empty
        ])
        
        self.model.grid = test_grid
        satisfaction = self.model.calculate_satisfaction()
        
        # Check that satisfaction is calculated for agents only
        agent_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for pos in agent_positions:
            self.assertIn(pos, satisfaction)
        
        # Empty positions should not be in satisfaction dict
        empty_positions = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
        for pos in empty_positions:
            self.assertNotIn(pos, satisfaction)
    
    def test_run_step(self):
        """Test single simulation step."""
        initial_grid = self.model.grid.copy()
        moves = self.model.run_step()
        
        # Check that moves is non-negative integer
        self.assertIsInstance(moves, int)
        self.assertGreaterEqual(moves, 0)
        
        # Check that grid still has same agent counts
        initial_counts = np.bincount(initial_grid.flatten(), minlength=3)
        final_counts = np.bincount(self.model.grid.flatten(), minlength=3)
        np.testing.assert_array_equal(initial_counts, final_counts)
    
    def test_simulate(self):
        """Test full simulation."""
        results = self.model.simulate(max_steps=5, verbose=False)
        
        # Check return value structure
        required_keys = ['converged', 'steps_taken', 'total_moves', 
                        'final_satisfaction_rate', 'final_grid', 'grid_history']
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check data types
        self.assertIsInstance(results['converged'], bool)
        self.assertIsInstance(results['steps_taken'], int)
        self.assertIsInstance(results['total_moves'], int)
        self.assertIsInstance(results['final_satisfaction_rate'], float)
        self.assertIsInstance(results['final_grid'], np.ndarray)
        self.assertIsInstance(results['grid_history'], list)
        
        # Check bounds
        self.assertGreaterEqual(results['steps_taken'], 0)
        self.assertLessEqual(results['steps_taken'], 5)  # max_steps
        self.assertGreaterEqual(results['final_satisfaction_rate'], 0.0)
        self.assertLessEqual(results['final_satisfaction_rate'], 1.0)
    
    def test_get_segregation_index(self):
        """Test segregation index calculation."""
        # Test with all same type (maximum segregation)
        uniform_grid = np.array([
            [1, 1, 1],
            [2, 2, 2], 
            [0, 0, 0]
        ])
        self.model.grid = uniform_grid
        segregation = self.model.get_segregation_index()
        self.assertGreater(segregation, 0.5)  # Should be highly segregated
        
        # Test with empty grid
        empty_grid = np.zeros((3, 3))
        self.model.grid = empty_grid
        segregation = self.model.get_segregation_index()
        self.assertEqual(segregation, 0.0)  # No agents, no segregation
    
    def test_get_bit_string_representation(self):
        """Test bit string representation."""
        # Create known grid
        test_grid = np.array([
            [0, 1, 2],  # empty, red, blue
            [1, 0, 2],  # red, empty, blue  
            [2, 1, 0]   # blue, red, empty
        ])
        
        self.model.grid = test_grid
        bit_string = self.model.get_bit_string_representation()
        
        # Check length (2 bits per site)
        expected_length = 2 * 9  # 3x3 grid
        self.assertEqual(len(bit_string), expected_length)
        
        # Check specific encodings
        expected = "000110100110011001"  # 00=empty, 01=red, 10=blue
        self.assertEqual(bit_string, expected)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        model1 = SchellingModel(**self.config)
        model2 = SchellingModel(**self.config)
        
        # Should have identical initial grids
        np.testing.assert_array_equal(model1.grid, model2.grid)
        
        # Should have identical simulation results
        results1 = model1.simulate(max_steps=3, verbose=False)
        results2 = model2.simulate(max_steps=3, verbose=False)
        
        self.assertEqual(results1['steps_taken'], results2['steps_taken'])
        self.assertEqual(results1['total_moves'], results2['total_moves'])
        np.testing.assert_array_equal(results1['final_grid'], results2['final_grid'])


class TestGridUtils(unittest.TestCase):
    """Test cases for grid utility functions."""
    
    def test_get_neighbors(self):
        """Test standalone neighbor function."""
        # Test with wrapping
        neighbors = get_neighbors(1, 1, 3, wrapping=True)
        self.assertEqual(len(neighbors), 8)
        
        # Test without wrapping
        neighbors = get_neighbors(0, 0, 3, wrapping=False)
        self.assertEqual(len(neighbors), 3)
        
        # Test edge case - center of larger grid
        neighbors = get_neighbors(2, 2, 5, wrapping=False)
        self.assertEqual(len(neighbors), 8)
    
    def test_calculate_grid_energy(self):
        """Test grid energy calculation."""
        # Perfect segregation (low energy)
        segregated_grid = np.array([
            [1, 1, 0],
            [2, 2, 0],
            [0, 0, 0]
        ])
        
        # Mixed configuration (high energy)
        mixed_grid = np.array([
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1]
        ])
        
        segregated_energy = calculate_grid_energy(segregated_grid, 3)
        mixed_energy = calculate_grid_energy(mixed_grid, 3)
        
        # Mixed grid should have higher energy
        self.assertGreater(mixed_energy, segregated_energy)
        self.assertGreaterEqual(segregated_energy, 0)
        self.assertGreaterEqual(mixed_energy, 0)


if __name__ == '__main__':
    # Set up test suite
    unittest.main(verbosity=2)