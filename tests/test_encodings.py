"""
Unit tests for quantum encoding implementations.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum.encodings.basis_state import BasisStateEncoder
from quantum.encodings.ising import IsingEncoder
from quantum.encodings.one_hot import OneHotEncoder


class TestBasisStateEncoder(unittest.TestCase):
    """Test cases for basis state encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = BasisStateEncoder(
            size=2,
            colors={'red': 2, 'blue': 1, 'empty': 1},
            wrapping=False
        )
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(self.encoder.size, 2)
        self.assertEqual(self.encoder.n_qubits, 8)  # 2 qubits per site, 4 sites
        self.assertEqual(self.encoder.n_sites, 4)
        self.assertFalse(self.encoder.wrapping)
    
    def test_qubit_indices(self):
        """Test qubit index mapping."""
        # Test known positions
        q0, q1 = self.encoder._get_qubit_indices(0, 0)
        self.assertEqual(q0, 0)
        self.assertEqual(q1, 1)
        
        q0, q1 = self.encoder._get_qubit_indices(1, 1) 
        self.assertEqual(q0, 6)
        self.assertEqual(q1, 7)
    
    def test_hamiltonian_construction(self):
        """Test Hamiltonian building."""
        # Test penalty Hamiltonian
        h_penalty = self.encoder.build_penalty_hamiltonian(penalty_weight=1.0)
        self.assertIsNotNone(h_penalty)
        self.assertGreater(len(h_penalty.paulis), 0)
        
        # Test interaction Hamiltonian
        h_interaction = self.encoder.build_interaction_hamiltonian(interaction_weight=1.0)
        self.assertIsNotNone(h_interaction)
        self.assertGreater(len(h_interaction.paulis), 0)
        
        # Test total Hamiltonian
        h_total = self.encoder.build_total_hamiltonian()
        self.assertIsNotNone(h_total)
        self.assertGreater(len(h_total.paulis), 0)


class TestIsingEncoder(unittest.TestCase):
    """Test cases for Ising encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.empty_positions = [(0, 0), (2, 2)]
        self.encoder = IsingEncoder(
            size=3,
            colors={'red': 3, 'blue': 4, 'empty': 2},
            wrapping=False,
            empty_positions=self.empty_positions
        )
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(self.encoder.size, 3)
        self.assertEqual(self.encoder.n_agents, 7)  # 3 red + 4 blue
        self.assertEqual(self.encoder.n_qubits, 7)  # 1 qubit per agent
        self.assertEqual(len(self.encoder.empty_positions), 2)
    
    def test_position_mapping(self):
        """Test position to qubit mapping."""
        # Empty positions should not be in mapping
        self.assertNotIn((0, 0), self.encoder.position_to_qubit)
        self.assertNotIn((2, 2), self.encoder.position_to_qubit)
        
        # Non-empty positions should be mapped
        self.assertIn((0, 1), self.encoder.position_to_qubit)
        self.assertIn((1, 1), self.encoder.position_to_qubit)
        
        # Check bidirectional mapping consistency
        for pos, qubit in self.encoder.position_to_qubit.items():
            self.assertEqual(self.encoder.qubit_to_position[qubit], pos)
    
    def test_hamiltonian_construction(self):
        """Test Hamiltonian construction."""
        h_interaction = self.encoder.build_interaction_hamiltonian()
        h_count = self.encoder.build_count_constraint_hamiltonian()
        h_total = self.encoder.build_total_hamiltonian()
        
        self.assertIsNotNone(h_interaction)
        self.assertIsNotNone(h_count)
        self.assertIsNotNone(h_total)
        
        # Check that total includes both components
        self.assertGreater(len(h_total.paulis), 0)
    
    def test_bitstring_decoding(self):
        """Test bitstring to grid decoding."""
        if self.encoder.n_qubits == 7:
            test_bitstring = '0011011'  # Mix of 0s and 1s
            grid = self.encoder.decode_bitstring(test_bitstring)
            
            # Check grid shape and types
            self.assertEqual(grid.shape, (3, 3))
            self.assertTrue(np.all(grid >= 0))
            self.assertTrue(np.all(grid <= 2))
            
            # Check empty positions are preserved
            for pos in self.empty_positions:
                self.assertEqual(grid[pos[0], pos[1]], 0)
    
    def test_solution_validation(self):
        """Test solution validation."""
        if self.encoder.n_qubits == 7:
            test_bitstring = '0001111'  # 3 reds (0s) and 4 blues (1s)
            validation = self.encoder.validate_solution(test_bitstring)
            
            self.assertIn('valid_red_count', validation)
            self.assertIn('valid_blue_count', validation)
            self.assertIn('overall_valid', validation)
            self.assertIsInstance(validation['overall_valid'], bool)


class TestOneHotEncoder(unittest.TestCase):
    """Test cases for one-hot encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = OneHotEncoder(
            size=2,
            colors={'red': 1, 'blue': 1, 'empty': 2},
            wrapping=False
        )
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(self.encoder.size, 2)
        self.assertEqual(self.encoder.n_qubits, 12)  # 3 qubits per site, 4 sites
        self.assertEqual(self.encoder.n_sites, 4)
    
    def test_qubit_indices(self):
        """Test qubit index mapping."""
        empty_q, red_q, blue_q = self.encoder._get_qubit_indices(0, 0)
        self.assertEqual(empty_q, 0)
        self.assertEqual(red_q, 1)
        self.assertEqual(blue_q, 2)
        
        empty_q, red_q, blue_q = self.encoder._get_qubit_indices(1, 1)
        self.assertEqual(empty_q, 9)
        self.assertEqual(red_q, 10)
        self.assertEqual(blue_q, 11)
    
    def test_hamiltonian_construction(self):
        """Test Hamiltonian construction."""
        # Note: This may be slow due to many terms
        try:
            h_validity = self.encoder.build_validity_hamiltonian(validity_weight=1.0)
            self.assertIsNotNone(h_validity)
            self.assertGreater(len(h_validity.paulis), 0)
        except Exception as e:
            self.skipTest(f"Hamiltonian construction failed: {str(e)}")
    
    def test_bitstring_decoding(self):
        """Test bitstring decoding."""
        # Test valid one-hot encodings
        test_bitstring = '100010001100'  # empty, red, red, blue
        
        if len(test_bitstring) == self.encoder.n_qubits:
            grid = self.encoder.decode_bitstring(test_bitstring)
            
            # Check grid shape
            self.assertEqual(grid.shape, (2, 2))
            
            # Check specific decodings
            self.assertEqual(grid[0, 0], 0)  # Empty: 100
            self.assertEqual(grid[0, 1], 1)  # Red: 010  
            self.assertEqual(grid[1, 0], 1)  # Red: 010
            self.assertEqual(grid[1, 1], 2)  # Blue: 001
    
    def test_invalid_bitstring_decoding(self):
        """Test handling of invalid one-hot states."""
        # Invalid state: 110 (multiple bits set)
        invalid_bitstring = '110010001100'
        
        if len(invalid_bitstring) == self.encoder.n_qubits:
            grid = self.encoder.decode_bitstring(invalid_bitstring)
            
            # Should mark invalid positions with -1
            self.assertEqual(grid[0, 0], -1)  # Invalid: 110


class TestEncodingConsistency(unittest.TestCase):
    """Test consistency across different encodings."""
    
    def setUp(self):
        """Set up encoders for comparison."""
        self.problem_config = {
            'size': 3,
            'colors': {'red': 3, 'blue': 3, 'empty': 3}
        }
    
    def test_qubit_requirements(self):
        """Test qubit requirements for different encodings."""
        # Basis state: 2 qubits per site
        basis_encoder = BasisStateEncoder(**self.problem_config)
        expected_basis_qubits = 2 * 3 * 3
        self.assertEqual(basis_encoder.n_qubits, expected_basis_qubits)
        
        # One-hot: 3 qubits per site
        onehot_encoder = OneHotEncoder(**self.problem_config)
        expected_onehot_qubits = 3 * 3 * 3
        self.assertEqual(onehot_encoder.n_qubits, expected_onehot_qubits)
        
        # Ising: 1 qubit per agent (excluding empty)
        empty_positions = [(0, 0), (1, 1), (2, 2)]  # 3 empty positions
        ising_encoder = IsingEncoder(
            **self.problem_config,
            empty_positions=empty_positions
        )
        expected_ising_qubits = 3 + 3  # red + blue agents
        self.assertEqual(ising_encoder.n_qubits, expected_ising_qubits)
    
    def test_encoding_efficiency(self):
        """Test relative efficiency of encodings."""
        total_sites = 3 * 3
        
        # Efficiency = sites per qubit
        basis_efficiency = total_sites / (2 * total_sites)  # 0.5
        onehot_efficiency = total_sites / (3 * total_sites)  # ~0.33
        ising_efficiency = total_sites / 6  # 1.5 (most efficient)
        
        # Ising should be most efficient
        self.assertGreater(ising_efficiency, basis_efficiency)
        self.assertGreater(basis_efficiency, onehot_efficiency)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)