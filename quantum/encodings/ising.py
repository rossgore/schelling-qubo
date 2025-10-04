"""
Ising spin encoding for Schelling model quantum optimization.

Uses 1 qubit per agent to represent:
- |0⟩ or spin-down (-1): red agent
- |1⟩ or spin-up (+1): blue agent

Empty spaces are handled through separate occupancy indicators
or by fixing certain qubits to specific values.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schelling.grid_utils import get_neighbor_pairs, grid_to_linear_index


class IsingEncoder:
    """
    Quantum encoding using 1 qubit per agent (Ising spin encoding).
    
    This is the most qubit-efficient encoding but limited to two agent types.
    Empty spaces require special handling through constraints or pre-assignment.
    """
    
    def __init__(
        self, 
        size: int, 
        colors: Dict[str, int], 
        wrapping: bool = True,
        empty_positions: List[Tuple[int, int]] = None
    ):
        """
        Initialize the Ising encoder.
        
        Args:
            size: Grid size (size x size)
            colors: Dictionary with agent counts {'red': n, 'blue': m, 'empty': k}
            wrapping: Whether to use periodic boundary conditions
            empty_positions: Fixed empty positions (if None, will be optimized)
        """
        self.size = size
        self.colors = colors
        self.wrapping = wrapping
        self.empty_positions = empty_positions or []
        
        # Only count occupied sites for qubits
        self.n_agents = colors['red'] + colors['blue']
        self.n_qubits = self.n_agents
        
        # Map grid positions to qubit indices (excluding empty spaces)
        self.position_to_qubit = {}
        self.qubit_to_position = {}
        
        self._create_position_mapping()
        
    def _create_position_mapping(self):
        """Create mapping between grid positions and qubit indices."""
        qubit_idx = 0
        
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) not in self.empty_positions:
                    self.position_to_qubit[(i, j)] = qubit_idx
                    self.qubit_to_position[qubit_idx] = (i, j)
                    qubit_idx += 1
                    
        assert qubit_idx == self.n_qubits, f"Qubit count mismatch: {qubit_idx} vs {self.n_qubits}"
    
    def build_interaction_hamiltonian(self, interaction_weight: float = 1.0) -> SparsePauliOp:
        """
        Build Hamiltonian to penalize unlike neighboring agents.
        
        For Ising spins: unlike neighbors contribute +1, like neighbors contribute -1.
        H_int = interaction_weight * Σ_{⟨i,j⟩} σz_i σz_j
        
        Args:
            interaction_weight: Strength of interaction penalty
            
        Returns:
            Interaction Hamiltonian as SparsePauliOp
        """
        neighbor_pairs = get_neighbor_pairs(self.size, self.wrapping)
        
        pauli_terms = []
        
        for (i1, j1), (i2, j2) in neighbor_pairs:
            # Skip if either position is empty
            if (i1, j1) in self.empty_positions or (i2, j2) in self.empty_positions:
                continue
                
            qubit1 = self.position_to_qubit[(i1, j1)]
            qubit2 = self.position_to_qubit[(i2, j2)]
            
            # Create Pauli Z ⊗ Z interaction
            pauli_string = ['I'] * self.n_qubits
            pauli_string[qubit1] = 'Z'
            pauli_string[qubit2] = 'Z'
            
            pauli_terms.append((''.join(pauli_string), interaction_weight))
        
        if not pauli_terms:
            # Return zero Hamiltonian if no interactions
            pauli_terms = [('I' * self.n_qubits, 0.0)]
            
        return SparsePauliOp.from_list(pauli_terms)
    
    def build_count_constraint_hamiltonian(self, count_weight: float = 5.0) -> SparsePauliOp:
        """
        Build Hamiltonian to enforce correct agent counts.
        
        For Ising encoding:
        N_red = (n_qubits - Σ_i σz_i) / 2  (number of -1 spins)
        N_blue = (n_qubits + Σ_i σz_i) / 2  (number of +1 spins)
        
        Args:
            count_weight: Strength of count constraint penalty
            
        Returns:
            Count constraint Hamiltonian as SparsePauliOp
        """
        target_red = self.colors['red']
        target_blue = self.colors['blue']
        
        # Build total magnetization operator: M = Σ_i σz_i
        magnetization_terms = []
        for qubit_idx in range(self.n_qubits):
            pauli_string = ['I'] * self.n_qubits
            pauli_string[qubit_idx] = 'Z'
            magnetization_terms.append((''.join(pauli_string), 1.0))
        
        # Add constant term
        magnetization_terms.append(('I' * self.n_qubits, 0.0))
        magnetization_op = SparsePauliOp.from_list(magnetization_terms)
        
        # Constraint: (N_blue - target_blue)² = ((n_qubits + M)/2 - target_blue)²
        # Simplify: (M - (2*target_blue - n_qubits))² / 4
        target_magnetization = 2 * target_blue - self.n_qubits
        
        # Build (M - target_magnetization)²
        identity = SparsePauliOp.from_list([('I' * self.n_qubits, 1.0)])
        
        constraint = (magnetization_op - target_magnetization * identity) ** 2
        
        return count_weight * constraint / 4.0
    
    def build_total_hamiltonian(
        self,
        interaction_weight: float = 1.0,
        count_weight: float = 5.0
    ) -> SparsePauliOp:
        """
        Build the complete Hamiltonian for Ising-encoded Schelling model.
        
        H = H_interaction + H_count
        
        Args:
            interaction_weight: Weight for unlike neighbor penalty
            count_weight: Weight for agent count constraints
            
        Returns:
            Complete Hamiltonian as SparsePauliOp
        """
        print("Building Ising interaction Hamiltonian...")
        h_interaction = self.build_interaction_hamiltonian(interaction_weight)
        
        print("Building Ising count constraint Hamiltonian...")
        h_count = self.build_count_constraint_hamiltonian(count_weight)
        
        print("Combining Hamiltonians...")
        total_hamiltonian = h_interaction + h_count
        
        return total_hamiltonian
    
    def decode_bitstring(self, bitstring: str) -> np.ndarray:
        """
        Convert quantum measurement bitstring back to grid representation.
        
        Args:
            bitstring: Binary string from quantum measurement
            
        Returns:
            2D numpy array representing the grid (0=empty, 1=red, 2=blue)
        """
        if len(bitstring) != self.n_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} doesn't match {self.n_qubits} qubits")
        
        grid = np.zeros((self.size, self.size), dtype=int)
        
        # Set empty positions
        for pos in self.empty_positions:
            grid[pos[0], pos[1]] = 0
        
        # Set agent positions based on spin values
        for qubit_idx, bit in enumerate(bitstring):
            pos = self.qubit_to_position[qubit_idx]
            if bit == '0':  # Spin down = red
                grid[pos[0], pos[1]] = 1
            else:  # Spin up = blue
                grid[pos[0], pos[1]] = 2
                
        return grid
    
    def validate_solution(self, bitstring: str) -> Dict[str, bool]:
        """
        Validate that a quantum solution satisfies constraints.
        
        Args:
            bitstring: Binary string solution
            
        Returns:
            Dictionary with validation results
        """
        grid = self.decode_bitstring(bitstring)
        
        # Count agents in solution
        red_count = np.sum(grid == 1)
        blue_count = np.sum(grid == 2)
        empty_count = np.sum(grid == 0)
        
        validation = {
            'valid_red_count': red_count == self.colors['red'],
            'valid_blue_count': blue_count == self.colors['blue'], 
            'valid_empty_count': empty_count == self.colors['empty'],
            'counts': {'red': red_count, 'blue': blue_count, 'empty': empty_count}
        }
        
        validation['overall_valid'] = all([
            validation['valid_red_count'],
            validation['valid_blue_count'],
            validation['valid_empty_count']
        ])
        
        return validation


if __name__ == "__main__":
    # Test the Ising encoding
    print("Testing Ising encoding...")
    
    # Simple test case with fixed empty positions
    empty_pos = [(0, 0), (2, 2)]  # Fix two corners as empty
    encoder = IsingEncoder(
        size=3,
        colors={'red': 3, 'blue': 4, 'empty': 2},
        wrapping=False,
        empty_positions=empty_pos
    )
    
    print(f"Number of qubits: {encoder.n_qubits}")
    print(f"Position to qubit mapping: {encoder.position_to_qubit}")
    
    # Build Hamiltonian
    hamiltonian = encoder.build_total_hamiltonian()
    print(f"Total Hamiltonian terms: {len(hamiltonian.paulis)}")
    
    # Test solution validation
    test_bitstring = '0011011'  # 3 reds (0s) and 4 blues (1s)
    if len(test_bitstring) == encoder.n_qubits:
        validation = encoder.validate_solution(test_bitstring)
        print(f"Test solution validation: {validation}")
        
        grid = encoder.decode_bitstring(test_bitstring)
        print(f"Decoded grid:\n{grid}")
    else:
        print(f"Test bitstring length {len(test_bitstring)} != {encoder.n_qubits} qubits")
    
    print("Ising encoding test complete!")