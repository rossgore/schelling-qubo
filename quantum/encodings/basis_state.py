"""
Basis state encoding for Schelling model quantum optimization.

Uses 2 qubits per grid site to represent:
- |00⟩: empty
- |01⟩: red agent  
- |10⟩: blue agent
- |11⟩: invalid state (penalized)
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schelling.grid_utils import get_neighbor_pairs, grid_to_linear_index


class BasisStateEncoder:
    """
    Quantum encoding using 2 qubits per site (basis state encoding).
    
    This is generally the most practical encoding, balancing qubit efficiency
    with representation flexibility.
    """
    
    def __init__(
        self, 
        size: int, 
        colors: Dict[str, int], 
        wrapping: bool = True
    ):
        """
        Initialize the basis state encoder.
        
        Args:
            size: Grid size (size x size)
            colors: Dictionary with agent counts {'red': n, 'blue': m, 'empty': k}
            wrapping: Whether to use periodic boundary conditions
        """
        self.size = size
        self.colors = colors
        self.wrapping = wrapping
        self.n_qubits = 2 * size * size  # 2 qubits per site
        self.n_sites = size * size
        
    def _get_qubit_indices(self, i: int, j: int) -> Tuple[int, int]:
        """
        Get the two qubit indices for grid position (i, j).
        
        Args:
            i, j: Grid coordinates
            
        Returns:
            Tuple of (first_qubit_index, second_qubit_index)
        """
        site_idx = grid_to_linear_index(i, j, self.size)
        return 2 * site_idx, 2 * site_idx + 1
    
    def _create_projector_operators(self) -> Dict[str, List[SparsePauliOp]]:
        """
        Create Pauli operators for state projections.
        
        Returns:
            Dictionary containing projection operators for each state
        """
        projectors = {'empty': [], 'red': [], 'blue': [], 'invalid': []}
        
        for i in range(self.size):
            for j in range(self.size):
                q0, q1 = self._get_qubit_indices(i, j)
                
                # Create Pauli strings for this site
                pauli_empty = ['I'] * self.n_qubits    # |00⟩ = (I+Z)(I+Z)/4
                pauli_red = ['I'] * self.n_qubits      # |01⟩ = (I+Z)(I-Z)/4  
                pauli_blue = ['I'] * self.n_qubits     # |10⟩ = (I-Z)(I+Z)/4
                pauli_invalid = ['I'] * self.n_qubits  # |11⟩ = (I-Z)(I-Z)/4
                
                # |00⟩ state projector: (I+Z) ⊗ (I+Z) / 4
                projectors['empty'].append([
                    SparsePauliOp.from_list([(''.join(pauli_empty), 0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_empty[:q0] + ['Z'] + pauli_empty[q0+1:]), 0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_empty[:q1] + ['Z'] + pauli_empty[q1+1:]), 0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_empty[:q0] + ['Z'] + pauli_empty[q0+1:q1] + ['Z'] + pauli_empty[q1+1:]), 0.25)])
                ])
                
                # |01⟩ state projector: (I+Z) ⊗ (I-Z) / 4
                projectors['red'].append([
                    SparsePauliOp.from_list([(''.join(pauli_red), 0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_red[:q0] + ['Z'] + pauli_red[q0+1:]), 0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_red[:q1] + ['Z'] + pauli_red[q1+1:]), -0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_red[:q0] + ['Z'] + pauli_red[q0+1:q1] + ['Z'] + pauli_red[q1+1:]), -0.25)])
                ])
                
                # |10⟩ state projector: (I-Z) ⊗ (I+Z) / 4
                projectors['blue'].append([
                    SparsePauliOp.from_list([(''.join(pauli_blue), 0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_blue[:q0] + ['Z'] + pauli_blue[q0+1:]), -0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_blue[:q1] + ['Z'] + pauli_blue[q1+1:]), 0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_blue[:q0] + ['Z'] + pauli_blue[q0+1:q1] + ['Z'] + pauli_blue[q1+1:]), -0.25)])
                ])
                
                # |11⟩ state projector: (I-Z) ⊗ (I-Z) / 4
                projectors['invalid'].append([
                    SparsePauliOp.from_list([(''.join(pauli_invalid), 0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_invalid[:q0] + ['Z'] + pauli_invalid[q0+1:]), -0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_invalid[:q1] + ['Z'] + pauli_invalid[q1+1:]), -0.25)]),
                    SparsePauliOp.from_list([(''.join(pauli_invalid[:q0] + ['Z'] + pauli_invalid[q0+1:q1] + ['Z'] + pauli_invalid[q1+1:]), 0.25)])
                ])
        
        return projectors
    
    def build_penalty_hamiltonian(self, penalty_weight: float = 10.0) -> SparsePauliOp:
        """
        Build Hamiltonian to penalize invalid |11⟩ states.
        
        H_penalty = penalty_weight * Σ_i |11⟩⟨11|_i
        
        Args:
            penalty_weight: Strength of penalty for invalid states
            
        Returns:
            Penalty Hamiltonian as SparsePauliOp
        """
        projectors = self._create_projector_operators()
        penalty_hamiltonian = None
        
        for site_idx in range(self.n_sites):
            # Sum all terms for this site's invalid state projector
            site_penalty = None
            for term in projectors['invalid'][site_idx]:
                if site_penalty is None:
                    site_penalty = penalty_weight * term
                else:
                    site_penalty += penalty_weight * term
            
            # Add to total penalty Hamiltonian
            if penalty_hamiltonian is None:
                penalty_hamiltonian = site_penalty
            else:
                penalty_hamiltonian += site_penalty
                
        return penalty_hamiltonian
    
    def build_interaction_hamiltonian(self, interaction_weight: float = 1.0) -> SparsePauliOp:
        """
        Build Hamiltonian to penalize unlike neighboring agents.
        
        H_int = interaction_weight * Σ_{⟨i,j⟩} (|red⟩_i|blue⟩_j + |blue⟩_i|red⟩_j)
        
        Args:
            interaction_weight: Strength of interaction penalty
            
        Returns:
            Interaction Hamiltonian as SparsePauliOp
        """
        projectors = self._create_projector_operators()
        neighbor_pairs = get_neighbor_pairs(self.size, self.wrapping)
        interaction_hamiltonian = None
        
        for (i1, j1), (i2, j2) in neighbor_pairs:
            site1_idx = grid_to_linear_index(i1, j1, self.size)
            site2_idx = grid_to_linear_index(i2, j2, self.size)
            
            # red_i * blue_j interaction
            red_blue_term = None
            for r_term in projectors['red'][site1_idx]:
                for b_term in projectors['blue'][site2_idx]:
                    term = interaction_weight * r_term * b_term
                    if red_blue_term is None:
                        red_blue_term = term
                    else:
                        red_blue_term += term
            
            # blue_i * red_j interaction
            blue_red_term = None
            for b_term in projectors['blue'][site1_idx]:
                for r_term in projectors['red'][site2_idx]:
                    term = interaction_weight * b_term * r_term
                    if blue_red_term is None:
                        blue_red_term = term
                    else:
                        blue_red_term += term
            
            # Add both interaction terms to total Hamiltonian
            pair_interaction = red_blue_term + blue_red_term
            if interaction_hamiltonian is None:
                interaction_hamiltonian = pair_interaction
            else:
                interaction_hamiltonian += pair_interaction
                
        return interaction_hamiltonian
    
    def build_count_constraint_hamiltonian(self, count_weight: float = 5.0) -> SparsePauliOp:
        """
        Build Hamiltonian to enforce correct agent counts.
        
        H_count = count_weight * [(N_red - target_red)² + (N_blue - target_blue)²]
        
        Args:
            count_weight: Strength of count constraint penalty
            
        Returns:
            Count constraint Hamiltonian as SparsePauliOp
        """
        projectors = self._create_projector_operators()
        
        # Build total count operators
        red_count_op = None
        blue_count_op = None
        
        for site_idx in range(self.n_sites):
            # Sum red projectors
            site_red_op = None
            for term in projectors['red'][site_idx]:
                if site_red_op is None:
                    site_red_op = term
                else:
                    site_red_op += term
                    
            if red_count_op is None:
                red_count_op = site_red_op
            else:
                red_count_op += site_red_op
            
            # Sum blue projectors
            site_blue_op = None
            for term in projectors['blue'][site_idx]:
                if site_blue_op is None:
                    site_blue_op = term
                else:
                    site_blue_op += term
                    
            if blue_count_op is None:
                blue_count_op = site_blue_op
            else:
                blue_count_op += site_blue_op
        
        # Build constraint terms: (N - target)²
        target_red = self.colors['red']
        target_blue = self.colors['blue']
        
        # Create identity operator for constant terms
        identity = SparsePauliOp.from_list([('I' * self.n_qubits, 1.0)])
        
        # (N_red - target_red)² = N_red² - 2*target_red*N_red + target_red²
        red_constraint = (red_count_op * red_count_op - 
                         2 * target_red * red_count_op + 
                         target_red**2 * identity)
        
        # (N_blue - target_blue)² = N_blue² - 2*target_blue*N_blue + target_blue²
        blue_constraint = (blue_count_op * blue_count_op - 
                          2 * target_blue * blue_count_op + 
                          target_blue**2 * identity)
        
        count_hamiltonian = count_weight * (red_constraint + blue_constraint)
        
        return count_hamiltonian
    
    def build_total_hamiltonian(
        self,
        penalty_weight: float = 10.0,
        interaction_weight: float = 1.0,
        count_weight: float = 5.0
    ) -> SparsePauliOp:
        """
        Build the complete Hamiltonian for Schelling model optimization.
        
        H = H_penalty + H_interaction + H_count
        
        Args:
            penalty_weight: Weight for invalid state penalty
            interaction_weight: Weight for unlike neighbor penalty  
            count_weight: Weight for agent count constraints
            
        Returns:
            Complete Hamiltonian as SparsePauliOp
        """
        print("Building penalty Hamiltonian...")
        h_penalty = self.build_penalty_hamiltonian(penalty_weight)
        
        print("Building interaction Hamiltonian...")
        h_interaction = self.build_interaction_hamiltonian(interaction_weight)
        
        print("Building count constraint Hamiltonian...")
        h_count = self.build_count_constraint_hamiltonian(count_weight)
        
        print("Combining Hamiltonians...")
        total_hamiltonian = h_penalty + h_interaction + h_count
        
        return total_hamiltonian


if __name__ == "__main__":
    # Test the encoding
    encoder = BasisStateEncoder(
        size=2,  # Start with small test case
        colors={'red': 2, 'blue': 2, 'empty': 0},
        wrapping=False
    )
    
    print(f"Number of qubits: {encoder.n_qubits}")
    
    # Build and test Hamiltonian components
    print("Testing Hamiltonian construction...")
    h_penalty = encoder.build_penalty_hamiltonian()
    h_interaction = encoder.build_interaction_hamiltonian()
    
    print(f"Penalty Hamiltonian terms: {len(h_penalty.paulis)}")
    print(f"Interaction Hamiltonian terms: {len(h_interaction.paulis)}")
    
    # Build complete Hamiltonian
    hamiltonian = encoder.build_total_hamiltonian()
    print(f"Total Hamiltonian terms: {len(hamiltonian.paulis)}")