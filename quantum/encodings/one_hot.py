"""
One-hot encoding for Schelling model quantum optimization.

Uses 3 qubits per grid site to represent:
- |100⟩: empty
- |010⟩: red agent
- |001⟩: blue agent

This is the most intuitive encoding but requires the most qubits.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schelling.grid_utils import get_neighbor_pairs, grid_to_linear_index


class OneHotEncoder:
    """
    Quantum encoding using 3 qubits per site (one-hot encoding).
    
    This encoding uses dedicated qubits for each possible state,
    making it intuitive but qubit-expensive.
    """
    
    def __init__(
        self, 
        size: int, 
        colors: Dict[str, int], 
        wrapping: bool = True
    ):
        """
        Initialize the one-hot encoder.
        
        Args:
            size: Grid size (size x size)
            colors: Dictionary with agent counts {'red': n, 'blue': m, 'empty': k}
            wrapping: Whether to use periodic boundary conditions
        """
        self.size = size
        self.colors = colors
        self.wrapping = wrapping
        self.n_qubits = 3 * size * size  # 3 qubits per site
        self.n_sites = size * size
        
    def _get_qubit_indices(self, i: int, j: int) -> Tuple[int, int, int]:
        """
        Get the three qubit indices for grid position (i, j).
        
        Args:
            i, j: Grid coordinates
            
        Returns:
            Tuple of (empty_qubit, red_qubit, blue_qubit)
        """
        site_idx = grid_to_linear_index(i, j, self.size)
        return 3 * site_idx, 3 * site_idx + 1, 3 * site_idx + 2
    
    def build_validity_hamiltonian(self, validity_weight: float = 10.0) -> SparsePauliOp:
        """
        Build Hamiltonian to ensure exactly one qubit is active per site.
        
        H_valid = validity_weight * Σ_i (1 - q_empty - q_red - q_blue)²
        
        This penalizes invalid states like |000⟩, |110⟩, |111⟩, etc.
        
        Args:
            validity_weight: Strength of validity constraint penalty
            
        Returns:
            Validity Hamiltonian as SparsePauliOp
        """
        pauli_terms = []
        
        for i in range(self.size):
            for j in range(self.size):
                empty_q, red_q, blue_q = self._get_qubit_indices(i, j)
                
                # Build (1 - q_empty - q_red - q_blue)² constraint
                # This equals (1 - sum)² = 1 - 2*sum + sum²
                
                # Constant term: +1
                pauli_constant = ['I'] * self.n_qubits
                pauli_terms.append((''.join(pauli_constant), validity_weight))
                
                # Linear terms: -2 * q_i for each qubit
                for qubit_idx in [empty_q, red_q, blue_q]:
                    pauli_linear = ['I'] * self.n_qubits
                    # Convert to Pauli: q_i = (I - Z_i)/2, so -q_i = (Z_i - I)/2
                    pauli_linear[qubit_idx] = 'Z'
                    pauli_terms.append((''.join(pauli_linear), validity_weight))
                    
                    # Constant part of -2*q_i = -2*(I - Z_i)/2 = -(I - Z_i)
                    pauli_terms.append((''.join(pauli_constant), -validity_weight))
                
                # Quadratic terms: q_i * q_j for all pairs
                for q1 in [empty_q, red_q, blue_q]:
                    for q2 in [empty_q, red_q, blue_q]:
                        if q1 <= q2:  # Avoid double counting
                            pauli_quad = ['I'] * self.n_qubits
                            # q_i * q_j = (I - Z_i)(I - Z_j)/4
                            weight = validity_weight if q1 == q2 else 2 * validity_weight
                            
                            # (I - Z_i)(I - Z_j) = I - Z_i - Z_j + Z_i Z_j
                            # Constant term I: already included above
                            
                            # -Z_i term
                            pauli_quad[q1] = 'Z'
                            pauli_terms.append((''.join(pauli_quad), -weight / 4))
                            
                            if q1 != q2:
                                # -Z_j term
                                pauli_quad = ['I'] * self.n_qubits
                                pauli_quad[q2] = 'Z'
                                pauli_terms.append((''.join(pauli_quad), -weight / 4))
                            
                            # Z_i Z_j term
                            pauli_quad = ['I'] * self.n_qubits
                            pauli_quad[q1] = 'Z'
                            if q1 != q2:
                                pauli_quad[q2] = 'Z'
                            pauli_terms.append((''.join(pauli_quad), weight / 4))
        
        return SparsePauliOp.from_list(pauli_terms).simplify()
    
    def build_interaction_hamiltonian(self, interaction_weight: float = 1.0) -> SparsePauliOp:
        """
        Build Hamiltonian to penalize unlike neighboring agents.
        
        H_int = interaction_weight * Σ_{⟨i,j⟩} (q_red_i * q_blue_j + q_blue_i * q_red_j)
        
        Args:
            interaction_weight: Strength of interaction penalty
            
        Returns:
            Interaction Hamiltonian as SparsePauliOp
        """
        neighbor_pairs = get_neighbor_pairs(self.size, self.wrapping)
        pauli_terms = []
        
        for (i1, j1), (i2, j2) in neighbor_pairs:
            empty_q1, red_q1, blue_q1 = self._get_qubit_indices(i1, j1)
            empty_q2, red_q2, blue_q2 = self._get_qubit_indices(i2, j2)
            
            # Red-Blue interaction: q_red_i * q_blue_j
            # q_i * q_j = (I - Z_i)(I - Z_j)/4 = (I - Z_i - Z_j + Z_i Z_j)/4
            
            # Constant term
            pauli_const = ['I'] * self.n_qubits
            pauli_terms.append((''.join(pauli_const), interaction_weight / 4))
            
            # -Z_i term
            pauli_zi = ['I'] * self.n_qubits
            pauli_zi[red_q1] = 'Z'
            pauli_terms.append((''.join(pauli_zi), -interaction_weight / 4))
            
            # -Z_j term
            pauli_zj = ['I'] * self.n_qubits
            pauli_zj[blue_q2] = 'Z'
            pauli_terms.append((''.join(pauli_zj), -interaction_weight / 4))
            
            # Z_i Z_j term
            pauli_zizj = ['I'] * self.n_qubits
            pauli_zizj[red_q1] = 'Z'
            pauli_zizj[blue_q2] = 'Z'
            pauli_terms.append((''.join(pauli_zizj), interaction_weight / 4))
            
            # Blue-Red interaction: q_blue_i * q_red_j (symmetric)
            # Constant term (already added above)
            
            # -Z_i term (blue at site i)
            pauli_zi_blue = ['I'] * self.n_qubits
            pauli_zi_blue[blue_q1] = 'Z'
            pauli_terms.append((''.join(pauli_zi_blue), -interaction_weight / 4))
            
            # -Z_j term (red at site j)
            pauli_zj_red = ['I'] * self.n_qubits
            pauli_zj_red[red_q2] = 'Z'
            pauli_terms.append((''.join(pauli_zj_red), -interaction_weight / 4))
            
            # Z_i Z_j term
            pauli_zizj_blue_red = ['I'] * self.n_qubits
            pauli_zizj_blue_red[blue_q1] = 'Z'
            pauli_zizj_blue_red[red_q2] = 'Z'
            pauli_terms.append((''.join(pauli_zizj_blue_red), interaction_weight / 4))
        
        return SparsePauliOp.from_list(pauli_terms).simplify()
    
    def build_count_constraint_hamiltonian(self, count_weight: float = 5.0) -> SparsePauliOp:
        """
        Build Hamiltonian to enforce correct agent counts.
        
        H_count = count_weight * [(N_red - target_red)² + (N_blue - target_blue)²]
        where N_red = Σ_i q_red_i, N_blue = Σ_i q_blue_i
        
        Args:
            count_weight: Strength of count constraint penalty
            
        Returns:
            Count constraint Hamiltonian as SparsePauliOp
        """
        target_red = self.colors['red']
        target_blue = self.colors['blue']
        
        pauli_terms = []
        
        # Build N_red and N_blue operators
        red_qubits = []
        blue_qubits = []
        
        for i in range(self.size):
            for j in range(self.size):
                _, red_q, blue_q = self._get_qubit_indices(i, j)
                red_qubits.append(red_q)
                blue_qubits.append(blue_q)
        
        # For red agents: (N_red - target_red)²
        # N_red = Σ_i q_red_i = Σ_i (I - Z_i)/2
        
        # Constant terms: target_red²
        pauli_const = ['I'] * self.n_qubits
        pauli_terms.append((''.join(pauli_const), count_weight * target_red**2))
        
        # Linear terms: -2 * target_red * N_red
        for red_q in red_qubits:
            # -2 * target_red * (I - Z_i)/2 = -target_red * (I - Z_i)
            pauli_terms.append((''.join(pauli_const), -count_weight * target_red))
            
            pauli_z = ['I'] * self.n_qubits
            pauli_z[red_q] = 'Z'
            pauli_terms.append((''.join(pauli_z), count_weight * target_red))
        
        # Quadratic terms: N_red²
        for red_q1 in red_qubits:
            for red_q2 in red_qubits:
                if red_q1 <= red_q2:  # Avoid double counting
                    # q_i * q_j = (I - Z_i)(I - Z_j)/4
                    weight = count_weight if red_q1 == red_q2 else 2 * count_weight
                    
                    # Expand (I - Z_i)(I - Z_j) = I - Z_i - Z_j + Z_i Z_j
                    pauli_terms.append((''.join(pauli_const), weight / 4))
                    
                    pauli_zi = ['I'] * self.n_qubits
                    pauli_zi[red_q1] = 'Z'
                    pauli_terms.append((''.join(pauli_zi), -weight / 4))
                    
                    if red_q1 != red_q2:
                        pauli_zj = ['I'] * self.n_qubits
                        pauli_zj[red_q2] = 'Z'
                        pauli_terms.append((''.join(pauli_zj), -weight / 4))
                    
                    pauli_zizj = ['I'] * self.n_qubits
                    pauli_zizj[red_q1] = 'Z'
                    if red_q1 != red_q2:
                        pauli_zizj[red_q2] = 'Z'
                    pauli_terms.append((''.join(pauli_zizj), weight / 4))
        
        # Similar terms for blue agents
        pauli_terms.append((''.join(pauli_const), count_weight * target_blue**2))
        
        for blue_q in blue_qubits:
            pauli_terms.append((''.join(pauli_const), -count_weight * target_blue))
            
            pauli_z = ['I'] * self.n_qubits
            pauli_z[blue_q] = 'Z'
            pauli_terms.append((''.join(pauli_z), count_weight * target_blue))
        
        for blue_q1 in blue_qubits:
            for blue_q2 in blue_qubits:
                if blue_q1 <= blue_q2:
                    weight = count_weight if blue_q1 == blue_q2 else 2 * count_weight
                    
                    pauli_terms.append((''.join(pauli_const), weight / 4))
                    
                    pauli_zi = ['I'] * self.n_qubits
                    pauli_zi[blue_q1] = 'Z'
                    pauli_terms.append((''.join(pauli_zi), -weight / 4))
                    
                    if blue_q1 != blue_q2:
                        pauli_zj = ['I'] * self.n_qubits
                        pauli_zj[blue_q2] = 'Z'
                        pauli_terms.append((''.join(pauli_zj), -weight / 4))
                    
                    pauli_zizj = ['I'] * self.n_qubits
                    pauli_zizj[blue_q1] = 'Z'
                    if blue_q1 != blue_q2:
                        pauli_zizj[blue_q2] = 'Z'
                    pauli_terms.append((''.join(pauli_zizj), weight / 4))
        
        return SparsePauliOp.from_list(pauli_terms).simplify()
    
    def build_total_hamiltonian(
        self,
        validity_weight: float = 10.0,
        interaction_weight: float = 1.0,
        count_weight: float = 5.0
    ) -> SparsePauliOp:
        """
        Build the complete Hamiltonian for one-hot encoded Schelling model.
        
        H = H_validity + H_interaction + H_count
        
        Args:
            validity_weight: Weight for one-hot validity constraints
            interaction_weight: Weight for unlike neighbor penalty
            count_weight: Weight for agent count constraints
            
        Returns:
            Complete Hamiltonian as SparsePauliOp
        """
        print("Building one-hot validity Hamiltonian...")
        h_validity = self.build_validity_hamiltonian(validity_weight)
        
        print("Building one-hot interaction Hamiltonian...")
        h_interaction = self.build_interaction_hamiltonian(interaction_weight)
        
        print("Building one-hot count constraint Hamiltonian...")
        h_count = self.build_count_constraint_hamiltonian(count_weight)
        
        print("Combining Hamiltonians...")
        total_hamiltonian = h_validity + h_interaction + h_count
        
        return total_hamiltonian
    
    def decode_bitstring(self, bitstring: str) -> np.ndarray:
        """
        Convert quantum measurement bitstring back to grid representation.
        
        Args:
            bitstring: Binary string from quantum measurement (3 bits per site)
            
        Returns:
            2D numpy array representing the grid (0=empty, 1=red, 2=blue)
        """
        if len(bitstring) != self.n_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} doesn't match {self.n_qubits} qubits")
        
        grid = np.zeros((self.size, self.size), dtype=int)
        
        for i in range(self.size):
            for j in range(self.size):
                empty_q, red_q, blue_q = self._get_qubit_indices(i, j)
                
                # Extract the 3 bits for this site
                empty_bit = int(bitstring[empty_q])
                red_bit = int(bitstring[red_q])
                blue_bit = int(bitstring[blue_q])
                
                # Determine state based on one-hot encoding
                if empty_bit == 1 and red_bit == 0 and blue_bit == 0:
                    grid[i, j] = 0  # Empty
                elif empty_bit == 0 and red_bit == 1 and blue_bit == 0:
                    grid[i, j] = 1  # Red
                elif empty_bit == 0 and red_bit == 0 and blue_bit == 1:
                    grid[i, j] = 2  # Blue
                else:
                    # Invalid state - could be |000⟩, |110⟩, |111⟩, etc.
                    grid[i, j] = -1  # Mark as invalid
                    
        return grid


if __name__ == "__main__":
    # Test the one-hot encoding
    print("Testing one-hot encoding...")
    
    encoder = OneHotEncoder(
        size=2,  # Small test case
        colors={'red': 2, 'blue': 1, 'empty': 1},
        wrapping=False
    )
    
    print(f"Number of qubits: {encoder.n_qubits}")
    
    # Build Hamiltonian components
    h_validity = encoder.build_validity_hamiltonian()
    h_interaction = encoder.build_interaction_hamiltonian()
    
    print(f"Validity Hamiltonian terms: {len(h_validity.paulis)}")
    print(f"Interaction Hamiltonian terms: {len(h_interaction.paulis)}")
    
    # Test solution decoding
    test_bitstring = '100010001100'  # 4 sites: empty, red, red, blue
    if len(test_bitstring) == encoder.n_qubits:
        grid = encoder.decode_bitstring(test_bitstring)
        print(f"Test bitstring: {test_bitstring}")
        print(f"Decoded grid:\n{grid}")
    else:
        print(f"Test bitstring length {len(test_bitstring)} != {encoder.n_qubits} qubits")
    
    print("One-hot encoding test complete!")