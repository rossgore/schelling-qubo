"""
Classical Schelling Segregation Model Implementation
"""
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter


class SchellingModel:
    """
    Classical implementation of the Schelling segregation model.
    
    The model simulates how agents of different types tend to cluster together
    based on their satisfaction with their neighborhood composition.
    """
    
    def __init__(
        self, 
        size: int, 
        colors: Dict[str, int], 
        similarity_threshold: float = 0.5,
        wrapping: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Schelling model.
        
        Args:
            size: Grid size (size x size)
            colors: Dictionary with counts for each agent type
                   e.g., {'red': 3, 'blue': 3, 'empty': 3}
            similarity_threshold: Minimum fraction of similar neighbors for satisfaction
            wrapping: Whether grid edges wrap around (torus topology)
            random_seed: Random seed for reproducibility
        """
        self.size = size
        self.colors = colors
        self.similarity_threshold = similarity_threshold
        self.wrapping = wrapping
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        # Initialize grid with random placement
        self.grid = self._setup_grid()
        self.history = [self.grid.copy()]
        
    def _setup_grid(self) -> np.ndarray:
        """
        Initialize a random grid with the specified color distribution.
        
        Returns:
            2D numpy array representing the grid
            0 = empty, 1 = red, 2 = blue
        """
        grid = np.zeros((self.size, self.size), dtype=int)
        total_agents = self.colors['red'] + self.colors['blue']
        
        # Get all possible positions and shuffle them
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(positions)
        
        # Place red agents (type 1)
        for i in range(self.colors['red']):
            pos = positions[i]
            grid[pos[0], pos[1]] = 1
            
        # Place blue agents (type 2)
        for i in range(self.colors['red'], total_agents):
            pos = positions[i]
            grid[pos[0], pos[1]] = 2
            
        return grid
    
    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """
        Get all neighbors for position (i, j).
        
        Args:
            i, j: Grid coordinates
            
        Returns:
            List of neighbor coordinates
        """
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:  # Skip the center position
                    continue
                    
                ni, nj = i + di, j + dj
                
                if self.wrapping:
                    # Wrap around edges (torus topology)
                    ni = ni % self.size
                    nj = nj % self.size
                    neighbors.append((ni, nj))
                elif 0 <= ni < self.size and 0 <= nj < self.size:
                    # Standard boundary conditions
                    neighbors.append((ni, nj))
                    
        return neighbors
    
    def calculate_satisfaction(self) -> Dict[Tuple[int, int], bool]:
        """
        Calculate satisfaction for each agent based on neighborhood composition.
        
        An agent is satisfied if the fraction of same-type neighbors
        meets or exceeds the similarity threshold.
        
        Returns:
            Dictionary mapping agent positions to satisfaction status
        """
        satisfied = {}
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:  # Skip empty spaces
                    continue
                    
                neighbors = self.get_neighbors(i, j)
                same_neighbors = 0
                total_neighbors = 0
                
                # Count neighbors by type
                for ni, nj in neighbors:
                    if self.grid[ni, nj] != 0:  # Not empty
                        total_neighbors += 1
                        if self.grid[ni, nj] == self.grid[i, j]:  # Same type
                            same_neighbors += 1
                
                # Determine satisfaction
                if total_neighbors == 0:
                    # No neighbors means satisfied by default
                    satisfied[(i, j)] = True
                else:
                    similarity = same_neighbors / total_neighbors
                    satisfied[(i, j)] = similarity >= self.similarity_threshold
                    
        return satisfied
    
    def run_step(self) -> int:
        """
        Run one step of the Schelling model simulation.
        
        Unsatisfied agents move to random empty spaces.
        
        Returns:
            Number of agents that moved
        """
        satisfied = self.calculate_satisfaction()
        
        # Identify unsatisfied agents and empty spaces
        unsatisfied = [(i, j) for (i, j), sat in satisfied.items() if not sat]
        empty_spaces = [
            (i, j) for i in range(self.size) for j in range(self.size) 
            if self.grid[i, j] == 0
        ]
        
        moves = 0
        random.shuffle(unsatisfied)
        random.shuffle(empty_spaces)
        
        # Move unsatisfied agents to empty spaces
        for agent_pos in unsatisfied:
            if not empty_spaces:
                break
                
            # Select new position and update lists
            new_pos = empty_spaces.pop(0)
            empty_spaces.append(agent_pos)
            
            # Perform the move
            agent_type = self.grid[agent_pos[0], agent_pos[1]]
            self.grid[new_pos[0], new_pos[1]] = agent_type
            self.grid[agent_pos[0], agent_pos[1]] = 0
            moves += 1
            
        return moves
    
    def simulate(self, max_steps: int = 100, verbose: bool = True) -> Dict:
        """
        Run the complete Schelling simulation.
        
        Args:
            max_steps: Maximum number of simulation steps
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with simulation results and statistics
        """
        if verbose:
            print(f"Starting Schelling simulation (threshold={self.similarity_threshold})")
            print(f"Initial grid:\n{self.grid}")
            
        steps_taken = 0
        total_moves = 0
        
        for step in range(max_steps):
            moves = self.run_step()
            total_moves += moves
            steps_taken += 1
            
            # Store grid state
            self.history.append(self.grid.copy())
            
            if verbose:
                print(f"Step {step + 1}: {moves} agents moved")
                
            if moves == 0:
                if verbose:
                    print("Simulation converged - no more moves!")
                break
                
        # Calculate final satisfaction
        final_satisfaction = self.calculate_satisfaction()
        satisfaction_rate = sum(final_satisfaction.values()) / len(final_satisfaction)
        
        results = {
            'converged': moves == 0,
            'steps_taken': steps_taken,
            'total_moves': total_moves,
            'final_satisfaction_rate': satisfaction_rate,
            'final_grid': self.grid.copy(),
            'grid_history': self.history
        }
        
        if verbose:
            print(f"Final satisfaction rate: {satisfaction_rate:.2%}")
            print(f"Final grid:\n{self.grid}")
            
        return results
    
    def get_segregation_index(self) -> float:
        """
        Calculate a simple segregation index based on neighbor similarity.
        
        Returns:
            Segregation index between 0 (fully mixed) and 1 (fully segregated)
        """
        total_pairs = 0
        same_type_pairs = 0
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:  # Skip empty spaces
                    continue
                    
                neighbors = self.get_neighbors(i, j)
                for ni, nj in neighbors:
                    if self.grid[ni, nj] != 0:  # Not empty
                        total_pairs += 1
                        if self.grid[ni, nj] == self.grid[i, j]:  # Same type
                            same_type_pairs += 1
        
        if total_pairs == 0:
            return 0.0
            
        return same_type_pairs / total_pairs
    
    def get_bit_string_representation(self) -> str:
        """
        Convert current grid to bit string for quantum comparison.
        
        Uses basis state encoding: 00=empty, 01=red, 10=blue
        
        Returns:
            Binary string representation of the grid
        """
        bit_string = ""
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:  # Empty
                    bit_string += "00"
                elif self.grid[i, j] == 1:  # Red
                    bit_string += "01"
                elif self.grid[i, j] == 2:  # Blue
                    bit_string += "10"
                    
        return bit_string


if __name__ == "__main__":
    # Example usage
    model = SchellingModel(
        size=3,
        colors={'red': 3, 'blue': 3, 'empty': 3},
        similarity_threshold=0.5,
        random_seed=42
    )
    
    results = model.simulate(max_steps=10)
    print(f"Segregation index: {model.get_segregation_index():.3f}")