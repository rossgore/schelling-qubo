"""
Grid utilities and neighbor calculation functions
"""
import numpy as np
from typing import List, Tuple


def get_neighbors(i: int, j: int, size: int, wrapping: bool = True) -> List[Tuple[int, int]]:
    """
    Get all neighbors for position (i, j) in a grid.
    
    Args:
        i, j: Grid coordinates
        size: Grid size (assumes square grid)
        wrapping: Whether edges wrap around (torus topology)
        
    Returns:
        List of neighbor coordinates
    """
    neighbors = []
    
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:  # Skip center position
                continue
                
            ni, nj = i + di, j + dj
            
            if wrapping:
                # Wrap around edges (torus topology)
                ni = ni % size
                nj = nj % size
                neighbors.append((ni, nj))
            elif 0 <= ni < size and 0 <= nj < size:
                # Standard boundary conditions
                neighbors.append((ni, nj))
                
    return neighbors


def get_neighbor_pairs(size: int, wrapping: bool = True) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Get all neighboring pairs in the grid for Hamiltonian construction.
    
    Args:
        size: Grid size (assumes square grid)
        wrapping: Whether edges wrap around
        
    Returns:
        List of neighbor pairs as ((i1, j1), (i2, j2))
    """
    pairs = []
    
    for i in range(size):
        for j in range(size):
            neighbors = get_neighbors(i, j, size, wrapping)
            for ni, nj in neighbors:
                # Avoid duplicate pairs by ordering
                if (i, j) < (ni, nj):
                    pairs.append(((i, j), (ni, nj)))
                    
    return pairs


def grid_to_linear_index(i: int, j: int, size: int) -> int:
    """
    Convert 2D grid coordinates to linear index.
    
    Args:
        i, j: Grid coordinates
        size: Grid size
        
    Returns:
        Linear index
    """
    return i * size + j


def linear_to_grid_index(idx: int, size: int) -> Tuple[int, int]:
    """
    Convert linear index back to 2D grid coordinates.
    
    Args:
        idx: Linear index
        size: Grid size
        
    Returns:
        Grid coordinates (i, j)
    """
    i = idx // size
    j = idx % size
    return i, j


def count_agent_types(grid: np.ndarray) -> dict:
    """
    Count the number of each agent type in the grid.
    
    Args:
        grid: 2D numpy array (0=empty, 1=red, 2=blue)
        
    Returns:
        Dictionary with counts for each type
    """
    unique, counts = np.unique(grid, return_counts=True)
    result = {'empty': 0, 'red': 0, 'blue': 0}
    
    for value, count in zip(unique, counts):
        if value == 0:
            result['empty'] = count
        elif value == 1:
            result['red'] = count
        elif value == 2:
            result['blue'] = count
            
    return result


def validate_grid_configuration(grid: np.ndarray, expected_counts: dict) -> bool:
    """
    Validate that grid matches expected agent counts.
    
    Args:
        grid: 2D numpy array representing the grid
        expected_counts: Dictionary with expected counts
        
    Returns:
        True if grid matches expected configuration
    """
    actual_counts = count_agent_types(grid)
    
    for agent_type in ['empty', 'red', 'blue']:
        if actual_counts[agent_type] != expected_counts[agent_type]:
            return False
            
    return True


def calculate_grid_energy(grid: np.ndarray, size: int, wrapping: bool = True) -> int:
    """
    Calculate the 'energy' of a grid configuration based on unlike neighbor pairs.
    
    Lower energy corresponds to higher segregation (fewer unlike neighbors).
    
    Args:
        grid: 2D numpy array (0=empty, 1=red, 2=blue)
        size: Grid size
        wrapping: Whether to use wrapping boundaries
        
    Returns:
        Energy value (number of unlike neighbor pairs)
    """
    energy = 0
    
    for i in range(size):
        for j in range(size):
            if grid[i, j] == 0:  # Skip empty spaces
                continue
                
            neighbors = get_neighbors(i, j, size, wrapping)
            for ni, nj in neighbors:
                neighbor_type = grid[ni, nj]
                if neighbor_type != 0 and neighbor_type != grid[i, j]:
                    # Unlike neighbor pair - contributes to energy
                    # Divide by 2 to avoid double counting
                    energy += 0.5
                    
    return int(energy)


if __name__ == "__main__":
    # Test functions
    size = 3
    
    # Test neighbor calculation
    neighbors = get_neighbors(1, 1, size, wrapping=True)
    print(f"Neighbors of (1,1) in 3x3 grid with wrapping: {neighbors}")
    
    # Test neighbor pairs
    pairs = get_neighbor_pairs(size, wrapping=True)
    print(f"Total neighbor pairs: {len(pairs)}")
    print(f"First few pairs: {pairs[:5]}")
    
    # Test grid energy calculation
    test_grid = np.array([[1, 2, 0], [2, 1, 2], [0, 1, 1]])
    energy = calculate_grid_energy(test_grid, size)
    print(f"Test grid energy: {energy}")
    print(f"Test grid:\n{test_grid}")