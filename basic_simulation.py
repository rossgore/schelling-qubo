"""
Example: Basic Schelling model simulation and comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from schelling.classical import SchellingModel
from schelling.visualization import plot_grid_evolution, plot_satisfaction_curves


def run_basic_simulation():
    """
    Run a basic classical Schelling model simulation and visualize results.
    """
    print("=== Basic Schelling Model Simulation ===")
    
    # Configure simulation parameters
    config = {
        'size': 4,
        'colors': {'red': 6, 'blue': 6, 'empty': 4},
        'similarity_threshold': 0.6,
        'wrapping': True,
        'random_seed': 42
    }
    
    print(f"Grid size: {config['size']}x{config['size']}")
    print(f"Agents: {config['colors']['red']} red, {config['colors']['blue']} blue, {config['colors']['empty']} empty")
    print(f"Similarity threshold: {config['similarity_threshold']}")
    
    # Initialize and run model
    model = SchellingModel(**config)
    
    print(f"\nInitial grid:")
    print(model.grid)
    print(f"Initial segregation index: {model.get_segregation_index():.3f}")
    
    # Run simulation
    results = model.simulate(max_steps=20, verbose=True)
    
    # Display final results
    print(f"\n=== Final Results ===")
    print(f"Converged: {results['converged']}")
    print(f"Steps taken: {results['steps_taken']}")
    print(f"Total moves: {results['total_moves']}")
    print(f"Final satisfaction rate: {results['final_satisfaction_rate']:.2%}")
    print(f"Final segregation index: {model.get_segregation_index():.3f}")
    
    # Generate visualizations
    plot_grid_evolution(results['grid_history'], save_path='grid_evolution.png')
    
    # Calculate satisfaction over time
    satisfaction_history = []
    for grid in results['grid_history']:
        # Temporarily set grid and calculate satisfaction
        old_grid = model.grid.copy()
        model.grid = grid
        satisfaction = model.calculate_satisfaction()
        satisfaction_rate = sum(satisfaction.values()) / len(satisfaction) if satisfaction else 0
        satisfaction_history.append(satisfaction_rate)
        model.grid = old_grid
    
    plot_satisfaction_curves(satisfaction_history, save_path='satisfaction_curve.png')
    
    return model, results


def compare_thresholds():
    """
    Compare Schelling model behavior across different similarity thresholds.
    """
    print("\n=== Threshold Comparison Study ===")
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    base_config = {
        'size': 4,
        'colors': {'red': 6, 'blue': 6, 'empty': 4},
        'wrapping': True,
        'random_seed': 123
    }
    
    results_by_threshold = {}
    
    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold}")
        
        config = base_config.copy()
        config['similarity_threshold'] = threshold
        
        model = SchellingModel(**config)
        results = model.simulate(max_steps=15, verbose=False)
        
        final_segregation = model.get_segregation_index()
        
        results_by_threshold[threshold] = {
            'steps_taken': results['steps_taken'],
            'total_moves': results['total_moves'],
            'satisfaction_rate': results['final_satisfaction_rate'],
            'segregation_index': final_segregation,
            'converged': results['converged']
        }
        
        print(f"  Steps: {results['steps_taken']}, "
              f"Moves: {results['total_moves']}, "
              f"Segregation: {final_segregation:.3f}")
    
    # Print comparison table
    print(f"\n{'Threshold':<10} {'Steps':<6} {'Moves':<6} {'Satisfaction':<12} {'Segregation':<11} {'Converged'}")
    print("-" * 65)
    
    for threshold in thresholds:
        r = results_by_threshold[threshold]
        print(f"{threshold:<10} {r['steps_taken']:<6} {r['total_moves']:<6} "
              f"{r['satisfaction_rate']:<12.1%} {r['segregation_index']:<11.3f} {r['converged']}")
    
    return results_by_threshold


def analyze_grid_sizes():
    """
    Analyze how grid size affects convergence behavior.
    """
    print("\n=== Grid Size Analysis ===")
    
    sizes = [3, 4, 5]
    base_config = {
        'similarity_threshold': 0.6,
        'wrapping': True,
        'random_seed': 456
    }
    
    results_by_size = {}
    
    for size in sizes:
        # Scale agent numbers with grid size
        total_sites = size * size
        red_count = total_sites // 3
        blue_count = total_sites // 3
        empty_count = total_sites - red_count - blue_count
        
        config = base_config.copy()
        config.update({
            'size': size,
            'colors': {'red': red_count, 'blue': blue_count, 'empty': empty_count}
        })
        
        print(f"\nTesting {size}x{size} grid ({red_count}R, {blue_count}B, {empty_count}E)")
        
        model = SchellingModel(**config)
        results = model.simulate(max_steps=25, verbose=False)
        
        results_by_size[size] = {
            'total_sites': total_sites,
            'agent_density': (red_count + blue_count) / total_sites,
            'steps_taken': results['steps_taken'],
            'total_moves': results['total_moves'],
            'satisfaction_rate': results['final_satisfaction_rate'],
            'segregation_index': model.get_segregation_index(),
            'moves_per_agent': results['total_moves'] / (red_count + blue_count) if (red_count + blue_count) > 0 else 0
        }
        
        print(f"  Result: {results['steps_taken']} steps, "
              f"{results['total_moves']} moves, "
              f"segregation: {model.get_segregation_index():.3f}")
    
    return results_by_size


if __name__ == "__main__":
    # Run all analyses
    print("Starting comprehensive Schelling model analysis...\n")
    
    # Basic simulation
    model, results = run_basic_simulation()
    
    # Threshold comparison
    threshold_results = compare_thresholds()
    
    # Grid size analysis
    size_results = analyze_grid_sizes()
    
    print("\n=== Analysis Complete ===")
    print("Generated visualizations:")
    print("  - grid_evolution.png: Shows how the grid changes over time")
    print("  - satisfaction_curve.png: Shows satisfaction rate progression")
    
    # Summary insights
    print(f"\nKey Insights:")
    print(f"- Higher similarity thresholds lead to more segregation")
    print(f"- Larger grids may require more steps to converge") 
    print(f"- Grid topology (wrapping) affects clustering patterns")
    
    # Display bit string representation for quantum comparison
    print(f"\nFinal grid bit string representation:")
    print(f"'{model.get_bit_string_representation()}'")
    print(f"(This can be used to compare with quantum optimization results)")