"""
Visualization utilities for Schelling model results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import seaborn as sns


def plot_grid(
    grid: np.ndarray, 
    title: str = "Schelling Grid",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6)
) -> None:
    """
    Plot a single Schelling model grid.
    
    Args:
        grid: 2D numpy array (0=empty, 1=red, 2=blue)
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap: empty (white), red, blue
    colors = ['white', 'red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Plot grid
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)
    
    # Add grid lines
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)
    
    # Configure plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_aspect('equal')
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Empty', 'Red', 'Blue'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to {save_path}")
    
    plt.show()


def plot_grid_evolution(
    grid_history: List[np.ndarray],
    title: str = "Schelling Model Evolution",
    save_path: Optional[str] = None,
    max_steps: int = 8
) -> None:
    """
    Plot the evolution of the grid over time.
    
    Args:
        grid_history: List of grid states over time
        title: Overall plot title
        save_path: Path to save the figure
        max_steps: Maximum number of steps to show
    """
    n_steps = min(len(grid_history), max_steps)
    
    # Calculate subplot dimensions
    cols = min(4, n_steps)
    rows = (n_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if n_steps == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Custom colormap
    colors = ['white', 'red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    for step in range(n_steps):
        ax = axes[step]
        grid = grid_history[step]
        
        # Plot grid
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)
        
        # Add grid lines
        for i in range(grid.shape[0] + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(grid.shape[1] + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5)
        
        # Configure subplot
        ax.set_title(f'Step {step}', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for step in range(n_steps, len(axes)):
        axes[step].set_visible(False)
    
    # Add overall title and colorbar
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Add single colorbar for all subplots
    cbar = fig.colorbar(im, ax=axes[:n_steps], ticks=[0, 1, 2], shrink=0.8)
    cbar.set_ticklabels(['Empty', 'Red', 'Blue'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution plot saved to {save_path}")
    
    plt.show()


def plot_satisfaction_curves(
    satisfaction_history: List[float],
    title: str = "Satisfaction Rate Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot satisfaction rate evolution over simulation steps.
    
    Args:
        satisfaction_history: List of satisfaction rates over time
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    steps = range(len(satisfaction_history))
    
    # Plot satisfaction curve
    ax.plot(steps, satisfaction_history, 'o-', linewidth=2, markersize=6,
            color='darkblue', label='Satisfaction Rate')
    
    # Add horizontal line at 100% satisfaction
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
               label='Perfect Satisfaction')
    
    # Configure plot
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Satisfaction Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Set axis limits
    ax.set_xlim(-0.5, len(satisfaction_history) - 0.5)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Satisfaction curve saved to {save_path}")
    
    plt.show()


def plot_segregation_comparison(
    results_dict: dict,
    x_param: str,
    title: str = "Segregation Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of segregation indices across different parameter values.
    
    Args:
        results_dict: Dictionary mapping parameter values to results
        x_param: Name of the x-axis parameter
        title: Plot title
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x_values = list(results_dict.keys())
    segregation_indices = [results_dict[x]['segregation_index'] for x in x_values]
    satisfaction_rates = [results_dict[x]['satisfaction_rate'] for x in x_values]
    steps_taken = [results_dict[x]['steps_taken'] for x in x_values]
    
    # Plot 1: Segregation Index vs Parameter
    ax1.plot(x_values, segregation_indices, 'o-', linewidth=2, markersize=8,
             color='darkred', label='Segregation Index')
    ax1.set_xlabel(x_param, fontsize=12)
    ax1.set_ylabel('Segregation Index', fontsize=12)
    ax1.set_title(f'Segregation vs {x_param}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Multiple metrics
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(x_values, satisfaction_rates, 'o-', linewidth=2, 
                     color='green', label='Satisfaction Rate')
    line2 = ax2_twin.plot(x_values, steps_taken, 's-', linewidth=2, 
                          color='orange', label='Steps to Convergence')
    
    ax2.set_xlabel(x_param, fontsize=12)
    ax2.set_ylabel('Satisfaction Rate', fontsize=12, color='green')
    ax2_twin.set_ylabel('Steps to Convergence', fontsize=12, color='orange')
    ax2.set_title(f'Performance vs {x_param}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Format satisfaction rate as percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.set_ylim(0, 1.05)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def plot_quantum_classical_comparison(
    classical_results: dict,
    quantum_results: dict,
    save_path: Optional[str] = None
) -> None:
    """
    Compare classical and quantum optimization results.
    
    Args:
        classical_results: Results from classical simulation
        quantum_results: Results from quantum optimization
        save_path: Path to save the figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Final grids comparison
    if 'final_grid' in classical_results:
        colors = ['white', 'red', 'blue']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        im1 = ax1.imshow(classical_results['final_grid'], cmap=cmap, vmin=0, vmax=2)
        ax1.set_title('Classical Result', fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        if 'final_grid' in quantum_results:
            im2 = ax2.imshow(quantum_results['final_grid'], cmap=cmap, vmin=0, vmax=2)
            ax2.set_title('Quantum Result', fontsize=14)
            ax2.set_xticks([])
            ax2.set_yticks([])
    
    # Plot 3: Performance metrics comparison
    metrics = ['satisfaction_rate', 'segregation_index', 'convergence_steps']
    classical_values = [classical_results.get(m, 0) for m in metrics]
    quantum_values = [quantum_results.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, classical_values, width, label='Classical', color='lightblue')
    ax3.bar(x + width/2, quantum_values, width, label='Quantum', color='lightcoral')
    
    ax3.set_xlabel('Metrics', fontsize=12)
    ax3.set_ylabel('Values', fontsize=12)
    ax3.set_title('Performance Comparison', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Satisfaction', 'Segregation', 'Steps'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence curves (if available)
    if 'satisfaction_history' in classical_results:
        ax4.plot(classical_results['satisfaction_history'], 'o-', 
                label='Classical', linewidth=2)
    
    if 'cost_history' in quantum_results:
        # Convert cost to satisfaction-like metric
        normalized_costs = 1 - np.array(quantum_results['cost_history']) / max(quantum_results['cost_history'])
        ax4.plot(normalized_costs, 's-', label='Quantum', linewidth=2)
    
    ax4.set_xlabel('Iteration/Step', fontsize=12)
    ax4.set_ylabel('Progress Metric', fontsize=12)
    ax4.set_title('Convergence Comparison', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Classical vs Quantum Optimization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test visualization functions with dummy data
    print("Testing visualization functions...")
    
    # Create test grid
    test_grid = np.array([[1, 2, 0], [0, 1, 2], [2, 0, 1]])
    
    # Test single grid plot
    plot_grid(test_grid, title="Test Grid")
    
    # Test evolution plot
    evolution = [test_grid, 
                np.roll(test_grid, 1, axis=0),
                np.roll(test_grid, 1, axis=1)]
    plot_grid_evolution(evolution, title="Test Evolution")
    
    # Test satisfaction curve
    satisfaction_data = [0.4, 0.6, 0.8, 0.9, 1.0]
    plot_satisfaction_curves(satisfaction_data, title="Test Satisfaction")
    
    print("Visualization tests complete!")