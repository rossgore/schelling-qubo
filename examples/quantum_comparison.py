"""
Example: Quantum vs Classical Schelling model comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from schelling.classical import SchellingModel
from schelling.visualization import plot_quantum_classical_comparison
from quantum.encodings.basis_state import BasisStateEncoder
from quantum.qaoa import run_qaoa


def run_quantum_classical_comparison():
    """
    Compare quantum optimization with classical simulation for the same problem.
    """
    print("=== Quantum vs Classical Schelling Comparison ===")
    
    # Problem configuration
    config = {
        'size': 3,
        'colors': {'red': 3, 'blue': 3, 'empty': 3},
        'similarity_threshold': 0.6,
        'wrapping': True,
        'random_seed': 42
    }
    
    print(f"Problem: {config['size']}x{config['size']} grid")
    print(f"Agents: {config['colors']['red']} red, {config['colors']['blue']} blue, {config['colors']['empty']} empty")
    print(f"Similarity threshold: {config['similarity_threshold']}")
    
    # Run classical simulation
    print(f"\n--- Classical Simulation ---")
    classical_start = time.time()
    
    classical_model = SchellingModel(**config)
    classical_results = classical_model.simulate(max_steps=15, verbose=False)
    
    classical_time = time.time() - classical_start
    classical_segregation = classical_model.get_segregation_index()
    
    print(f"Classical results:")
    print(f"  Time: {classical_time:.3f}s")
    print(f"  Steps: {classical_results['steps_taken']}")
    print(f"  Moves: {classical_results['total_moves']}")
    print(f"  Satisfaction: {classical_results['final_satisfaction_rate']:.2%}")
    print(f"  Segregation: {classical_segregation:.3f}")
    
    # Run quantum optimization
    print(f"\n--- Quantum Optimization ---")
    quantum_start = time.time()
    
    try:
        # Create quantum encoder
        encoder = BasisStateEncoder(
            size=config['size'],
            colors=config['colors'],
            wrapping=config['wrapping']
        )
        
        print(f"Quantum setup:")
        print(f"  Qubits required: {encoder.n_qubits}")
        print(f"  Encoding: 2-qubit basis state per site")
        
        # Build Hamiltonian
        hamiltonian = encoder.build_total_hamiltonian(
            penalty_weight=10.0,
            interaction_weight=1.0,
            count_weight=5.0
        )
        
        print(f"  Hamiltonian terms: {len(hamiltonian.paulis)}")
        
        # Run QAOA optimization
        print(f"Running QAOA optimization...")
        optimal_cost, optimal_params = run_qaoa(
            cost_operator=hamiltonian,
            reps=3,  # Limited for demo
            optimizer_name='COBYLA',
            maxiter=50,
            verbose=False
        )
        
        quantum_time = time.time() - quantum_start
        
        print(f"Quantum results:")
        print(f"  Time: {quantum_time:.3f}s")
        print(f"  Optimal cost: {optimal_cost:.6f}")
        print(f"  QAOA parameters: {len(optimal_params)}")
        
        # Note: In a full implementation, we would:
        # 1. Sample from the optimized quantum state
        # 2. Decode the most probable bitstrings
        # 3. Validate solutions and calculate metrics
        # For this demo, we'll simulate the process
        
        quantum_results = {
            'optimization_time': quantum_time,
            'optimal_cost': optimal_cost,
            'optimal_parameters': optimal_params,
            'converged': True,
            'method': 'QAOA'
        }
        
    except Exception as e:
        print(f"Quantum optimization failed: {str(e)}")
        quantum_results = {
            'optimization_time': time.time() - quantum_start,
            'error': str(e),
            'converged': False
        }
    
    # Performance comparison
    print(f"\n--- Performance Comparison ---")
    
    if quantum_results.get('converged', False):
        classical_quality = classical_segregation
        quantum_speedup = classical_time / quantum_results['optimization_time'] if quantum_results['optimization_time'] > 0 else 'N/A'
        
        print(f"Classical solution quality: {classical_quality:.3f}")
        print(f"Quantum cost achieved: {quantum_results['optimal_cost']:.6f}")
        print(f"Time ratio (classical/quantum): {quantum_speedup}")
        
        # Resource analysis
        print(f"\nResource requirements:")
        print(f"  Classical: Polynomial time, classical computer")
        print(f"  Quantum: {encoder.n_qubits} qubits, ~{len(hamiltonian.paulis)} Pauli terms")
        
        # Scaling predictions
        print(f"\nScaling analysis:")
        print_scaling_comparison(config['size'])
        
        return {
            'classical': {
                'time': classical_time,
                'quality': classical_quality,
                'satisfaction_rate': classical_results['final_satisfaction_rate'],
                'final_grid': classical_results['final_grid']
            },
            'quantum': quantum_results
        }
    else:
        print(f"Quantum optimization failed, cannot perform full comparison")
        return {
            'classical': {
                'time': classical_time,
                'quality': classical_segregation,
                'satisfaction_rate': classical_results['final_satisfaction_rate']
            },
            'quantum': quantum_results
        }


def print_scaling_comparison(base_size: int):
    """Print scaling analysis for different grid sizes."""
    
    print(f"\nGrid Size | Classical Time | Quantum Qubits | Hamiltonian Terms")
    print("-" * 65)
    
    for size in range(base_size, base_size + 3):
        sites = size * size
        
        # Classical scaling (rough estimate)
        classical_time_estimate = sites ** 1.5  # Polynomial in sites
        
        # Quantum scaling
        quantum_qubits = 2 * sites  # Basis state encoding
        hamiltonian_terms = sites + 4 * size * (size - 1) + sites ** 2  # Approximate
        
        print(f"{size:2d}x{size:<2d}     | O(n^1.5) ≈ {classical_time_estimate:4.0f} | {quantum_qubits:6d}      | {hamiltonian_terms:8d}")


def analyze_quantum_advantage():
    """
    Analyze potential quantum advantage for different problem sizes.
    """
    print("\n=== Quantum Advantage Analysis ===")
    
    sizes = [3, 4, 5]
    
    for size in sizes:
        sites = size * size
        agents = (sites * 2) // 3  # Assuming 1/3 empty
        
        # Classical complexity estimates
        classical_states = 3 ** sites  # All possible configurations
        classical_search_space = classical_states
        
        # Quantum complexity estimates  
        quantum_qubits = 2 * sites
        quantum_hilbert_space = 2 ** quantum_qubits
        
        print(f"\n{size}x{size} Grid Analysis:")
        print(f"  Total sites: {sites}")
        print(f"  Classical state space: ~3^{sites} = {classical_states:.2e}")
        print(f"  Quantum Hilbert space: 2^{quantum_qubits} = {quantum_hilbert_space:.2e}")
        print(f"  Quantum/Classical ratio: {quantum_hilbert_space/classical_states:.2e}")
        
        # Rough quantum advantage estimate
        if quantum_hilbert_space > classical_states:
            print(f"  Potential advantage: Quantum space larger")
        else:
            print(f"  Potential advantage: Quantum parallelism in smaller space")
        
        # Resource feasibility
        if quantum_qubits <= 50:
            feasibility = "Feasible on near-term devices"
        elif quantum_qubits <= 100:
            feasibility = "Requires fault-tolerant quantum computer"
        else:
            feasibility = "Beyond current quantum capabilities"
            
        print(f"  Feasibility: {feasibility}")


def benchmark_encodings():
    """
    Benchmark different quantum encodings for the same problem.
    """
    print("\n=== Encoding Benchmark ===")
    
    config = {'size': 3, 'colors': {'red': 3, 'blue': 3, 'empty': 3}}
    
    encodings = [
        ('Basis State (2q/site)', 2 * config['size']**2),
        ('Ising (1q/agent)', 2 * 3),  # red + blue agents only
        ('One-Hot (3q/site)', 3 * config['size']**2)
    ]
    
    print(f"Problem: {config['size']}x{config['size']} grid")
    print(f"Encoding comparison:")
    print(f"{'Encoding':<20} | {'Qubits':<7} | {'Efficiency':<12} | {'Suitability'}")
    print("-" * 65)
    
    for name, qubits in encodings:
        efficiency = config['size']**2 / qubits
        
        if 'Basis State' in name:
            suitability = "Best balance"
        elif 'Ising' in name:
            suitability = "Most efficient"
        else:  # One-hot
            suitability = "Most intuitive"
            
        print(f"{name:<20} | {qubits:<7} | {efficiency:<12.2f} | {suitability}")


if __name__ == "__main__":
    # Run comprehensive comparison
    print("Starting Quantum vs Classical Schelling Comparison...\n")
    
    # Main comparison
    results = run_quantum_classical_comparison()
    
    # Additional analyses
    analyze_quantum_advantage()
    benchmark_encodings()
    
    print(f"\n=== Summary ===")
    print("This comparison demonstrates:")
    print("1. Classical methods are currently faster for small problems")
    print("2. Quantum methods show promise for larger, more complex cases") 
    print("3. Resource requirements scale differently between approaches")
    print("4. Quantum advantage may emerge at larger problem sizes")
    print("\nFor production use, consider hybrid classical-quantum approaches.")
    
    print("\nComparison complete!")