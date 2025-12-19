#!/usr/bin/env python3
"""
Script to run the core ACT experiment and reproduce the prediction for α⁻¹.
"""

from act_model import ACTAdvancedModel
from datetime import datetime
import numpy as np

print("="*60)
print("Algebraic Causality Theory: Fine-Structure Constant Prediction")
print("="*60)

# Initialize the model with a fixed seed for reproducibility
print("\n[1/3] Initializing ACT model...")
model = ACTAdvancedModel(
    n_vertices=60,       # Number of vertices in the causal graph
    temperature=1.0,     # System temperature
    use_clifford=True,   # Use Clifford algebra Cℓ(1,3)
    seed=42              # Seed for reproducible results
)

# Run the simulation
print("\n[2/3] Running Monte Carlo simulation...")
measurements = model.simulate(
    n_steps=500,          # Number of Monte Carlo steps
    thermalization=100,   # Thermalization steps
    measure_interval=10   # Measure α every 10 steps
)

# Analyze results
print("\n[3/3] Analyzing results...")
if measurements:
    alphas = [m['alpha_precise'] for m in measurements]
    alphas = np.array(alphas)
    alphas = alphas[alphas > 0]  # Filter out zeros
    
    if len(alphas) > 0:
        alpha_mean = np.mean(alphas)
        alpha_std = np.std(alphas)
        
        print(f"\n--- RESULT ---")
        print(f"Predicted α⁻¹ = {alpha_mean:.2f} ± {alpha_std:.2f}")
        print(f"Accepted value = 137.036")
        print(f"Relative deviation = {abs(alpha_mean-137.036)/137.036*100:.2f}%")
        
        # Save a simple report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        with open(f'result_alpha_{timestamp}.txt', 'w') as f:
            f.write(f"ACT Prediction Run\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Predicted α⁻¹: {alpha_mean:.4f} ± {alpha_std:.4f}\n")
            f.write(f"Number of samples: {len(alphas)}\n")
        print(f"\nDetailed report saved to 'result_alpha_{timestamp}.txt'")
    else:
        print("No valid α measurements were collected.")
else:
    print("Simulation failed to produce measurements.")

print("\n" + "="*60)
print("Experiment complete. Check the saved files for results.")
print("="*60)
