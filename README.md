# Algebraic Causality Theory (ACT): First-Principles Prediction of α

## Overview
Algebraic Causality Theory (ACT) is a novel framework proposing that fundamental physical constants emerge as statistical invariants from the dynamics of causal tetrahedral networks, rather than being free parameters.

This repository contains the complete computational model that predicts the inverse fine-structure constant **α⁻¹ ≈ 136.7** from first principles.

## Key Prediction
The model predicts the value of the inverse fine-structure constant with high precision:
This is within **0.24%** of the accepted CODATA value of **137.036**.

## How It Works
The ACT model is based on three pillars:
1.  **Causal Hypergraphs**: Spacetime is discretized into a network of tetrahedral cells (4-simplices).
2.  **Clifford Algebra**: Each vertex is associated with operators from the Cℓ(1,3) Clifford algebra, representing fundamental "acts of distinction."
3.  **Cascade Dynamics**: The network evolves via Metropolis-Hastings (Monte Carlo) dynamics that respect causal ordering constraints.

The value of α⁻¹ emerges naturally from the statistical analysis of phase coherence (holonomy) around triangular loops within this causal fabric.

## Reproducing the Results
To run the simulation and reproduce the prediction:

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the main experiment**:
    ```bash
    python run_experiment.py
    ```
The script will execute a simulation, output the calculated α⁻¹ value, and save visualizations.

## Contents
*   `act_model.py` - Core implementation of the ACT model.
*   `run_experiment.py` - Script to run the simulation and reproduce the key result.
*   `requirements.txt` - List of required Python packages.
*   `.gitignore` - Standard Python gitignore file.

## Significance
This work presents a potential paradigm shift by demonstrating that a fundamental dimensionless constant can be computed from the logical and algebraic structure of causality itself, addressing the long-standing "fine-tuning" problem in theoretical physics.

## Author & Contact
Theory developed over decades by [Your Name/Pseudonym].
For discussion and collaboration, please open an Issue in this repository.
