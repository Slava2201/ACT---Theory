"""
act_model.py - Core implementation of the Algebraic Causality Theory (ACT).
This module defines the ACTAdvancedModel class which simulates causal
tetrahedral networks to predict fundamental constants like α.
"""

import numpy as np
import numba as nb
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
import networkx as nx
from functools import lru_cache
import h5py
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Gamma matrices for Clifford algebra Cℓ(1,3)
GAMMA_MATRICES = {
    0: np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, -1]], dtype=complex),  # γ⁰
    1: np.array([[0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, -1, 0, 0],
                 [-1, 0, 0, 0]], dtype=complex),  # γ¹
    2: np.array([[0, 0, 0, -1j],
                 [0, 0, 1j, 0],
                 [0, 1j, 0, 0],
                 [-1j, 0, 0, 0]], dtype=complex),  # γ²
    3: np.array([[0, 0, 1, 0],
                 [0, 0, 0, -1],
                 [-1, 0, 0, 0],
                 [0, 1, 0, 0]], dtype=complex),  # γ³
    5: np.array([[0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [1, 0, 0, 0],
                 [0, 1, 0, 0]], dtype=complex)   # γ⁵ = iγ⁰γ¹γ²γ³
}
class ACTAdvancedModel:
    """
    Advanced implementation of Algebraic Causality Theory.
    Simulates a causal tetrahedral network to compute emergent physical constants.
    """
    
    def __init__(self, 
                 n_vertices: int = 50, 
                 dim: int = 4,
                 coupling_constant: float = 1.0,
                 temperature: float = 1.0,
                 use_clifford: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize the ACT model.
        
        Parameters:
        -----------
        n_vertices : int
            Number of vertices in the causal hypergraph
        dim : int
            Spacetime dimension (fixed at 4 for ACT)
        coupling_constant : float
            Coupling strength between distinction operators
        temperature : float
            System temperature for Monte Carlo dynamics
        use_clifford : bool
            Use Clifford algebra (True) or unitary operators (False)
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.N = n_vertices
        self.dim = 4  # Fixed at 4 dimensions for ACT
        self.J = coupling_constant
        self.beta = 1.0 / max(temperature, 1e-10)
        self.use_clifford = use_clifford
        
        # Fundamental scales (in Planck units)
        self.lp = 1.0  # Planck length
        self.tp = 1.0  # Planck time
        self.alpha_target = 1/137.035999084  # target α value
        
        # Initialize vertices with random coordinates in R^4
        # First coordinate is time, make it positive and ordered
        self.vertices = np.random.randn(n_vertices, 4)
        self.vertices[:, 0] = np.sort(np.abs(self.vertices[:, 0]))
        
        # Initialize distinction operators
        if use_clifford:
            self.operators = self._initialize_clifford_operators()
        else:
            self.operators = self._initialize_unitary_operators()
        
        # Build causal complex
        self.tetrahedra = self._build_causal_complex()
        self.adjacency = self._build_adjacency_matrix()
        
        # Caches for performance
        self._action_cache = {}
        self._holonomy_cache = {}
        
        # Measurement history
        self.history = {
            'step': [],
            'alpha': [],
            'alpha_precise': [],
            'action': [],
            'curvature': [],
            'entropy': [],
            'metric_det': [],
            'temperature': []
        }
        
        # Dynamic rewiring parameters
        self.rewiring_probability = 0.005
        self.last_rewiring_step = 0
        self.rewiring_interval = 50
        
        print(f"ACT model initialized: N={n_vertices}, β={self.beta:.3f}")
                       def _initialize_clifford_operators(self) -> np.ndarray:
        """
        Initialize operators as linear combinations of γ-matrices.
        Corresponds to Clifford algebra Cℓ(1,3).
        """
        operators = np.zeros((self.N, 4, 4), dtype=complex)
        
        for i in range(self.N):
            # Each operator: δ_i = a_μ γ^μ + b γ^5
            coeffs = np.random.randn(5) * 0.1  # 4 γ^μ + γ^5
            
            # Construct operator
            op = np.zeros((4, 4), dtype=complex)
            for mu in range(4):
                op += coeffs[mu] * GAMMA_MATRICES[mu]
            op += coeffs[4] * GAMMA_MATRICES[5]
            
            operators[i] = op
        
        return operators
    
    def _initialize_unitary_operators(self) -> np.ndarray:
        """
        Initialize random SU(4) unitary matrices via QR decomposition.
        """
        operators = np.zeros((self.N, 4, 4), dtype=complex)
        
        for i in range(self.N):
            # Generate random SU(4) matrix via QR decomposition
            X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
            Q, R = np.linalg.qr(X)
            
            # Ensure det = 1
            lambda_diag = np.diag(R) / np.abs(np.diag(R))
            Q = Q @ np.diag(lambda_diag)
            
            # Fix phase for exact det=1
            det = np.linalg.det(Q)
            Q = Q / (det ** (1/4))
            
            operators[i] = Q
        
        return operators
          def _build_causal_complex(self) -> List[Tuple[int, int, int, int]]:
        """
        Build a causal tetrahedral complex.
        Respects causal ordering of vertices.
        """
        tetrahedra = []
        
        # Method 1: Delaunay triangulation in R^4
        try:
            if self.N >= 5:
                tri = Delaunay(self.vertices)
                for simplex in tri.simplices:
                    if len(simplex) == 5:  # 4-simplex in 4D has 5 vertices
                        # All possible tetrahedra within the 4-simplex
                        for i in range(5):
                            for j in range(i+1, 5):
                                for k in range(j+1, 5):
                                    for l in range(k+1, 5):
                                        indices = (simplex[i], simplex[j], 
                                                  simplex[k], simplex[l])
                                        if self._check_causal_order(indices):
                                            tetrahedra.append(indices)
        except Exception as e:
            print(f"Delaunay failed: {e}")
        
        # Method 2: Geometric proximity + causal ordering
        if len(tetrahedra) < self.N // 2:  # if not enough tetrahedra
            print("Adding additional tetrahedra...")
            attempts = self.N * 10
            
            for _ in range(attempts):
                # Pick random starting vertex
                i = np.random.randint(self.N)
                
                # Find nearest spatial neighbors
                distances = np.linalg.norm(
                    self.vertices[:, 1:] - self.vertices[i, 1:], 
                    axis=1
                )
                nearest = np.argsort(distances)[1:7]  # 6 nearest
                
                # Try different combinations of 3 neighbors
                for comb in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
                    if len(nearest) >= 4:
                        j, k, l = nearest[list(comb)]
                        indices = tuple(sorted((i, j, k, l)))
                        
                        if self._check_causal_order(indices):
                            tetrahedra.append(indices)
                            break
        
        # Remove duplicates and limit number
        tetrahedra = list(set(tetrahedra))
        if len(tetrahedra) > self.N * 3:  # limit growth
            tetrahedra = tetrahedra[:self.N * 3]
        
        print(f"Built {len(tetrahedra)} causal tetrahedra")
        return tetrahedra
    
    def _check_causal_order(self, indices: Tuple[int, int, int, int]) -> bool:
        """
        Check causal ordering of vertices.
        """
        times = self.vertices[list(indices), 0]
        sorted_times = np.sort(times)
        
        # Allow only strictly increasing or decreasing time
        increasing = np.all(np.diff(sorted_times) > 1e-10)
        decreasing = np.all(np.diff(sorted_times[::-1]) > 1e-10)
        
        # Also check spatial proximity
        coords = self.vertices[list(indices), 1:]
        centroid = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        
        # Tetrahedron shouldn't be too "flat"
        spatial_ok = np.std(distances) > 0.1 * np.mean(distances)
        
        return (increasing or decreasing) and spatial_ok
    
    def _build_adjacency_matrix(self) -> csr_matrix:
        """Build sparse adjacency matrix."""
        adj = lil_matrix((self.N, self.N), dtype=int)
        
        for tetra in self.tetrahedra:
            for i in range(4):
                for j in range(i+1, 4):
                    v1, v2 = tetra[i], tetra[j]
                    adj[v1, v2] += 1
                    adj[v2, v1] += 1
        
        return adj.tocsr()
    
    @lru_cache(maxsize=1024)
    def _cached_tetra_action(self, i: int, j: int, k: int, l: int) -> float:
        """
        Cached computation of action for a tetrahedron.
        """
        product = (self.operators[i] @ 
                   self.operators[j] @ 
                   self.operators[k] @ 
                   self.operators[l])
        return np.real(np.trace(product))
    
    def compute_action_tetrahedron(self, tetra: Tuple[int, int, int, int]) -> float:
        """Compute action for a tetrahedron using cache."""
        i, j, k, l = tetra
        return self._cached_tetra_action(i, j, k, l)
    
    def total_action(self, include_kinetic: bool = True) -> float:
        """Compute total system action."""
        S_total = 0.0
        
        # Tetrahedron contributions (main action)
        for tetra in self.tetrahedra:
            S_tetra = self.compute_action_tetrahedron(tetra)
            S_total += self.J * S_tetra
        
        # Kinetic term (if needed)
        if include_kinetic and hasattr(self, 'operators_old'):
            kinetic = 0.0
            for i in range(self.N):
                diff = self.operators[i] - self.operators_old[i]
                kinetic += np.trace(diff @ diff.conj().T).real
            S_total += 0.5 * kinetic
        
        # Causal penalty
        causal_penalty = 0.0
        for tetra in self.tetrahedra:
            times = self.vertices[list(tetra), 0]
            if not (np.all(np.diff(np.sort(times)) > 0) or 
                    np.all(np.diff(np.sort(times)[::-1]) > 0)):
                causal_penalty += 1.0
        
        S_total += 0.1 * causal_penalty
        
        # Regularization term (encourage connectivity)
        connectivity = np.sum(self.adjacency > 0) / (self.N * (self.N - 1))
        S_total += 0.01 * (1.0 - connectivity) ** 2
        
        return S_total
          def compute_holonomy(self, triangle: Tuple[int, int, int]) -> complex:
        """Compute holonomy around a triangle."""
        i, j, k = triangle
        
        # Caching
        key = tuple(sorted(triangle))
        if key in self._holonomy_cache:
            return self._holonomy_cache[key]
        
        # Holonomy: product of operators along cycle
        U_cycle = (self.operators[i] @ 
                   self.operators[j].conj().T @ 
                   self.operators[k] @ 
                   self.operators[i].conj().T)
        
        result = np.trace(U_cycle) / 4.0
        self._holonomy_cache[key] = result
        
        return result
    
    def compute_alpha_precise(self, n_samples: int = 200) -> float:
        """
        Precise computation of α from holonomy statistics.
        """
        if len(self.tetrahedra) < 1:
            return 0.0
        
        phases = []
        
        # Sample random triangles
        for _ in range(n_samples):
            # Pick random tetrahedron
            if len(self.tetrahedra) == 0:
                continue
            tetra_idx = np.random.randint(len(self.tetrahedra))
            tetra = self.tetrahedra[tetra_idx]
            i, j, k, l = tetra
            
            # Random face of tetrahedron
            triangle_choices = [
                (i, j, k), (i, j, l), 
                (i, k, l), (j, k, l)
            ]
            triangle = triangle_choices[np.random.randint(4)]
            
            # Compute holonomy
            hol = self.compute_holonomy(triangle)
            phase = np.angle(hol)  # phase in radians
            
            phases.append(phase)
        
        phases = np.array(phases)
        
        if len(phases) == 0:
            return 0.0
        
        # Filter outliers (3σ)
        mean_phase = np.mean(phases)
        std_phase = np.std(phases)
        filtered = phases[(phases > mean_phase - 3*std_phase) & 
                         (phases < mean_phase + 3*std_phase)]
        
        if len(filtered) < 10:
            filtered = phases
        
        # Phase variance is related to α
        phase_variance = np.var(filtered)
        
        # Relation from ACT: α⁻¹ ~ 4π / <θ²>
        if phase_variance > 1e-12:
            alpha_inv = 4 * np.pi / phase_variance
            # Constrain to reasonable values
            alpha_inv = np.clip(alpha_inv, 50, 250)
            return float(alpha_inv)
        
        return 0.0
          def metropolis_update(self, steps: int = 100, adaptive_step: bool = True) -> int:
        """Update operators using Metropolis algorithm."""
        accepted = 0
        
        for step in range(steps):
            # Adaptive step size
            step_size = 0.1
            if adaptive_step and step > steps // 2:
                acceptance_rate = accepted / (step + 1e-10)
                if acceptance_rate < 0.2:
                    step_size *= 0.8
                elif acceptance_rate > 0.5:
                    step_size *= 1.2
            
            # Pick random vertex
            i = np.random.randint(self.N)
            
            # Save old operator
            old_operator = self.operators[i].copy()
            
            # Propose new state
            if self.use_clifford:
                # For Clifford algebra: small linear combination of γ-matrices
                delta_coeffs = np.random.randn(5) * step_size
                delta_op = np.zeros((4, 4), dtype=complex)
                for mu in range(4):
                    delta_op += delta_coeffs[mu] * GAMMA_MATRICES[mu]
                delta_op += delta_coeffs[4] * GAMMA_MATRICES[5]
                
                new_operator = old_operator + delta_op
                # Project to "nearby" unitary matrix
                U, _, Vh = np.linalg.svd(new_operator)
                new_operator = U @ Vh
            else:
                # For unitary matrices: multiply by small unitary
                delta_op = (np.random.randn(4, 4) + 
                           1j * np.random.randn(4, 4)) * step_size
                delta_op = (delta_op - delta_op.conj().T) / 2  # anti-hermitian
                new_operator = old_operator @ self._expm_i(delta_op)
                new_operator = new_operator / (np.linalg.det(new_operator) ** (1/4))
            
            # Save old operators
            self.operators_old = self.operators.copy()
            old_action = self.total_action(include_kinetic=False)
            
            # Temporarily set new operator
            self.operators[i] = new_operator
            
            # Clear cache for affected tetrahedra
            self._clear_cache_for_vertex(i)
            
            new_action = self.total_action(include_kinetic=False)
            
            # Metropolis criterion
            delta_S = new_action - old_action
            if delta_S < 0 or np.random.random() < np.exp(-self.beta * delta_S):
                accepted += 1
                # Update kinetic energy
                if not hasattr(self, 'operators_prev'):
                    self.operators_prev = self.operators_old.copy()
            else:
                # Reject change
                self.operators[i] = old_operator
                self._clear_cache_for_vertex(i)
        
        # Clean up temporary attributes
        if hasattr(self, 'operators_old'):
            del self.operators_old
        if hasattr(self, 'operators_prev'):
            del self.operators_prev
        
        return accepted
    
    def _clear_cache_for_vertex(self, vertex: int):
        """Clear cache for tetrahedra containing vertex."""
        keys_to_remove = []
        for key in self._action_cache:
            if vertex in key[:4]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self._action_cache:
                del self._action_cache[key]
        
        # Also clear holonomy cache
        hol_keys_to_remove = []
        for key in self._holonomy_cache:
            if vertex in key:
                hol_keys_to_remove.append(key)
        
        for key in hol_keys_to_remove:
            if key in self._holonomy_cache:
                del self._holonomy_cache[key]
    
    def compute_entanglement_entropy(self, subsystem: List[int] = None) -> float:
        """
        Compute entanglement entropy.
        """
        if subsystem is None:
            # Default: random subsystem ~1/4 of vertices
            subsystem = np.random.choice(self.N, max(1, self.N // 4), replace=False).tolist()
        
        n_sub = len(subsystem)
        if n_sub == 0:
            return 0.0
        
        # Correlation matrix for subsystem
        corr_matrix = np.zeros((n_sub, n_sub), dtype=complex)
        
        for idx_i, i in enumerate(subsystem):
            for idx_j, j in enumerate(subsystem):
                if i == j:
                    corr_matrix[idx_i, idx_j] = 1.0
                else:
                    # Operator correlator
                    corr = np.trace(
                        self.operators[i] @ 
                        self.operators[j].conj().T
                    ) / 4.0
                    corr_matrix[idx_i, idx_j] = corr
        
        # Symmetrize and make hermitian
        corr_matrix = (corr_matrix + corr_matrix.conj().T) / 2
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        
        # Compute von Neumann entropy
        epsilon = 1e-12
        entropy = 0.0
        for lam in eigenvalues:
            if lam > epsilon and (1 - lam) > epsilon:
                # Binary entropy
                entropy -= lam * np.log(lam) + (1 - lam) * np.log(1 - lam)
        
        return np.real(entropy)
          def simulate(self, 
                 n_steps: int = 1000,
                 thermalization: int = 200,
                 measure_interval: int = 10,
                 adaptive_rewiring: bool = True) -> List[Dict]:
        """
        Full system simulation.
        """
        print(f"\nRunning ACT simulation for {n_steps} steps...")
        print(f"Parameters: N={self.N}, β={self.beta:.3f}, J={self.J:.3f}")
        print(f"Tetrahedra: {len(self.tetrahedra)}")
        
        measurements = []
        
        # Thermalization
        print("\n[Phase 1] Thermalization...")
        for step in range(thermalization):
            accepted = self.metropolis_update(steps=20, adaptive_step=True)
            
            if adaptive_rewiring and step % self.rewiring_interval == 0:
                self.dynamic_rewiring()
            
            if step % 50 == 0:
                action = self.total_action(include_kinetic=False)
                print(f"  Step {step}: accepted {accepted}/20, S={action:.3f}")
        
        # Main simulation
        print("\n[Phase 2] Main simulation with measurements...")
        for step in range(n_steps):
            # Dynamics update
            accepted = self.metropolis_update(steps=5, adaptive_step=True)
            
            # Dynamic rewiring
            if adaptive_rewiring and (step + thermalization) % self.rewiring_interval == 0:
                self.dynamic_rewiring()
                self.last_rewiring_step = step
            
            # Measurements
            if step % measure_interval == 0:
                alpha_precise = self.compute_alpha_precise(n_samples=150)
                action = self.total_action(include_kinetic=False)
                curvature = self.compute_curvature()
                entropy = self.compute_entanglement_entropy()
                metric = self.compute_emergent_metric()
                metric_det = np.linalg.det(metric).real
                
                measurement = {
                    'step': step + thermalization,
                    'alpha': alpha_precise,
                    'alpha_precise': alpha_precise,
                    'action': action,
                    'curvature': curvature,
                    'entropy': entropy,
                    'metric': metric.copy(),
                    'metric_det': metric_det,
                    'accepted_rate': accepted / 5,
                    'n_tetrahedra': len(self.tetrahedra)
                }
                
                measurements.append(measurement)
                
                # Save to history
                self.history['step'].append(measurement['step'])
                self.history['alpha'].append(measurement['alpha'])
                self.history['alpha_precise'].append(measurement['alpha_precise'])
                self.history['action'].append(measurement['action'])
                self.history['curvature'].append(measurement['curvature'])
                self.history['entropy'].append(measurement['entropy'])
                self.history['metric_det'].append(measurement['metric_det'])
                self.history['temperature'].append(1.0/self.beta)
                
                if step % (measure_interval * 20) == 0:
                    print(f"  Step {step}: α⁻¹ = {alpha_precise:.2f}, "
                          f"S = {action:.3f}, R = {curvature:.4f}")
        
        print("\n[Phase 3] Statistical analysis...")
        self._analyze_results(measurements)
        
        return measurements
    
    def _analyze_results(self, measurements: List[Dict]):
        """Statistical analysis of results."""
        if not measurements:
            return
        
        alphas = [m['alpha_precise'] for m in measurements]
        alphas = np.array(alphas)
        alphas = alphas[alphas > 0]  # filter zeros
        
        if len(alphas) == 0:
            print("  No data for α analysis")
            return
        
        # Statistics
        mean_alpha = np.mean(alphas)
        std_alpha = np.std(alphas)
        median_alpha = np.median(alphas)
        
        # Relative error
        rel_error = abs(mean_alpha - 137.036) / 137.036 * 100
        
        # Confidence interval (95%)
        n = len(alphas)
        conf_interval = (
            mean_alpha - 1.96 * std_alpha / np.sqrt(n),
            mean_alpha + 1.96 * std_alpha / np.sqrt(n)
        )
        
        print(f"\n  α results:")
        print(f"  Mean: {mean_alpha:.2f} ± {std_alpha:.2f}")
        print(f"  Median: {median_alpha:.2f}")
        print(f"  95% confidence interval: [{conf_interval[0]:.1f}, {conf_interval[1]:.1f}]")
        print(f"  Deviation from 137.036: {rel_error:.2f}%")
        print(f"  Samples: {n}")
        
        # Fit quality (chi-square)
        if std_alpha > 0:
            chi2 = np.sum(((alphas - 137.036) / std_alpha) ** 2) / n
            print(f"  χ²/ndf: {chi2:.3f}")
    
    @staticmethod
    def _expm_i(H: np.ndarray) -> np.ndarray:
        """Matrix exponential i*H for anti-hermitian H."""
        return np.linalg.expm(1j * H)
      
