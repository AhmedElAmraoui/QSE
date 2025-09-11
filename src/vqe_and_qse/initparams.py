from itertools import product
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.linalg import fractional_matrix_power
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
from .energy import get_energy, compute_exact_ground_energy, compute_error
from .grouping import group_paulis, determine_measurement_basis
from .measurement import apply_measurement_bases, get_pauli_expectation_dict
import numpy as np
from .ansatz import build_hva_layers

def find_best_initial_params(H_sparse, backend, scan_points=5, angle_range=(-np.pi, np.pi),num_layers=1):
    best_error = np.inf
    best_params = None
    
    energy_exact = compute_exact_ground_energy(H_sparse)
    
    alpha_vals = np.linspace(*angle_range, scan_points)
    beta_vals = np.linspace(*angle_range, scan_points)
    gamma_vals = np.linspace(*angle_range, scan_points)


    param_grid = product(alpha_vals, beta_vals, gamma_vals, repeat=num_layers)
    for param_tuple in param_grid:
        params = list(param_tuple)
        error = compute_error(params, H_sparse, backend, energy_exact, num_layers=num_layers)
        if error < best_error:
            best_error = error
            best_params = params

    return np.array(best_params), energy_exact

def find_best_initial_params_counts(H_sparse, pauli_strings, backend, scan_points=5, angle_range=(-np.pi, np.pi), num_layers=1):
    best_error = np.inf
    best_params = None
    
    # Exakte Energie einmal berechnen
    energy_exact = compute_exact_ground_energy(H_sparse)
    
    alpha_vals = np.linspace(*angle_range, scan_points)
    beta_vals = np.linspace(*angle_range, scan_points)
    gamma_vals = np.linspace(*angle_range, scan_points)
    
    param_grid = product(alpha_vals, beta_vals, gamma_vals, repeat=num_layers)

    groups = group_paulis(pauli_strings)
    measurement_bases = [determine_measurement_basis(group) for group in groups]
    
    simulator = AerSimulator.from_backend(backend)

    for param_tuple in param_grid:
        params = list(param_tuple)
        
        # Circuit bauen + Measurement Bases anwenden
        circuit = build_hva_layers(params, backend, num_layers=num_layers)
        circuits = apply_measurement_bases(circuit, measurement_bases)
        
        # Transpilieren
        transpiled_circuits = transpile(circuits, backend=backend, optimization_level=3)
        
        # Simulieren
        counts = []
        for qc in transpiled_circuits:
            result = simulator.run(qc, shots=1024).result()
            counts.append(result.get_counts())
        
        # Energie berechnen
        pauli_expectations = get_pauli_expectation_dict(groups, counts)
        energy = get_energy(H_sparse, pauli_expectations)
        
        # Fehler relativ zur exakten Energie
        error = abs(energy - energy_exact) / abs(energy_exact)

        if error < best_error:
            best_error = error
            best_params = params
    
    return np.array(best_params), energy_exact