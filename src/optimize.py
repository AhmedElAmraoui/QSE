from .energy import compute_error, get_energy
import numpy as np
from .hamiltonians import build_sparse_hamiltonian
from itertools import product
from tqdm import tqdm
from .initparams import find_best_initial_params, find_best_initial_params_counts
import matplotlib.pyplot as plt
from .ansatz import build_hva_layers
from .measurement import apply_measurement_bases, get_pauli_expectation_dict
from qiskit import transpile
from scipy.optimize import minimize
from .modes import RunConfig, EstimatorMode, ExpectationEngine
from .energy import compute_exact_ground_energy

def optimize_energy(initial_params, H_sparse, backend, energy_exact,num_layers=1):
    result = minimize(
        compute_error,
        initial_params,
        args=(H_sparse, backend, energy_exact,num_layers),
        method='COBYLA',
        options={'maxiter': 200}
    )
    return result.x, result.fun


def optimize_energy_objective(x0, objective, method="COBYLA", maxiter=200):
    """
    Minimiert eine gegebene objective(params)->float.
    Gibt (x_opt, f_opt) zurück.
    """
    res = minimize(objective, x0, method=method, options={'maxiter': maxiter})
    return res.x, float(res.fun)

def run_hva_grid_scan(
    run_cfg: RunConfig,
    hx_range=np.linspace(-2, 2, 20),
    hz_range=np.linspace(-2, 2, 20),
    plot_2d=True,
    plot_3d=False,
    J=-1,
    num_layers=1
):
    engine = ExpectationEngine(run_cfg)

    hx_vals = hx_range
    hz_vals = hz_range
    error_grid = np.zeros((len(hx_vals), len(hz_vals)))
    param_grid = np.zeros((len(hx_vals), len(hz_vals), 3 * num_layers))

    for (i, hx), (j, hz) in tqdm(product(enumerate(hx_vals), enumerate(hz_vals)),
                                 total=len(hx_vals) * len(hz_vals),
                                 desc="HVA Grid Scan"):

        # 1. Hamiltonian erstellen
        H, pauli_strings = build_sparse_hamiltonian(hx, hz, run_cfg.backend, J=J, return_paulis=True)

        # 2. Exakte Energie nur wenn möglich
        if run_cfg.mode == EstimatorMode.STATEVECTOR or H.num_qubits <= 12:
            E_exact = compute_exact_ground_energy(H)
        else:
            E_exact = None

        # 3. Initialparameter bestimmen
        if run_cfg.mode == EstimatorMode.STATEVECTOR:
            x0, _ = find_best_initial_params(H, run_cfg.backend, num_layers=num_layers)
        else:
            x0, _ = find_best_initial_params_counts(H, pauli_strings, run_cfg.backend, num_layers=num_layers)

        # 4. Cost-Funktion (immer nur Energie)
        def cost(params):
            circ = build_hva_layers(params, backend=run_cfg.backend, num_layers=num_layers)
            return engine.energy(H, circ).real

        # 5. Optimierung
        x_opt, E_var = optimize_energy_objective(x0, cost, method="COBYLA", maxiter=100)

        # 6. Relativer Fehler (nur falls exact bekannt)
        if E_exact is not None:
            rel_err = abs(E_var - E_exact) / abs(E_exact)
        else:
            rel_err = np.nan

        # 7. Ergebnisse speichern
        error_grid[i, j] = rel_err
        param_grid[i, j, :] = x_opt

    # Mesh erstellen für Plot
    hx_mesh, hz_mesh = np.meshgrid(hz_vals, hx_vals)

    if plot_2d:
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(hz_mesh, hx_mesh, error_grid, levels=50, cmap='viridis')
        plt.colorbar(cp, label=r'Relativer Fehler $|E_{\mathrm{var}} - E_{\mathrm{exact}}| / |E_{\mathrm{exact}}|$')
        plt.xlabel(r'$h_z$')
        plt.ylabel(r'$h_x$')
        plt.title(f'HVA-Fehler bei {num_layers} Layer(n), J={J}')
        plt.show()

    if plot_3d:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(hx_mesh, hz_mesh, error_grid,
                               cmap='viridis', edgecolor='k', linewidth=0.3, antialiased=True)
        ax.set_xlabel(r'$h_x$')
        ax.set_ylabel(r'$h_z$')
        ax.set_zlabel(r'Relativer Fehler')
        ax.set_title(f'HVA Fehleroberfläche ({num_layers} Layer, J={J})')
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Fehler')
        plt.show()

    return error_grid, param_grid

def vqe_cost(params,
             backend, 
             simulator,
             num_layers,
             meas_bases,
             groups,
             Hamiltonian,
             exact_energy):
    
    circuit = build_hva_layers(params, backend, num_layers=num_layers)
    circuits = apply_measurement_bases(circuit, meas_bases)
    transpiled = transpile(circuits, backend=backend, optimization_level=3)
    
    counts = []
    for circ in transpiled:
        result = simulator.run(circ, shots=16*1024).result()
        counts.append(result.get_counts())
    
    # Erwartungswerte berechnen
    pauli_expectations = get_pauli_expectation_dict(groups, counts)
    energy = get_energy(Hamiltonian, pauli_expectations)
    rel_error = abs(exact_energy - energy) / abs(exact_energy)
    return rel_error