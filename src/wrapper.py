from dataclasses import dataclass
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeQuitoV2
from scipy.linalg import fractional_matrix_power

from .modes import EstimatorMode, RunConfig, ExpectationEngine, SubspaceMode
from .hamiltonians import build_sparse_hamiltonian, dict_to_sparsepauliop
from .ansatz import build_hva_layers
from .initparams import find_best_initial_params, find_best_initial_params_counts
from .energy import compute_exact_ground_energy
from .optimize import optimize_energy_objective

@dataclass
class ExperimentResult:
    params: np.ndarray
    vqe_energy: float
    subspace_energy: float | None
    rel_error_vqe: float | None
    rel_error_subspace: float | None
    improvement_abs: float | None
    improvement_vs_vqe_pct: float | None
    improvement_vs_exact_pct: float | None
    mode: EstimatorMode
    subspace_mode: SubspaceMode
    meta: dict

def run_experiment(hx: float, hz: float, J: float = -1,
                   num_layers: int = 1,
                   mode: str = "statevector",
                   subspace: str | bool = False,
                   shots: int = 8192,
                   N_subspace: int = 1,
                   backend= FakeQuitoV2()):

    # --- Mode -> RunConfig ---
    # Mode-Mapping
    if mode == "statevector":
        est_mode = EstimatorMode.STATEVECTOR
    elif mode == "qasm_ideal":
        est_mode = EstimatorMode.QASM_IDEAL
    elif mode == "qasm_fake":
        est_mode = EstimatorMode.QASM_FAKE
    else:
        raise ValueError("mode must be 'statevector', 'qasm_ideal' or 'qasm_fake'")

    # Subspace-Mapping
    if subspace is True:
        subspace_mode = SubspaceMode.COARSE
    elif subspace is False or subspace == "none":
        subspace_mode = SubspaceMode.NONE
    elif subspace == "fine":
        subspace_mode = SubspaceMode.FINE
    elif subspace == "coarse":
        subspace_mode = SubspaceMode.COARSE
    else:
        raise ValueError("subspace must be False/True or 'none'/'coarse'/'fine'")

    run_cfg = RunConfig(
        mode=est_mode,
        backend=backend,
        shots=shots,
        subspace_mode=subspace_mode,
        N_subspace=N_subspace
    )
    engine = ExpectationEngine(run_cfg)

    # --- Hamiltonian ---
    H, pauli_strings = build_sparse_hamiltonian(hx=hx, hz=hz, backend=run_cfg.backend, J=J, return_paulis=True)

    # Exakte Energie, wenn sinnvoll (für relative Fehler-Ausgabe)
    num_qubits = H.num_qubits
    if mode == "statevector" or num_qubits <= 12:
        E_exact = compute_exact_ground_energy(H)
    else:
        E_exact = None

    # --- Initial-Guess ---
    if run_cfg.mode == EstimatorMode.STATEVECTOR:
        x0, _ = find_best_initial_params(H, run_cfg.backend, num_layers=num_layers)
    else:
        x0, _ = find_best_initial_params_counts(H, pauli_strings, run_cfg.backend, num_layers=num_layers)

    # --- Cost: immer Energie minimieren ---
    def cost(params):
        circ = build_hva_layers(params, backend=run_cfg.backend, num_layers=num_layers)
        return float(engine.energy(H, circ))

    # --- Optimierung (VQE) ---
    x_opt, E_vqe = optimize_energy_objective(x0, cost, method="COBYLA", maxiter=100)
    circ_opt = build_hva_layers(x_opt, backend=run_cfg.backend, num_layers=num_layers)
    E_vqe = float(engine.energy(H, circ_opt))

    # Relativer Fehler VQE (falls E_exact da)
    rel_err_vqe = (abs(E_vqe - E_exact) / abs(E_exact)) if E_exact is not None else None

    # --- Ohne Subspace: direkt zurück ---
    if not subspace:
        return ExperimentResult(
            params=np.asarray(x_opt),
            vqe_energy=E_vqe,
            subspace_energy=None,
            rel_error_vqe=rel_err_vqe,
            rel_error_subspace=None,
            improvement_abs=None,
            improvement_vs_vqe_pct=None,
            improvement_vs_exact_pct=None,
            mode=run_cfg.mode,
            subspace_mode=run_cfg.subspace_mode,
            meta={"shots": shots, "N_subspace": None, "num_qubits": num_qubits}
        )

    # --- Subspace-QSE ---
    H_dict = build_sparse_hamiltonian(hx=hx, hz=hz, backend=run_cfg.backend, J=J, as_dict=True)
    H_red, S_red = engine.expectations_for_observable(H_dict, circ_opt)
    
    # Hermitisieren & numerisch stabilisieren
    H_red = 0.5 * (H_red + H_red.conj().T)
    H_red = np.real_if_close(H_red)
    S_red = 0.5 * (S_red + S_red.conj().T)
    S_red = np.real_if_close(S_red)

    # Generalized EVP mit Trace-Regularisierung
    w, U = np.linalg.eigh(S_red)

    # Negative Eigenwerte auf 0 setzen
    w_reg = np.clip(w, 0, None)

    # Zielspur = Dimension des Subspace
    trace_target = S_red.shape[0]

    if np.sum(w_reg) > 0:
        scale = trace_target / np.sum(w_reg)
        w_reg *= scale
    else:
        # Falls alles weggeschnitten → Notfall: Einheitsmatrix
        w_reg = np.ones_like(w)

    # Reguliertes S rekonstruieren
    S_reg = (U * w_reg) @ U.conj().T

    # Reduziertes Generalized EVP
    wS, US = np.linalg.eigh(S_reg)
    mask = wS > 1e-12
    rank = int(mask.sum())
    V = US[:, mask]

    S_proj = V.T.conj() @ S_reg @ V
    H_proj = V.T.conj() @ H_red @ V

    S_inv_sqrt = fractional_matrix_power(S_proj, -0.5)
    H_eff = S_inv_sqrt @ H_proj @ S_inv_sqrt
    evals, _ = np.linalg.eigh(H_eff)
    E_sub = float(np.min(evals))

    # Relativer Fehler Subspace (falls E_exact da)
    rel_err_sub = (abs(E_sub - E_exact) / abs(E_exact)) if E_exact is not None else None

    # Verbesserungsmetriken
    improvement_abs = E_vqe - E_sub                    # >0 heißt: Subspace niedriger → besser
    improvement_vs_vqe_pct = (improvement_abs / abs(E_vqe) * 100.0) if E_vqe != 0 else None
    improvement_vs_exact_pct = None
    if rel_err_vqe is not None and rel_err_sub is not None and rel_err_vqe != 0:
        improvement_vs_exact_pct = ( (rel_err_vqe - rel_err_sub) / rel_err_vqe ) * 100.0

    return ExperimentResult(
        params=np.asarray(x_opt),
        vqe_energy=E_vqe,
        subspace_energy=E_sub,
        rel_error_vqe=rel_err_vqe,
        rel_error_subspace=rel_err_sub,
        improvement_abs=improvement_abs,
        improvement_vs_vqe_pct=improvement_vs_vqe_pct,
        improvement_vs_exact_pct=improvement_vs_exact_pct,
        mode=run_cfg.mode,
        subspace_mode=run_cfg.subspace_mode,
        meta={"shots": shots, "N_subspace": N_subspace, "rank": rank, "num_qubits": num_qubits}
    )

