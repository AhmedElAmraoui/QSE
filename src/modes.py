# src/qexp/modes.py
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

from .grouping import group_paulis, determine_measurement_basis, strip_phase
from .measurement import apply_measurement_bases, get_pauli_expectation_dict
from .energy import get_energy
from typing import Any

from .krylov_subspace import evaluate_powers_expectation, build_powers, collect_all_paulis, pauli_mul_with_phase, merge_expectations
from .grouping import group_paulis, determine_measurement_bases
from .measurement import apply_measurement_bases, get_pauli_expectation_dict
from .hamiltonians import dict_to_sparsepauliop

class EstimatorMode(Enum):
    STATEVECTOR = auto()
    QASM_IDEAL = auto()
    QASM_FAKE = auto()

class SubspaceMode(Enum):
    NONE = "none"
    COARSE = "coarse"
    FINE = "fine"

@dataclass
class RunConfig:
    mode: EstimatorMode
    backend: Any
    shots: int = 8192
    opt_level: int = 3
    seed: int | None = None
    subspace_mode: SubspaceMode = SubspaceMode.NONE
    N_subspace: int = 0

class ExpectationEngine:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        if cfg.mode == EstimatorMode.STATEVECTOR:
            self.sim = None
        elif cfg.mode == EstimatorMode.QASM_IDEAL:
            self.sim = AerSimulator(seed_simulator=cfg.seed)
        elif cfg.mode == EstimatorMode.QASM_FAKE:
            # Noise-inklusive Simulator, kalibriert auf Fake-Backend
            self.sim = AerSimulator.from_backend(cfg.backend, seed_simulator=cfg.seed)
        else:
            raise ValueError("Unknown mode")

    def energy(self, H: SparsePauliOp, circuit, *, reuse_grouping=True):
        """
        Berechnet <H> für den gegebenen Circuit, je nach Mode über
        Statevector ODER Mess-Gruppen + Counts.
        """
        if self.cfg.mode == EstimatorMode.STATEVECTOR:
            psi = Statevector.from_instruction(circuit)
            return float(np.real(psi.expectation_value(H)))

        # Measurement-Pfad (QASM_IDEAL/FAKE)
        pauli_strings = [p.to_label() for p in H.paulis]
        groups = group_paulis(pauli_strings)
        meas_bases = [determine_measurement_basis(g) for g in groups]

        circuits = apply_measurement_bases(circuit, meas_bases)
        transpiled = transpile(circuits, backend=self.cfg.backend, optimization_level=self.cfg.opt_level)

        counts_all = []
        for qc in transpiled:
            res = self.sim.run(qc, shots=self.cfg.shots).result()
            counts_all.append(res.get_counts())

        pauli_expectations = get_pauli_expectation_dict(groups, counts_all)
        return float(np.real(get_energy(H, pauli_expectations)))

    def expectations_for_observable(self, H_dict: dict, circuit):
        """
        Erwartungswert eines Observablen (SparsePauliOp) für einen gegebenen Circuit.
        """
        
        if self.cfg.subspace_mode == SubspaceMode.COARSE:
            powers = build_powers(H_dict, self.cfg.N_subspace)
            # --- Statevector-Modus: direkt berechnen ---
            if self.cfg.mode == EstimatorMode.STATEVECTOR:
                expvals = []
                psi = Statevector.from_instruction(circuit)
                for p in powers:
                    Obs = dict_to_sparsepauliop(p)
                    expvals.append(psi.expectation_value(Obs))

            # --- Samplingsmodus: Messung über gruppierte Paulis ---
            # Baue die Subspace-Potenzen und sammle alle benötigten Paulis
            else:
                pauli_strings = [strip_phase(p) for p in collect_all_paulis(powers)]
                groups = group_paulis(pauli_strings)

                # Bestimme Messbasen für jede Gruppe
                measurement_bases = determine_measurement_bases(groups)

                # Erzeuge Messcircuits für alle Gruppen
                circuits = apply_measurement_bases(circuit, measurement_bases)
                transpiled = transpile(
                    circuits,
                    backend=self.cfg.backend,
                    optimization_level=self.cfg.opt_level
                )

                # Führe alle Messungen aus
                counts_all = []
                for qc in transpiled:
                    res = self.sim.run(qc, shots=self.cfg.shots).result()
                    counts_all.append(res.get_counts())

                # Erwartungswerte der einzelnen Paulis bestimmen
                pauli_expectations = get_pauli_expectation_dict(groups, counts_all)
                expvals = evaluate_powers_expectation(powers, pauli_expectations)

            # Erwartungswert für Obs aus den Pauli-Erwartungswerten zusammensetzen
            n = self.cfg.N_subspace
            H_red = np.zeros((n+1, n+1), dtype=complex)
            S_red = np.zeros((n+1, n+1), dtype=complex)
    
            for i in range(n+1):
                for j in range(n+1):
                    H_red[i, j] = expvals[i+j+1]
                    S_red[i, j] = expvals[i+j]
            
            return H_red, S_red
        elif self.cfg.subspace_mode == SubspaceMode.FINE:
            powers = build_powers(H_dict, self.cfg.N_subspace)

            # sammle alle Pauli-Erwartungswerte (einmal global)
            pauli_strings = [strip_phase(p) for p in collect_all_paulis(powers)]
            groups = group_paulis(pauli_strings)
            measurement_bases = determine_measurement_bases(groups)
            circuits = apply_measurement_bases(circuit, measurement_bases)
            transpiled = transpile(circuits, backend=self.cfg.backend, optimization_level=self.cfg.opt_level)

            counts_all = [self.sim.run(qc, shots=self.cfg.shots).result().get_counts()
                        for qc in transpiled]
            pauli_expectations = merge_expectations(get_pauli_expectation_dict(groups, counts_all))

            # Indexiere Basisvektoren durch einzelne Paulis
            basis = []
            for k in range(self.cfg.N_subspace+1):
                for P in powers[k].keys():
                    basis.append((k, strip_phase(P)))   # (Moment, Pauli)

            dim = len(basis)
            H_red = np.zeros((dim, dim), dtype=complex)
            S_red = np.zeros((dim, dim), dtype=complex)

            # Matrixelemente berechnen
            for a, (ka, Pa) in enumerate(basis):
                for b, (kb, Pb) in enumerate(basis):
                    Pab, phase = pauli_mul_with_phase(Pa, Pb)
                    S_red[a, b] = phase * pauli_expectations.get(Pab, 0.0)

                    val = 0.0 + 0j
                    for Q, c in H_dict.items():
                        PQ, phase2 = pauli_mul_with_phase(Pa, Q)
                        PQPb, phase3 = pauli_mul_with_phase(PQ, Pb)
                        val += c * phase2 * phase3 * pauli_expectations.get(PQPb, 0.0)
                    H_red[a, b] = val
            return H_red, S_red
