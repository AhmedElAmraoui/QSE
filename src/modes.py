# src/qexp/modes.py
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

from .grouping import group_paulis, determine_measurement_basis
from .measurement import apply_measurement_bases, get_pauli_expectation_dict
from .energy import get_energy
from typing import Any

class EstimatorMode(Enum):
    STATEVECTOR = auto()
    QASM_IDEAL = auto()
    QASM_FAKE = auto()

@dataclass
class RunConfig:
    mode: EstimatorMode
    backend: Any              # oder: backend: object
    shots: int = 8192
    opt_level: int = 3
    seed: int | None = None

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

    def expectations_for_observable(self, Obs: SparsePauliOp, circuit):
        """
        Wie energy(), aber für ein beliebiges SparsePauliOp.
        """
        if self.cfg.mode == EstimatorMode.STATEVECTOR:
            psi = Statevector.from_instruction(circuit)
            return float(np.real(psi.expectation_value(Obs)))

        pauli_strings = [p.to_label() for p in Obs.paulis]
        groups = group_paulis(pauli_strings)
        meas_bases = [determine_measurement_basis(g) for g in groups]

        circuits = apply_measurement_bases(circuit, meas_bases)
        transpiled = transpile(circuits, backend=self.cfg.backend, optimization_level=self.cfg.opt_level)

        counts_all = []
        for qc in transpiled:
            res = self.sim.run(qc, shots=self.cfg.shots).result()
            counts_all.append(res.get_counts())

        pauli_expectations = get_pauli_expectation_dict(groups, counts_all)
        return float(np.real(get_energy(Obs, pauli_expectations)))
