from qiskit.quantum_info import Pauli

def pauli_mul_with_phase(label_a: str, label_b: str) -> tuple[str, complex]:
    Pa, Pb = Pauli(label_a), Pauli(label_b)
    Pc = (Pa @ Pb)
    return Pc.to_label(), (-1j)**(Pc.phase)

from collections import defaultdict

def op_convolve(A: dict[str, complex], B: dict[str, complex]) -> dict[str, complex]:
    acc = defaultdict(complex)
    for P, a in A.items():
        for Q, b in B.items():
            R, phase = pauli_mul_with_phase(P, Q)
            acc[R] += a * b * phase
    # kleine Koeffizienten kappen
    return {L: c for L, c in acc.items() if abs(c) > 1e-12}

def build_powers(H1: dict[str, complex], K: int) -> list[dict[str, complex]]:
    powers = []
    n_qubits = len(next(iter(H1.keys())))
    H0 = {"I"*n_qubits: 1.0+0j}
    powers.append(H0)       # H^0
    powers.append(H1)       # H^1
    for m in range(1, 2*K+1):      # baut H^2 ... H^(2K+1)
        powers.append(op_convolve(powers[m], H1))
    return powers  # index m ↦ H^m

def collect_all_paulis(powers: list[dict[str, complex]]) -> list[str]:
    """
    Sammelt alle Pauli-Strings aus einer Liste von Operator-Potenzen.
    """
    P_all = set()
    for Hm in powers:
        P_all.update(Hm.keys())
    return sorted(P_all)


def merge_expectations(pauli_expectations: list[dict[str, float]]) -> dict[str, float]:
    """
    Fasst die Erwartungswerte aller Gruppen zu einem Dictionary zusammen.
    """
    merged = {}
    for d in pauli_expectations:
        merged.update(d)   # kein Konflikt, jeder Pauli sollte nur in einer Gruppe vorkommen
    return merged


def evaluate_powers_expectation(
    powers: list[dict[str, complex]], 
    pauli_expectations: list[dict[str, float]]
) -> list[complex]:
    """
    Erwartungswert für jedes Power H^m bestimmen.
    """
    merged = merge_expectations(pauli_expectations)
    expvals = []
    for Hm in powers:
        val = 0.0 + 0j
        for P, coeff in Hm.items():
            exp_val_P = merged.get(P, 0.0)  # falls Pauli nicht gemessen wurde
            val += coeff * exp_val_P
        expvals.append(val)
    return expvals