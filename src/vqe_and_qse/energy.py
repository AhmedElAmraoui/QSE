import numpy as np
from qiskit.quantum_info import Statevector, SparsePauliOp
from .ansatz import build_hva_layers
from .hamiltonians import build_sparse_hamiltonian
from scipy.linalg import eigh


def get_energy(H, pauli_expectations):
    paulis = [str(p) for p in H.paulis]
    coeffs = H.coeffs

    energy = 0
    for pauli, coeff in zip(paulis, coeffs):
        # Suche Erwartungswert in deiner Liste der Gruppen
        found = False
        for group_expectation in pauli_expectations:
            if pauli in group_expectation:
                energy += coeff * group_expectation[pauli]
                found = True
                break
        if not found:
            raise ValueError(f"Erwartungswert für {pauli} nicht gefunden!")
        
    return energy

def compute_energy(params, H_sparse, backend,num_layers=1):
    qc = build_hva_layers(params, backend,num_layers)
    psi = Statevector.from_instruction(qc)
    return np.real(psi.expectation_value(H_sparse))


def compute_exact_ground_energy(H_sparse):
    # Konvertiere SparsePauliOp in vollständige Matrix
    H_dense = H_sparse.to_matrix()
    eigvals = eigh(H_dense, eigvals_only=True)
    return np.min(eigvals)

def compute_error(params, H_sparse, backend, energy_exact,num_layers=1):
    energy_approx = compute_energy(params, H_sparse, backend,num_layers)
    return abs(energy_approx-energy_exact)/abs(energy_exact)


def test_energy_consistency(params, backend, hx, hz, J=-1, num_layers=1, atol=1e-8):
    # Baue Hamiltonian (mit Pauli-Strings und Koeffizienten)
    H_sparse, paulis = build_sparse_hamiltonian(hx=hx, hz=hz, backend=backend, J=J, return_paulis=True)
    
    # Berechne den Zustand
    qc = build_hva_layers(params, backend, num_layers=num_layers)
    psi = Statevector.from_instruction(qc)
    
    # Berechne "direkt" die Energie über gesamtes H
    energy_direct = np.real(psi.expectation_value(H_sparse))
    
    # Berechne Einzelterme (pauli_i * coeff_i)
    expectation_values = []
    for pauli_str in paulis:
        op = SparsePauliOp.from_list([(pauli_str, 1.0)])
        expval = np.real(psi.expectation_value(op))
        expectation_values.append(expval)
    
    # Rekonstruiere Energie aus Einzeltermen
    coeffs = H_sparse.coeffs
    energy_sum = np.dot(expectation_values, coeffs)

    # Ausgabe
    print(f"Energie direkt        : {energy_direct:.10f}")
    print(f"Energie aus Summe     : {energy_sum:.10f}")
    print(f"Relativer Fehler      : {abs(energy_direct - energy_sum)/abs(energy_direct):.2e}")

    # Testbedingung
    assert np.isclose(energy_direct, energy_sum, atol=atol), \
        "Energie aus Erwartungswerten stimmt nicht mit gesamter Hamiltonian-Erwartung überein."
    

def compute_individual_expectation_values(params, backend, hx, hz,J=-1, num_layers=1, inst_map=None):
    # Hamiltonian mit Pauli-Liste holen
    H_sparse, paulis = build_sparse_hamiltonian(hx = hx, hz=hz,backend= backend, J=J,return_paulis=True)
    
    qc = build_hva_layers(params, backend, num_layers=num_layers, inst_map=inst_map)
    psi = Statevector.from_instruction(qc)

    expectation_values = []
    for pauli_str in paulis:
        op = SparsePauliOp.from_list([(pauli_str, 1.0)])
        expval = np.real(psi.expectation_value(op))
        expectation_values.append(expval)

    return expectation_values, paulis