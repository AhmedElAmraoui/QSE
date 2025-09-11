from itertools import product
from tqdm import tqdm
from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np

def pauli_multiply(p1, p2):
    """
    Multipliziert zwei Pauli-Strings (als Strings) und gibt das Ergebnis zurück.
    Global phases werden ignoriert.
    """
    result = []
    for a, b in zip(p1, p2):
        if a == 'I':
            result.append(b)
        elif b == 'I':
            result.append(a)
        elif a == b:
            result.append('I')
        else:
            # Pauli multiplication table without phase
            ab = {('X','Y'):'Z', ('Y','X'):'Z',
                  ('Y','Z'):'X', ('Z','Y'):'X',
                  ('Z','X'):'Y', ('X','Z'):'Y'}
            result.append(ab.get((a,b), ab.get((b,a))))
    return ''.join(result)

def pauli_power_basis(pauli_strings, k):
    """
    Gibt alle Pauli-Strings zurück, die durch Produkte von bis zu k Hamiltonian-Terms entstehen.
    """
    if k == 0:
        return {'I' * len(pauli_strings[0])}
    elif k == 1:
        return set(pauli_strings)
    
    previous = pauli_power_basis(pauli_strings, k-1)
    result = set()
    for p_prev, p in product(previous, pauli_strings):
        result.add(pauli_multiply(p_prev, p))
    return result

def unique_paulis_up_to_N(pauli_strings, N):
    """
    Gibt die Menge aller einzigartigen Pauli-Strings von H^0 bis H^N zurück,
    verwendet vorhandene pauli_power_basis()-Funktion.
    """
    all_paulis = set()
    for k in range(N+1):
        paulis_k = pauli_power_basis(pauli_strings, k)
        all_paulis.update(paulis_k)
    return all_paulis

def transform_hamiltonian_pi_hp_pk(H_sparse, P_i, P_k):
    
    num_qubits = H_sparse.num_qubits
    pauli_i = Pauli(P_i)
    pauli_k = Pauli(P_k)
    
    new_paulis = []
    coeffs = []
    
    for pauli_term, coeff in zip(H_sparse.paulis, H_sparse.coeffs):
        term_pauli = Pauli(pauli_term)
        new_pauli = (pauli_i @ term_pauli @ pauli_k)
        # Zwinge Ausgabe auf volle Qubit-Länge
        new_paulis.append(new_pauli[:].to_label())
        coeffs.append(coeff*pow(-1j,new_pauli.phase))
    
    return SparsePauliOp.from_list(list(zip(new_paulis, coeffs)))

def generate_all_transformed_hamiltonians(H_sparse, pauli_strings):
    """
    Für alle Kombinationen von (P_i, P_k) aus pauli_strings wird P_i H P_k berechnet.
    
    Args:
        H_sparse: SparsePauliOp
        pauli_strings: Liste von Pauli-Strings (z.B. ["XII", "IZI", ...])
    
    Returns:
        Dictionary mit Schlüsseln (P_i, P_k) und Werten = transformierter SparsePauliOp
    """
    results = {}
    
    for P_i, P_k in product(pauli_strings, repeat=2):
        pauli_i = Pauli(P_i)
        pauli_k = Pauli(P_k)
        
        new_paulis = []
        coeffs = []
        
        for pauli_term, coeff in zip(H_sparse.paulis, H_sparse.coeffs):
            # Multiplikation: P_i * term * P_k
            new_pauli = (pauli_i @ Pauli(pauli_term) @ pauli_k)
            new_paulis.append(new_pauli[:].to_label())
            coeffs.append(coeff*np.pow(-1j,new_pauli.phase))
        transformed_H = SparsePauliOp.from_list(list(zip(new_paulis, coeffs)))
        results[(P_i, P_k)] = transformed_H
        
    return results

def generate_all_pauli_products(pauli_strings):
    results = {}
    
    for P_i, P_k in product(pauli_strings, repeat=2):
        pauli_i = Pauli(P_i)
        pauli_k = Pauli(P_k)

        # Multiplikation: P_i * P_k
        new_pauli = (pauli_i @ pauli_k)
        new_pauli = new_pauli[:].to_label()
        results[(P_i, P_k)] = H = SparsePauliOp.from_list([(new_pauli, 1.0)])
        
    return results