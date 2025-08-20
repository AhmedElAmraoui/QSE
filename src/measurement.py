from qiskit.quantum_info import Pauli

def apply_measurement_bases(base_circuit, measurement_bases):
    """
    Für jede Messbasis wird eine Kopie des Circuits erstellt, 
    mit Umwandlung der Messbasis vor der Messung.
    
    Args:
        base_circuit: QuantumCircuit ohne Messung.
        measurement_bases: Liste von Listen, z.B. [['X', 'Z', 'I'], ['Y', 'Z', 'I']]
    
    Returns:
        List[QuantumCircuit]: jeweils mit Messungen in der gegebenen Basis.
    """
    circuits=[]

    for basis in measurement_bases:
        qc = base_circuit.copy()

        for qubit, b in enumerate(basis):
            if b == 'X':
                qc.h(qubit)
            elif b == 'Y':
                qc.sdg(qubit)
                qc.h(qubit)
            elif b == 'Z':
                pass  # keine Änderung
            elif b == 'I':
                pass  # keine Änderung oder optional Messung weglassen
            else:
                raise ValueError(f"Ungültige Messbasis: {b} für Qubit {qubit}")

        qc.measure_all()
        circuits.append(qc)

    return circuits

def get_expectation(pauli, counts):
    """
    Erwartungswert eines Pauli-Strings aus Z-Basis-Counts nach passender Basis-Rotation.
    Annahme: X/Y wurden vor der Messung nach Z rotiert (H bzw. Sdg+H).

    Qiskit-Konvention:
    - Pauli-Label: links = Qubit n-1, rechts = Qubit 0
    - Bitstring:   links = Qubit n-1, rechts = Qubit 0
    """
    label = pauli.to_label() if hasattr(pauli, "to_label") else str(pauli)
    shots = sum(counts.values())
    if shots == 0:
        return 0.0

    exp = 0.0
    for bitstring, cnt in counts.items():
        parity = 0
        for pos, p in enumerate(label):         # pos=0 ↔ qubit n-1 … pos=n-1 ↔ qubit 0
            if p != 'I':
                parity ^= (bitstring[pos] == '1')
        exp += (1.0 if parity == 0 else -1.0) * cnt

    return exp / shots

def get_pauli_expectation_dict(groups, counts):
    pauli_expectations = []

    for idx, group in enumerate(groups):
        group_expectation = {}
        for pauli_str in group:
            pauli = Pauli(pauli_str)
            exp_val = get_expectation(pauli=pauli, counts=counts[idx])
            group_expectation[pauli_str] = exp_val
        pauli_expectations.append(group_expectation)
        
    return pauli_expectations