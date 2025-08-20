from .hamiltonians import get_backend_info, get_unique_bonds
from qiskit import QuantumCircuit

def build_hva_layers(params, backend, num_layers=1, inst_map=None):
    assert len(params) == 3 * num_layers, "Parameteranzahl stimmt nicht mit Anzahl der Layer überein!"
    
    num_qubits, coupling_map = get_backend_info(backend, inst_map=inst_map)
    bonds = get_unique_bonds(coupling_map)
    qc = QuantumCircuit(num_qubits)

    # Initialzustand: |+>^N durch H auf allen Qubits
    qc.h(range(num_qubits))

    # Schleife über die Layer
    for layer in range(num_layers):
        alpha = params[3 * layer]
        beta = params[3 * layer + 1]
        gamma = params[3 * layer + 2]

        # ZZ-Terme (nur für gekoppelte Qubits)
        for i, j in bonds:
            qc.cx(i, j)
            qc.rz(alpha, j)
            qc.cx(i, j)

        # Z-Feld (longitudinal)
        for i in range(num_qubits):
            qc.rz(beta, i)

        # X-Feld (transversal)
        for i in range(num_qubits):
            qc.rx(gamma, i)

    return qc