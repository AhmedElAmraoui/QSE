from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp


def get_backend_info(backend, inst_map=None):
    if isinstance(backend, GenericBackendV2):
        num_qubits = backend.num_qubits
        coupling_map = backend.coupling_map
    else:
        num_qubits = backend.configuration().num_qubits
        coupling_map = CouplingMap(backend.configuration().coupling_map)

    # Falls inst_map angegeben ist, filtere die Qubits und die Kopplungen entsprechend
    if inst_map is not None:
        # Filtere die Kopplungen: nur Bonds, bei denen beide Qubits in inst_map sind
        if isinstance(coupling_map, CouplingMap):
            filtered_coupling = [
                (i, j) for (i, j) in coupling_map.get_edges()
                if i in inst_map and j in inst_map
            ]
        else:
            filtered_coupling = [
                (i, j) for (i, j) in coupling_map
                if i in inst_map and j in inst_map
            ]
        # Setze die Qubit-Anzahl auf die Länge von inst_map
        num_qubits = len(inst_map)
        coupling_map = filtered_coupling

    return num_qubits, coupling_map

def get_unique_bonds(coupling_map):
    bond_set = set()
    for i, j in coupling_map:
        bond = tuple(sorted([i, j]))
        bond_set.add(bond)
    return list(bond_set)

def build_sparse_hamiltonian(hx, hz, backend, J=-1, return_paulis=False):
    num_qubits, coupling_map = get_backend_info(backend)
    bonds = get_unique_bonds(coupling_map)
    paulis = []
    coeffs = []

    # ZZ-Terme
    for i, j in bonds:
        z_str = ['I'] * num_qubits
        z_str[i] = 'Z'
        z_str[j] = 'Z'
        pauli = ''.join(reversed(z_str))
        paulis.append(pauli)
        coeffs.append(J)

    # X- und Z-Feld-Terme
    for i in range(num_qubits):
        x_str = ['I'] * num_qubits
        x_str[i] = 'X'
        pauli_x = ''.join(reversed(x_str))
        paulis.append(pauli_x)
        coeffs.append(hx)

        z_str = ['I'] * num_qubits
        z_str[i] = 'Z'
        pauli_z = ''.join(reversed(z_str))
        paulis.append(pauli_z)
        coeffs.append(hz)

    hamiltonian = SparsePauliOp.from_list(list(zip(paulis, coeffs)))

    if return_paulis:
        return hamiltonian, paulis
    else:
        return hamiltonian