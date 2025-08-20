from qiskit.quantum_info import Pauli
import networkx as nx


def commute(pauli1, pauli2):
    """Prüft ob zwei Pauli-Strings kommutieren."""
    p1 = Pauli(pauli1)
    p2 = Pauli(pauli2)
    return p1.commutes(p2)

def locally_commutable(p1, p2):
    """
    Prüft ob zwei Pauli-Strings lokal messbar sind: 
    auf keinem Qubit gibt es gleichzeitig z.B. X und Y.
    """
    for a, b in zip(p1, p2):
        if a == 'I' or b == 'I':
            continue
        if a != b:
            return False  # unterschiedliche Nicht-I-Terms → nicht lokal gleichzeitig messbar
    return True

def group_paulis(pauli_strings):
    """
    Gruppiert Pauli-Strings in Gruppen mit gemeinsamer Messbasis (lokal messbar).
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(pauli_strings)))

    # Kante falls sie NICHT gemeinsam messbar sind (Konfliktgraph)
    for i in range(len(pauli_strings)):
        for j in range(i + 1, len(pauli_strings)):
            if not locally_commutable(pauli_strings[i], pauli_strings[j]):
                G.add_edge(i, j)

    # Graph Coloring
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")

    # Gruppieren nach Farben
    groups = {}
    for idx, color in coloring.items():
        groups.setdefault(color, []).append(pauli_strings[idx])

    return list(groups.values())

def determine_measurement_basis(pauli_group):
    """
    Bestimmt eine Messbasis (als Liste von 'X', 'Y', 'Z', 'I') für eine Gruppe kommutierender Pauli-Strings.
    Die zurückgegebene Basis diagonalisiert alle Paulis in der Gruppe gleichzeitig.
    """
    num_qubits = len(pauli_group[0])
    basis = ['Z'] * num_qubits  # Default: Messung in Z

    for qubit in range(num_qubits):
        #paulis_on_qubit = set(p[qubit] for p in pauli_group if p[qubit] != 'I')
        paulis_on_qubit = set(p[num_qubits-1-qubit] for p in pauli_group if p[num_qubits-1-qubit] != 'I')
        
        if not paulis_on_qubit:
            basis[qubit] = 'I'
        elif paulis_on_qubit == {'Z'}:
            basis[qubit] = 'Z'
        elif paulis_on_qubit == {'X'}:
            basis[qubit] = 'X'
        elif paulis_on_qubit == {'Y'}:
            basis[qubit] = 'Y'
        else:
            # Mehrere unterschiedliche Paulis (X,Y,Z) → nicht gemeinsam diagonal.
            # In Gruppenbildung sollte das nicht vorkommen.
            raise ValueError(f"Nicht kompatible Paulis auf Qubit {qubit}: {paulis_on_qubit}")

    return basis