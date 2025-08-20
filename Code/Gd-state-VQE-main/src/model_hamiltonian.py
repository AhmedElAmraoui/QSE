# Essential imports 
from qiskit.quantum_info import SparsePauliOp 


# def model_input( J , hx , hz ):
#     """
#     Function gets the input for the problem hamiltonian: Mixed Ising field model for HVA
#     """
#     J = J
#     hx = hx 
#     hz = hz
#     return J, hx, hz

def ham_str_creation(num_qubits = 5,ham_pauli = "Z", bonds =[], num = 2):
    paulis_str = []
    s = "I"*(num_qubits - num) 
    if num == 2:
        for pauli in [ham_pauli]:
            for bond in bonds: 
                list_s = list(s)
                list_s.insert(bond[0], pauli)
                list_s.insert(bond[1], pauli)
                paulis_str.append(''.join(list_s))
    elif num == 1:
        for pauli in [ham_pauli]:
            for i in range(num_qubits):
                list_s = list(s)
                list_s.insert(i, pauli)
                paulis_str.append(''.join(list_s))
    return paulis_str


def ham_term_strings(num_qubits = 5, bonds = []):
    paulis_ZZ = ham_str_creation(num_qubits= num_qubits,ham_pauli = "Z", bonds =bonds,num = 2)
    paulis_Z = ham_str_creation(num_qubits= num_qubits, ham_pauli = "Z", bonds =bonds,num = 1)
    paulis_X = ham_str_creation(num_qubits= num_qubits, ham_pauli = "X", bonds = bonds,num = 1)
    ham_ZZ = ["".join(reversed([p for p in pauli])) for pauli in paulis_ZZ]
    return ham_ZZ, paulis_Z,paulis_X



def Hamiltonian_MFIM(model_input = (-1.0,-1.0,0.5),num_qubits = 5, bonds = []):
    """
    To build the Mixed-Field Ising model hamiltonian with the native qubit connectivitiy of the backend
    using sparse pauli operator for qiskit 
    """
    ham_ZZ, paulis_Z,paulis_X = ham_term_strings(num_qubits, bonds)
    J, hx, hz = model_input 
    # the hamitlonian created below with SparsePauliOp is to speed up calculations with qiskit
    hamiltonian  = SparsePauliOp(ham_ZZ, coeffs = J)+SparsePauliOp(paulis_Z, coeffs= hz)+SparsePauliOp(paulis_X, coeffs = hx)
    return hamiltonian 



def circuit_optimized_parameters(geometry = "FakeQuitoV2"): 
    """ Function consist of the optimized parameters for minimum energy after the state-vector VQE optimization step"""

    if geometry == "FakeQuitoV2":
        #initial_layout = [0, 1, 2, 3, 4]    
        # VQE solution for 1 layer HVA------- hardcoded here but are originally derived from optimizing the VQE solution. (Need to check)
        theta_Z_L_1 = -1.0903836560221376
        theta_X_L_1 = 1.5707963013100128
        theta_ZZ_L_1 = -1.290063556534689e-08

            # VQE solution for 2 layer HVA for 4 qubit chain
            #theta_Z_L_2 = [-0.9253781962387742, 0.05297769164990435]
            #theta_X_L_2 = [1.1782568203539736, 0.44552055156550735]
            #theta_ZZ_L_2 = [0.2425000962970552, -0.10748314808466695]
    elif geometry == "FakeCasablancaV2":
            # Casablanca geometry
            # initial_layout = [0, 1, 2, 3, 4, 5, 6]
            # VQE solution for 1 layer HVA for Casablanca geometry
        theta_Z_L_1 = -1.114862237442442
        theta_X_L_1 = 1.5707966423051756
        theta_ZZ_L_1 = 6.874680103745465e-07

            # VQE solution for 2 layer HVA for Casablanca geometry
            #theta_Z_L_2 = [-1.0493592817846746, 0.07760329617674103]
            #theta_X_L_2 = [1.2057488386027533, 0.34794432057731883]
            #theta_ZZ_L_2 = [0.218276186042823, -0.16232253800006316]
    elif geometry == "FakeGuadalupeV2":
        # initial_layout = range(16)
        # VQE solution for 1 layer HVA for Quadalupe geometry 
        theta_Z_L_1 = -1.16677864
        theta_X_L_1 = 1.57079632
        theta_ZZ_L_1 = 4.90858079e-09
    else: 
        print("Geometry not supported so far")

    return [theta_Z_L_1, theta_X_L_1, theta_ZZ_L_1]