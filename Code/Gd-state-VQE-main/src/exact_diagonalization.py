from qutip import qeye, sigmax, sigmay, sigmaz, tensor, Qobj
from model_hamiltonian import *

def complete_Hamiltonian_str(model_input= ( -1, -1,0.5),num_qubits = 5, bonds = []):
    """
    Generate complete hamiltonian strings
    """
    ham_ZZ, paulis_Z,paulis_X = ham_term_strings(num_qubits, bonds)
    J, hx, hz = model_input
    Hamiltonian_str = [str(J)+"*"+element for element in ham_ZZ] \
        + [str(hz)+"*"+element for element in paulis_Z]\
              + [str(hx)+"*"+element for element in paulis_X]
    return Hamiltonian_str 

def map_hamiltonian(s_list):
    '''
    Write down the original string of spin-S operators in terms of Pauli
    matrices, while respecting the chosen encoding using Qutip
    '''
    pauli = [qeye(2), sigmax(), sigmay(), sigmaz()]

    pm_dict = {"I": pauli[0], "X": pauli[1], \
                        "Y": pauli[2], "Z": pauli[3]}

    op_total = 0.
    for op in s_list:
        coef, label = float(op.split('*')[0]), op.split('*')[1]
        op_total += float(coef)*tensor([pm_dict[x] for x in label])
        #print(op_total)

    return op_total

def get_lowest_states(hamiltonian, num):
    '''
    get the lowest num eigenvalues and eigenstates.
    '''
    dim = hamiltonian.shape[0]
    w, v = hamiltonian.eigenstates(eigvals=num, sparse=dim>4)
    return w, v
