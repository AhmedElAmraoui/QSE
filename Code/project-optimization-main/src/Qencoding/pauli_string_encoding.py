import numpy as np 
from qiskit import QuantumCircuit, Aer, transpile
from typing import Dict, List, Tuple, Callable
from qiskit.circuit import Parameter


def brickwall_circuit(k:int = 2, num_variables:int = 9, parameters:list=[]):
    """
    hardware efficient circuit ansatz    
    """
    # write a function for finding roots of a polynomial
    print("initial parameters: ", parameters)
    ansatz_encoded_qubits =  np.sqrt(num_variables)
    qc = QuantumCircuit(ansatz_encoded_qubits)
    even_bonds, odd_bonds = bonds_without_native_connectivity(qc)
    num_params_one_layer = (4*qc.num_qubits+len(even_bonds)+len(odd_bonds))
    reps = 1
    ### parameters = [Parameter(f'theta{i}') for i in range(reps*num_params_one_layer)] 

    for i in range(reps):
        params = parameters[i*num_params_one_layer:(i+1)*num_params_one_layer]
        add_ansatz_layer(qc, params)

    return qc
def add_ansatz_layer(qc: QuantumCircuit, params_one_layer:list= []):
    rx1_angles, rz1_angles, rx2_angles, rz2_angles = np.split(np.array(params_one_layer[:4 * qc.num_qubits]), 4) 
    #single_qubits_params = params_one_layer[:4*qc.num_qubits]
    #two_qubits_params = params_one_layer[4*qc.num_qubits:]



    append_rx_gate(qc,  rx1_angles)#single_qubits_params[:qc.num_qubits])
    append_rz_gate(qc, rz1_angles)# single_qubits_params[qc.num_qubits:2*qc.num_qubits])
    even_bonds, odd_bonds = bonds_without_native_connectivity(qc)
    even_angles, odd_angles = np.split(np.array(params_one_layer[4 * qc.num_qubits:]), [len(even_bonds)])
    #print(even_angles,odd_angles)

    append_brick_bonds(qc,bonds = even_bonds, angles = even_angles) #two_qubits_params[:len(even_bonds)])
    append_rx_gate(qc, rx2_angles) #single_qubits_params[2*qc.num_qubits:3*qc.num_qubits])
    append_rz_gate(qc, rz2_angles)# single_qubits_params[-qc.num_qubits:])
    append_brick_bonds(qc,bonds = odd_bonds, angles = odd_angles)# two_qubits_params[-len(odd_bonds):])
    qc.barrier()
   

def append_rx_gate(qc: QuantumCircuit, angles: list[float]) -> None:
    """
    Appends an X-rotation term to the quantum circuit for the given qubit and beta parameter.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to modify.
    q1 (int): The qubit index for the X-rotation.
    angle (float): The beta parameter controlling the X-rotation.
    """
    for i,angle in enumerate(angles):
        qc.rx(angle,i)

def append_rz_gate(qc: QuantumCircuit, angles: list[float]) -> None:
    """
    Appends an X-rotation term to the quantum circuit for the given qubit and beta parameter.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to modify.
    q1 (int): The qubit index for the Z-rotation.
    angle (float): The beta parameter controlling the Z-rotation.
    """
    for i,angle in enumerate(angles):
        qc.rz(angle,i)


    
def append_brick_bonds(qc: QuantumCircuit,bonds:list =[], angles:list =[]):
    for idx,(i,j) in enumerate(bonds):
        qc.cx(i,j)
        qc.rz(angles[idx],j)
        qc.cx(i,j)



def bonds_without_native_connectivity(qc: QuantumCircuit):
    even_bonds = []
    odd_bonds = []

    for i in range(qc.num_qubits - 1):
        if i % 2 == 0:
            even_bonds.append((i, i + 1))
        else:
            odd_bonds.append((i, i + 1))
    return even_bonds, odd_bonds


init_params = np.random.uniform(0,np.pi,(4*4+2+1))
cir = brickwall_circuit(k = 2, num_variables= 16,parameters = init_params)
print(cir)

def add_X_measurement(qc: QuantumCircuit):
    qc.h(range(qc.num_qubits))
    qc.measure_all()
    return qc

def add_Y_measurement(qc: QuantumCircuit):
    qc.sdg(range(qc.num_qubits))
    qc.h(range(qc.num_qubits))
    return qc

def add_Z_measurement(qc: QuantumCircuit):
    qc.measure_all()
    return qc


def staircase_circuit():
    """
    
    """

    return


def generate_Pauli_correlation_encoding():
    
    return 


