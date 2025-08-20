from qiskit_aer import Aer
import json 
import pickle
from qiskit.providers.fake_provider import FakeQuitoV2, FakeGuadalupeV2


class Quantum_system:
    """
    Class which containes the common quantum parameters for the project. 
    """
    def __init__(self, backend = None, initial_layout = [0, 1, 2, 3, 4]):
        self.backend = backend
        # print("here backend",self.backend)
        self.initial_layout = initial_layout # could be different than all qubit connectivity as used in this project
        self.geometry = self.backend_geometry()

    def backend_geometry(self):
        if type(self.backend)==type(FakeQuitoV2()):
            self.geometry = "FakeQuitoV2"
        elif type(self.backend)==type(FakeGuadalupeV2()):
            self.geometry = "FakeGuadalupeV2"
        return self.geometry

    def executor(self, circuits, shots = 1):
        return self.backend.run(circuits, shots = shots).result().get_counts()


class QSimulator(Quantum_system):
    """
    Class which has both ideal and noisy quantum simulators of Qiskit 
    """
    def __init__(self,backend):
        super().__init__(backend)
        
    def State_Vector_Simulator(self,circuits):
        # The circuits here are without measurements
        count = Aer.get_backend('statevector_simulator').run(circuits[0]).result().get_statevector()
        return count

    def QASM_Simulator(self,circuits,shots=1000):

        # For Z measurements
        count_QASM_Z =Aer.get_backend('qasm_simulator').run(circuits[0], shots=shots).result().get_counts()
        # For X measurements
        count_QASM_X =Aer.get_backend('qasm_simulator').run(circuits[1], shots=shots).result().get_counts()
        return count_QASM_Z,count_QASM_X

    def Noisy_backend_Simulato(self,circuits,shots=1000):
        count_Z = self.backend.run(circuits[0], shots=shots).result().get_counts()
        count_X = self.backend.run(circuits[1], shots=shots).result().get_counts()
        return count_Z, count_X


def data_loader_json(input_file = None):
    """
    Load a json file 
    Input: 
    input_file: path to the file 
    Output:
    Returns a loaded data 
    """
    file = open(input_file)
    data = json.load(file)
    return data

def data_loader_pickle(input_file = None):
    """
    Load a pickle file 
    Input: 
    input_file: path to the file 
    Output:
    Returns a loaded data 
    """
    with open(input_file, "rb") as f:
        file  = pickle.load(f)
    return file

def data_dump_pickle(output_data = None, output_file_name = None):
    """
    Dump a pickle file 
    Input:
    output_data: object to be dumped
    output_file_name: name of the output file
 
    """
    with open(output_file_name, "wb") as f:
        pickle.dump(output_data, f)
    # print("done pickle dump at ", output_file_name)
    

def convert_big_to_little_endian(pauli_groups):
    """
    Converts the pauli strings from big endian notation to little endian notation
    """
    little_endian_pauli_groups = []
    for pauli_group in pauli_groups:
        little_endian_pauli_group = ["".join(reversed([p for p in pauli])) for pauli in pauli_group]
        little_endian_pauli_groups.append(little_endian_pauli_group)
    return little_endian_pauli_groups
