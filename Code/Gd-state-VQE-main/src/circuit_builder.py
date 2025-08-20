# Essential imports for the calculations
from qiskit import QuantumCircuit, Aer, transpile
from helper import Quantum_system

        
class CircuitBuilder(Quantum_system):
    """
    Class to build a VQE circuit based on the native qubit connectivity of the chosen fake backend. 
    Currently implemented are FakeQuitV2 and FakeGuadalupeV2
    """
    def __init__(self,params,backend , initial_layout , nlayers = int):
        super().__init__(backend,initial_layout)
        self.params = params
        self.num_qubits = len(self.initial_layout)
        self.nlayers = nlayers
       

    def vqeLayer_FakeQuito(self,params):#theta_ZZ, theta_Z, theta_X): #params has to be a list a list for each layer
        """ 
        VQE layer for the FakeQuito geometry using all qubits and native connectivity
        """
        theta_Z = params[0] # make bonds a list of bond list 
        theta_X = params[1]
        theta_ZZ = params[2]
        vqeLayer = QuantumCircuit(self.num_qubits)
        # Choosen bond pairs according to the native qubit connectivity of the backend
        bonds_1 = [[0, 1], [3, 4]] 
        bonds_2 = [[1, 2]] 
        bonds_3 = [[1, 3]]
        # the RZ and RZ terms for the field terms of the hamiltonian. 
        # Applied first to get the sequence of layers for PER later to come out correctly, i.e., single qubit gates first followed by clifford gates. 
        
        vqeLayer.rz(theta_Z, range(self.num_qubits))
        vqeLayer.rx(theta_X, range(self.num_qubits))
    
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))
        vqeLayer.rz(theta_ZZ, [bonds_1[i][1] for i in range(len(bonds_1))])
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))

        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))
        vqeLayer.rz(theta_ZZ, [bonds_2[i][1] for i in range(len(bonds_2))])
        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))

        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))
        vqeLayer.rz(theta_ZZ, [bonds_3[i][1] for i in range(len(bonds_3))])
        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))

        return vqeLayer
    
    def vqeLayer_FakeGuadalupeV2(self,params):
        """ VQE layer for the FakeGuadalupeV2() geometry using all qubits and native connectivity"""
        theta_Z = params[0] # make bonds a list of bond list 
        theta_X = params[1]
        theta_ZZ = params[2]
        vqeLayer = QuantumCircuit(self.num_qubits)
        # Choosen bond pairs according to the native qubit connectivity of the backend
        bonds_1 = [(0, 1), (2, 3), (4, 7), (10, 12)]
        bonds_2 = [[1, 2], [3, 5],[7,6],[8,9],[12,13]] 
        bonds_3 = [[1, 4], [7,10],[12,15]] # ,[8,11]
        bonds_4 = [(5, 8) ,(11, 14)]
        bonds_5 = [[8,11]]

        vqeLayer.rz(theta_Z, range(self.num_qubits))
        vqeLayer.rx(theta_X, range(self.num_qubits))
        #vqeLayer.barrier()
    
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))
        vqeLayer.rz(theta_ZZ, [bonds_1[i][1] for i in range(len(bonds_1))])
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))
        #vqeLayer.barrier()

        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))
        vqeLayer.rz(theta_ZZ, [bonds_2[i][1] for i in range(len(bonds_2))])
        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))
        #vqeLayer.barrier()

        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))
        vqeLayer.rz(theta_ZZ, [bonds_3[i][1] for i in range(len(bonds_3))])
        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))
        #vqeLayer.barrier()

        vqeLayer.cx(*zip(*[bonds_4[i] for i in range(len(bonds_4))]))
        vqeLayer.rz(theta_ZZ, [bonds_4[i][1] for i in range(len(bonds_4))])
        vqeLayer.cx(*zip(*[bonds_4[i] for i in range(len(bonds_4))]))
        #vqeLayer.barrier()

        vqeLayer.cx(*zip(*[bonds_5[i] for i in range(len(bonds_5))]))
        vqeLayer.rz(theta_ZZ, [bonds_5[i][1] for i in range(len(bonds_5))])
        vqeLayer.cx(*zip(*[bonds_5[i] for i in range(len(bonds_5))]))
    
        #vqeLayer.barrier()
        return vqeLayer
    
   
    def makevqeCircuit(self, measure = False, meas_basis = "Z"): 
        vqeCircuit = QuantumCircuit(self.num_qubits) 
        vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
        vqeCircuit.barrier()
        for i in range(self.nlayers):
            if self.geometry == "FakeQuitoV2":
                vqeL = self.vqeLayer_FakeQuito(self.params[3*i:3*(i+1)]) #self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
            elif self.geometry == "FakeGuadalupeV2":
                vqeL = self.vqeLayer_FakeGuadalupeV2(self.params[3*i:3*(i+1)])#self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
            vqeCircuit = vqeCircuit.compose(vqeL)
            vqeCircuit.barrier()
               
        if measure == True:
            if meas_basis == "Z":
                vqeCircuit.measure_all()
                transpiled = transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
            elif meas_basis =="X":
                vqeCircuit.h(range(self.num_qubits))
                vqeCircuit.measure_all()
                transpiled = transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
            else: 
                print("Measurement not defined")    # Y-measurements can be added
        else: 
            transpiled = vqeCircuit # transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
        return transpiled
    

