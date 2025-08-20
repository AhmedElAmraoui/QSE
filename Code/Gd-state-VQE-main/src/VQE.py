# Essential imports for the calculations
from qiskit.providers.fake_provider import FakeQuitoV2, FakeGuadalupeV2
import scipy
import numpy as np
from circuit_builder import CircuitBuilder
from model_hamiltonian import *
from exact_diagonalization import *
from helper import QSimulator



# Here VQE is done with SV simulator 
def VQE_parameterized_circuit_sv(params =[],backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4], num_layers = 1):
    """
    Builds a parameterized circuits to perform the VQE step using HVA ansatz
    """
    Vcircuit= CircuitBuilder(params, backend, initial_layout, nlayers = num_layers)
    # here circuits are built without any measurement since State-vector simulator is used for VQE optimization
    circ_w_no_meas = [Vcircuit.makevqeCircuit(measure = False)]
    return circ_w_no_meas

def VQE_energy(params =[],backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4],bonds = [], num_layers = 1, model_input= ()):
    """
    Cost function for the optimizer, i.e., energy of the hamiltonian 
    """
    circ_w_no_meas = VQE_parameterized_circuit_sv(params, backend, initial_layout, num_layers )
    qsim = QSimulator(backend)
    SV_counts = qsim.State_Vector_Simulator(circ_w_no_meas) 
    res = SV_counts.expectation_value(Hamiltonian_MFIM(model_input = model_input, num_qubits = backend.num_qubits, bonds = bonds))
    res = np.real(res)
    return res



def optimizer(init_params,backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], bonds = [],num_layers = 1, model_input= ()):
    """
    Classical optimization step of VQE
    """
    res = scipy.optimize.minimize(VQE_energy, init_params, args = ( backend, initial_layout, bonds,num_layers,model_input),method ="BFGS")
    return res

def VQE_optimization_Step(init_params,backend = FakeGuadalupeV2(), initial_layout = [i for i in range(16)], bonds = [],num_layers = 1, model_input=()):
    res_vqe_sv = optimizer(init_params,backend,initial_layout, bonds, num_layers, model_input)
    if res_vqe_sv.success:
        print('Optimization was successful!')
        optimum_params = res_vqe_sv.x
        lowest_fun = res_vqe_sv.fun
        
    else:
        print(res_vqe_sv.success)    
    
    return optimum_params, lowest_fun

def exact_gd_state_energy(model_input, num_qubits,bonds):
    """
    function to evaluate the MFIM's exact energy using qutip
    """

    exact_ham = map_hamiltonian(complete_Hamiltonian_str(model_input,num_qubits,bonds))

    return get_lowest_states(exact_ham, 1)[0][0]

def main():

    # update the phase space parameter below 
    J = -1.0
    hx = -1.0
    hz = 0.5
    model_input = (J, hx, hz)
    print(f"Considered MFIM model parameters (J = {J}, hx = {hx}, hz = {hz})")

    # native connectivity of the Guadalupe backend
    bonds_Guad = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 9), (8, 11), (11, 14), (14, 13),
                   (13, 12), (12, 15), (1, 4), (4, 7), (7, 6), (7, 10), (10, 12)]
    # native connectivity of Quito backend
    bonds_Quito = [[0, 1],[1, 2],[1, 3],[3, 4]]
    # there are three parameters for each layer in this VQE HVA ansatz 
    num_params = 3 
    # number of layers for HVA ansatz 
    num_layers = 1

    # this is not needed for the time being
    # optimized parameters for the VQE step as used in the paper
    # opt_params_Quito =  circuit_optimized_parameters("FakeQuitoV2")
    # opt_params_Guad = circuit_optimized_parameters("FakeGuadalupeV2")
    

    # Chosing random parameters to perform VQE
    init_params =  np.random.uniform(-np.pi/3, np.pi/3, num_layers*num_params)
        
    # backend is chosen to decide the lattice geometry as well
    # for the time being only Quito and Guadalupe geometries are implemented
    # If you want to try a Guadalupe geometry change the backend to FakeGuadalupeV2() and bonds = bonds_Guad
    backend = FakeQuitoV2()  # FakeGuadalupeV2() 
    bonds = bonds_Quito # bonds_Guad 

    init_VQE_circuit = VQE_parameterized_circuit_sv(params = init_params,backend = backend,
                                       initial_layout = [i for i in range(backend.num_qubits)], 
                                       num_layers=num_layers)[0]
    init_VQE_energy = VQE_energy(params = init_params,backend = backend,
                                       initial_layout = [i for i in range(backend.num_qubits)], 
                                       bonds = bonds, num_layers=num_layers, model_input=model_input)
    
    # for reference printing the intial starting point for the VQE calculations
    print(f"Quantum circuit for randomly sample initial parameters: {init_params} with the corresponding VQE energy: {init_VQE_energy} is \n ", init_VQE_circuit)

    # Optimzation step of VQE
    optimal_params, lowest_energy = VQE_optimization_Step(init_params,backend = backend, initial_layout = [i for i in range(backend.num_qubits)], bonds=bonds, num_layers = num_layers, model_input=model_input)

    # Printing the optimized parameters and the corresponding lowest obtained energy 
    print(f"optimized parameters are {optimal_params} and obtained lowest energy is {lowest_energy}")  
    
    # below we printed the exact ground state energy of the hamiltonian 
    print(f"Exact ground state energy: {exact_gd_state_energy(model_input,backend.num_qubits,bonds)}")


if __name__ == "__main__":
    main()