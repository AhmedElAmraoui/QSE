import numpy as np
from typing import Tuple, Dict
from collections import defaultdict

def create_dense_clause_matrix(A_mat, num_vars = 6):
    """
    take matrix like this 
    array([[2, 4, 5],
        [0, 1, 4],
        [0, 3, 4],
        [0, 1, 3],
        [1, 2, 3],
        [0, 1, 5],
        [0, 2, 5],
        [0, 1, 2]])
    
    and convert it into 
    [[0. 0. 1. 0. 1. 1.]
    [1. 1. 0. 0. 1. 0.]
    [1. 0. 0. 1. 1. 0.]
    [1. 1. 0. 1. 0. 0.]
    [0. 1. 1. 1. 0. 0.]
    [1. 1. 0. 0. 0. 1.]
    [1. 0. 1. 0. 0. 1.]
    [1. 1. 1. 0. 0. 0.]]
    
    """
    num_clauses = A_mat.shape[0]
    # Initialize the matrix with zeros
    mat_full = np.zeros((num_clauses, num_vars))

    # make assignment matrix for the clauses
    for idx, row in enumerate(A_mat):
        mat_full[idx, row] = 1  
    
    return mat_full

def random_planted_sat_clauses(num_clauses, PSPP_sat_clauses ):
    return sorted(np.random.choice(num_clauses, size = PSPP_sat_clauses, replace = False))

def create_sparse_clause_matrix(mat_full):
    """
    Takes np.array([[0., 0., 1., 0., 1., 1.],
       [1., 1., 0., 0., 1., 0.],
       [1., 0., 0., 1., 1., 0.],
       [1., 1., 0., 1., 0., 0.],
       [0., 1., 1., 1., 0., 0.],
       [1., 1., 0., 0., 0., 1.],
       [1., 0., 1., 0., 0., 1.],
       [1., 1., 1., 0., 0., 0.]])
       
       returns 
       array([[2, 4, 5],
       [0, 1, 4],
       [0, 3, 4],
       [0, 1, 3],
       [1, 2, 3],
       [0, 1, 5],
       [0, 2, 5],
       [0, 1, 2]])
    """
    
    return np.array([list(np.where(row ==1)[0]) for row in mat_full])

        

def mapping_XORSAT_to_Hamiltonian(xor_prob: Tuple[np.ndarray, np.ndarray]) -> Dict[str, int]: 
    """
    Maps XOR-SAT problem clauses to a Hamiltonian form.

    Parameters:
    xor_prob (Tuple[np.ndarray, np.ndarray]): A tuple where the first element is a 2D array of integers
                                              representing the XOR-SAT clauses, and the second element 
                                              is a 1D array of integers (0 or 1) representing the xorsat solution.

    Returns:
    Dict[str, int]: A dictionary where keys are the string representations of the clauses 
                    and values are -1 or 1, depending on the parity.
    
    """
    xorsat_hamiltonian  = {}
    # Join the clause elements as a string and calculate the Hamiltonian value
    for clause, xorsat_solution in zip(xor_prob[0],xor_prob[1]):
        xorsat_hamiltonian ["".join(map(str,clause))] = (-1)**xorsat_solution

    return xorsat_hamiltonian 


def invert_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Reverses the bitstring keys in the given dictionary of counts.
    
    Parameters:
    counts (Dict[str, int]): A dictionary where the keys are bitstrings (binary strings) 
                             and the values are integer counts.
                             
    Returns:
    Dict[str, int]: A new dictionary with the bitstrings reversed in the keys.
    
    Raises:
    ValueError: If any of the keys are not strings or if any of the values are not integers.
    """
    # Input validation
    if not all(isinstance(k, str) for k in counts.keys()):
        raise ValueError("All keys must be strings representing bitstrings.")
    
    if not all(isinstance(v, int) for v in counts.values()):
        raise ValueError("All values must be integers representing counts.")
    
    # Reverse the bitstrings in the keys
    return {k[::-1]: v for k, v in counts.items()}

from typing import Dict

def compute_Hijk(bitstr: str, xorsat_hamiltonian: Dict[str, int]) -> int:
    """
    Computes the Hijk value for a given bitstring and an xorsat hamiltonian dictionary.
    
    Parameters:
    bitstr (str): A binary string representing the bitstring (e.g., '110011').
    xorsat_hamiltonian (Dict[str, int]): A dictionary where keys are clauses (e.g., '012') and 
                                         values are integers representing the xorsat solution 
                                         (-1 or 1).
    
    Returns:
    int: The computed Hijk value based on the given bitstring and the xorsat hamiltonian.
    
    Raises:
    ValueError: If the bitstr contains invalid characters (anything other than '0' or '1').
    KeyError: If a key in xorsat_hamiltonian refers to indices that are out of bounds in bitstr.
    """
    # Validate the bitstring (only '0' and '1' allowed)
    if not all(bit in '01' for bit in bitstr):
        raise ValueError("bitstr must be a binary string (containing only '0' and '1').")
    
    # Initialize the total Hijk value
    tot_hijk = 0
    
    # Loop through each clause in the xorsat_hamiltonian
    for clause_key, clause_result in xorsat_hamiltonian.items():
        # Extract bits from the bitstring according to the clause key
        try:
            bits = [int(bitstr[int(index)]) for index in clause_key]
        except IndexError:
            raise KeyError(f"Key '{clause_key}' contains indices that are out of bounds for the bitstring.")
        
        # Compute the hijk value for the current clause
        hijk_val = 1
        for bit in bits:
            # Calculate the contribution of the current bit to hijk_val
            hijk_val *= -clause_result * (-1) ** bit
        
        # Add the computed hijk_val to the total
        tot_hijk += hijk_val
    
    return tot_hijk




def compute_satisfied_constraints(bitstr: str, xorsat_hamiltonian: Dict[str, int]) -> int:
    """
    Calculates the number of satisfied constraints (clauses) for a given bitstring 
    in an XORSAT problem, based on the provided xorsat_hamiltonian.

    Parameters:
    bitstr (str): A binary string representing the bitstring (e.g., '110011').
    xorsat_hamiltonian (Dict[str, int]): A dictionary where keys represent clauses 
                                         (e.g., '012' for bits at positions 0, 1, and 2) and 
                                         values are integers (either 1 or -1) representing 
                                         whether the clause is satisfied or not.

    Returns:
    int: The number of satisfied constraints (clauses).
    
    Raises:
    ValueError: If bitstr contains invalid characters (anything other than '0' or '1').
    KeyError: If a key in xorsat_hamiltonian does not map to valid indices in bitstr.
    """
    # Validate that bitstr only contains '0' or '1'
    if not all(bit in '01' for bit in bitstr):
        raise ValueError("bitstr must be a binary string (containing only '0' and '1').")
    
    num_sat_constraints = 0  # Initialize the count of satisfied constraints
    
    # Loop through the clauses in the xorsat_hamiltonian dictionary
    for clause_key, clause_result in xorsat_hamiltonian.items():
        # Extract the bits from the bitstring corresponding to the current clause
        try:
            bits = [int(bitstr[int(index)]) for index in clause_key]
        except IndexError:
            raise KeyError(f"Clause key '{clause_key}' contains indices that are out of bounds for the bitstring.")
        
        # Check if the bitstring satisfies the current clause
        if (-1) ** (sum(bits) % 2) == clause_result:
            num_sat_constraints += 1
    
    return num_sat_constraints


def compute_max_xorsat_energy(counts: Dict[str, int], xorsat_hamiltonian: Dict[str, int]) -> float:
    """
    Computes the maximum XORSAT energy based on measurement counts and their respective objective values.
    
    Parameters:
    counts (Dict[str, int]): A dictionary where keys are bitstrings (measurement outcomes) and 
                             values are the number of times each bitstring was measured.
    # compute_Hijk (function): A function that computes the Hijk value for a given bitstring.
    xorsat_hamiltonian (Dict[str, int]): A dictionary where keys represent clauses 
                                         (e.g., '012' for bits at positions 0, 1, and 2) and 
                                         values are integers (either 1 or -1) representing 
                                         whether the clause is satisfied or not.

    Returns:
    float: The normalized XORSAT energy.

    Raises:
    ValueError: If the counts dictionary is empty or total measurement count is zero.
    """
    if not counts:
        raise ValueError("The counts dictionary is empty.")
    
    energy = 0
    total_count = 0
    
    # Calculate the energy based on measurement outcomes and their counts
    for meas, meas_count in counts.items():
        obj_4_meas = compute_Hijk(bitstr=meas, xorsat_hamiltonian= xorsat_hamiltonian)  # Compute the objective value (Hijk) for each bitstring
        energy += obj_4_meas * meas_count
        total_count += meas_count
    
    if total_count == 0:
        raise ValueError("Total count of measurements is zero, cannot compute energy.")

    return energy / total_count

def compute_max_xorsat_energy_cVar(counts: Dict[str, int], xorsat_hamiltonian: Dict[str, int]) -> float:
    """
    Computes the average energy of the lowest 7 (to be changed later) measurement outcomes in the XORSAT problem.

    Parameters:
    counts (Dict[str, int]): A dictionary where keys are measurement outcomes (bitstrings)
                              and values are the corresponding counts.
    xorsat_hamiltonian (Dict[str, int]): A dictionary where keys represent clauses 
                                         (e.g., '012' for bits at positions 0, 1, and 2) and 
                                         values are integers (either 1 or -1) representing 
                                         whether the clause is satisfied or not.


    Returns:
    float: The average energy of the lowest 7 measurement outcomes, normalized by total count.
    
    Raises:
    ValueError: If counts are empty or do not contain valid measurement data.
    """
    if not counts:
        raise ValueError("The counts dictionary is empty. Please provide valid measurement counts.")

    low_energies = defaultdict(tuple)
    energy = 0
    total_count = 0

    for meas, meas_count in counts.items():
        obj_4_meas = compute_Hijk(bitstr=meas,xorsat_hamiltonian= xorsat_hamiltonian )
        low_energies[meas] = (meas_count, obj_4_meas)

    # Sort the low_energies dictionary by the objective value and select the lowest 7
    energy_10 = dict(sorted(low_energies.items(), key=lambda x: x[1][1])[:7])
    
    for basis, val in energy_10.items():
        energy += val[1] * val[0]  # Weighted energy
        total_count += val[0]      # Total counts of selected measurements

    if total_count == 0:
        raise ValueError("Total count of measurements is zero. Cannot compute average energy.")
    # Return the average energy based on the selected measurements
    return energy / (7 * total_count)  

xor_prob =   (np.array([[2, 4, 5],
       [0, 1, 4],
       [0, 3, 4],
       [0, 1, 3],
       [1, 2, 3],
       [0, 1, 5],
      [0, 2, 5],[0, 1, 2]]),
 np.array([1, 1, 0, 0, 1, 1, 1, 1]))

#print(mapping_XORSAT_to_Hamiltonian(xor_prob))

