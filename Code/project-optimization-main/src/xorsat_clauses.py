# ToDo 
# add function to build the clauses for the Max 3-XORSAT problem  
# 
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 
import random
from math import comb
import itertools 
import warnings
from utils import *
from graph_plot import plot_graph 


def generate_random_3_xorsat(r: int =3 ,num_variables = int, num_clauses=int):
    
 
    """
    Generates a random 3-XORSAT problem with a specified number of variables and clauses.

    A 3-XORSAT problem involves selecting clauses where each clause is made up of 3 variables,
    and each clause has a randomly assigned XOR result (0 or 1).

    :param r: Number of variables per clause (default is 3 for 3-XORSAT).
    :param num_variables: Total number of variables to choose from.
    :param num_clauses: Number of clauses to generate.
    :return: A tuple consisting of a numpy array of clauses and a numpy array of XOR results.
    """
    # Calculate the maximum possible number of clauses (n choose r)
    max_clauses = comb(num_variables,r)

    if num_clauses>max_clauses:
        
        warnings.warn(f"Requested {num_clauses} clauses, but only {max_clauses} are possible. "
                      f"Using {max_clauses} instead.", UserWarning)
        num_clauses = max_clauses
        #print(f"No. of asked clauses is larger than maximum possible clauses, therfore, only {max_clauses} will be generated")

    # Generate all possible combinations of r variables from the set of num_variables
    all_possible_clauses = list(itertools.combinations(range(num_variables), r))

    # Randomly sample the required number of clauses from all possible clauses
    clauses = random.sample(all_possible_clauses, num_clauses)

    # Generate random XOR results (0 or 1) for each clause
    xor_result = np.random.choice(2, num_clauses)
    
    # Convert the clauses to a numpy array for consistency in return type
    return np.array(clauses), xor_result




def find_num_solutions(A_pass, b_pass):
    # Taken from https://github.com/dilinanp/chook/blob/master/chook/planters/regular_xorsat.py
    """
        Gauss Elimination method
        Using row reduction, determines the number of solutions that satisfy 
        the linear system of equations given by A_pass.X = b_pass mod 2.
        Returns zero if no solutions exist.
 
    """
    A = np.copy(A_pass)
    b = np.copy(b_pass)

    M, N = A.shape

    h = 0
    k = 0

    while h < M and k < N:

        max_i = h

        for i in range(h, M):
            if A[i, k] == 1:
                max_i = i
                break

        if A[max_i, k] == 0:
            k += 1
        else:
            if h != max_i:
                A[[h, max_i]] = A[[max_i, h]]
                b[[h, max_i]] = b[[max_i, h]]

            for u in range((h + 1), M):
                flip_val = A[u, k]
                A[u] = (A[u] + flip_val * A[h]) % 2
                b[u] = (b[u] + flip_val * b[h]) % 2

            h += 1
            k += 1

    # Find rows with all zeros
    num_all_zeros_rows = 0

    solutions_exist = True

    for i in range(M):
        if not np.any(A[i]):  # All-zero row encountered

            if b[i] != 0:
                solutions_exist = False
                break

            num_all_zeros_rows += 1

    if solutions_exist:
        rank = M - num_all_zeros_rows
        num_solutions = np.power(2, N - rank)
    else:
        num_solutions = 0

    return num_solutions


def generate_xorsat_prob_with_finite_sols(r ,num_variables ,num_clauses):
    """
    Generate XORSAT problem which has atleast one solution possible. 
    """
    sol = 0
    # run the loop until you find one of randomly generate xorsat problem which has a solution
    # based on Guass elimination
    while sol==0:
        Ap,bp = generate_random_3_xorsat(r,num_variables ,num_clauses)
        sol = find_num_solutions(A_pass = Ap, b_pass=bp)
        if sol!=0:
            break
    return Ap, bp, sol
    



def planted_partial_solution_xorsat(xor_prob, num_vars = 6, epsilon = 0.2):
    """
    Generate a planted solution by changing the b's (or Vijk) 
    for a randomly generated XORSAT problem for a specific number of clauses 
    Logic is based on arXiv:2312.06104
    """
     
    A_mat, b_mat = xor_prob # we  could in principle build a b_mat here as well 
    num_clauses = A_mat.shape[0]
    # print(A_mat.shape)
    # calculate the partial num of clauses to be satisfied in the solution
    partial_num_clauses = int(np.ceil((1-epsilon)*num_clauses))
    # print(partial_num_clauses)
               
    mat_full = create_dense_clause_matrix(A_mat,num_vars)
    print(mat_full)
    # randomly selcting some contraints included, given by partial_num_clauses
    # sorted so that it is trackable 
    random_partial_constraint = random_planted_sat_clauses(num_clauses, partial_num_clauses )
       
    # make the new matrix with randomly selected clauses
    mat_PPSP = mat_full[random_partial_constraint]
    b_PPSP = b_mat[random_partial_constraint]
    
    # seed can be removed later 
    np.random.seed(1)
    # randomly select a solution bitstring X
    X_PPSP = np.random.choice(2, num_vars)
    # Find the b matrix when all selected clauses are satisfied
    b_plant = (mat_PPSP@X_PPSP)%2
    
    # b_copy = b_partial.copy()
    # b_copy = np.where(b_plant != b_copy, b_plant, b_copy)
    # flip the value of b (Vijk) to fulfill all randomly selected prrtial constraints and rest stays unsatisfied/or satisfied
    b_PPSP = np.where(b_plant != b_PPSP, b_plant, b_PPSP) 

    assert(np.array_equal(b_plant, b_PPSP))
    # put the planted b values back in the original b matrix
    b_full_w_PPSP = b_mat.copy()
    b_full_w_PPSP[random_partial_constraint] = b_PPSP
    assert(np.array_equal(((mat_PPSP@X_PPSP)%2), b_PPSP))
    return mat_PPSP, X_PPSP,b_full_w_PPSP,b_full_w_PPSP 
    
    

    
    
    
xor_prob = generate_random_3_xorsat(r=3,num_variables =6,num_clauses=8)
#        (np.array([[2, 4, 5],
#        [0, 1, 4],
#        [0, 3, 4],
#        [0, 1, 3],
#        [1, 2, 3],
#        [0, 1, 5],
#        [0, 2, 5],
#        [0, 1, 2]]),
# np.array([1, 1, 0, 0, 1, 1, 1, 1]))

#print(generate_xorsat_prob_with_finite_sols(r=3,num_variables =6,num_clauses=8))

num_variables = 6
mat_P, _,_,_= planted_partial_solution_xorsat(xor_prob,6,0.2)
print(create_sparse_clause_matrix(mat_full= mat_P))
#plot_graph(xor_prob, num_variables)