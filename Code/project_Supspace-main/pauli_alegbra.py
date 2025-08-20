import numpy as np
import qiskit
#import pennylane as qml

def get_phase(a,b):
# Return the phase after the product of multiplication of the Pauli's represented as (I,X,Y,Z) ==(0,1,2,3)
# for eg. Z*X = iY ==> then function returns 1j
    phase  = None
    if a==b or a==0 or b==0:
        # if anyone is idenitity or the pauli are same and then return 1
        return 1.0
    if b>a and a%2 != b%2:
        # check for cyclicness, i.e., XY (12), YZ (23) should give 1j
        phase = 1j
    else:
        # XZ(13) returns -1j
        phase = -1j
    if a>b and a%2 == b%2:
        # look for ZX(31) and returns 1j
        phase = 1j
    else:
        # check for ZY(32) and YX(21) and returns -1j
        phase = -1j
    return phase


def get_tot_phase(x,y):
    #  Perform qubit-wise multiplication of Pauli and returns the overall phase 
    #  using this mapping (I,X,Y,Z) ==(0,1,2,3) 
    #  for eg. ZZ*IX = (ZI)(ZX) = iZY ==> returns 1j
    phase_tot = 1.
    for a,b in zip(x,y):
        phase_tot = phase_tot*get_phase(a,b)
     
    if np.imag(phase_tot)==0:
        return np.real(phase_tot)
    return phase_tot 

def get_pauli_product(x,y):

    # logic: Actually remainder with 4 fills in the whole product table nicely. 
    # -1%4 = 3 
    return tuple((a - b) % 4 if a % 2 else (b - a) % 4
            for a, b in zip(x, y))

def get_pauli_list_product(pauli_list_1,pauli_list_2,redundant_list):
    # set(pauli pdts) removes the reduntant terms in it after multiplication 
    # then subtracting the set of redundant_list remove repetition of repeated 
    return list(set([get_pauli_product(a,b) for a in pauli_list_1 for b in pauli_list_2])-set(redundant_list))

def kse_pauli_strings(paulilist,power):
    ref_map = ['I','X','Y','Z']
    kse_strings = [[tuple(ref_map.index(pauli) for pauli in paulistr.split('*')[1]) for paulistr in paulilist]]
    # the list below is used to remove the reduntant the all Identity term and paulilist initially passed. 
    list_to_remove_duplicates = kse_strings[0]+[tuple(0 for _ in range(len(paulilist[0].split("*")[1])))]
    for i in range(power-1):
        kse_strings.extend([get_pauli_list_product(kse_strings[0],kse_strings[-1],list_to_remove_duplicates)])
        list_to_remove_duplicates.extend(kse_strings[-1])
    kse_strings_w_coeff =[['1.0*'+''.join(ref_map[i] for i in string) for string in kse_string ] for kse_string in kse_strings]
    return kse_strings_w_coeff


def pauli_strings_list_multiplication(Pauli_string_1,Pauli_string_2):
    #basically for the hamiltonian!!!
    ref_map = ['I','X','Y','Z']
    Pstr1_as_list_of_tuple = [tuple(ref_map.index(pauli) for pauli in paulistr.split('*')[1]) for paulistr in Pauli_string_1]
    coeff_Pstr1 = [float(paulistr.split('*')[0]) for paulistr in Pauli_string_1]

    Pstr2_as_list_of_tuple = [tuple(ref_map.index(pauli) for pauli in paulistr.split('*')[1]) for paulistr in Pauli_string_2]
    coeff_Pstr2 = [float(paulistr.split('*')[0]) for paulistr in Pauli_string_2]
    pdt_list = [] # [(coe)]
    for coeff_1, pauli_1 in zip(coeff_Pstr1,Pstr1_as_list_of_tuple):
        product = [(coeff_1*coeff_2*get_tot_phase(pauli_1,pauli_2),get_pauli_product(pauli_1,pauli_2))for coeff_2, pauli_2 in zip(coeff_Pstr2,Pstr2_as_list_of_tuple)]
        # below we are removing all terms with imaginary overall coefficients--considering 
        # only terms in pauli for a hermitian operator like hamiltonian
        pdt_list.extend([pauli for pauli in product if isinstance(pauli[0],float)])

    paulis_to_label = [''.join(ref_map[pauli] for pauli in pauli_lt[1]) for pauli_lt in pdt_list]
    paulis_to_label_w_coeff = [f'{pauli_lt[0]}*'+''.join(ref_map[pauli] for pauli in pauli_lt[1]) for pauli_lt in pdt_list]

    #return paulis_to_label_w_coeff
    # make every terms coeff as 1 and remove the duplicated no matter what their coeff is..!!
    return [f"1.0*{x}" for x in sorted(list(set(paulis_to_label)))] 



def pauli_strings_list_multiplication_wout_duplicates(Pauli_string_1,Pauli_string_2):
    #basically for the hamiltonian without duplicates!!!
    ref_map = ['I','X','Y','Z']
    Pstr1_as_list_of_tuple = [tuple(ref_map.index(pauli) for pauli in paulistr.split('*')[1]) for paulistr in Pauli_string_1]
    coeff_Pstr1 = [float(paulistr.split('*')[0]) for paulistr in Pauli_string_1]

    Pstr2_as_list_of_tuple = [tuple(ref_map.index(pauli) for pauli in paulistr.split('*')[1]) for paulistr in Pauli_string_2]
    coeff_Pstr2 = [float(paulistr.split('*')[0]) for paulistr in Pauli_string_2]
    pdt_list = [] # [(coe)]
    for coeff_1, pauli_1 in zip(coeff_Pstr1,Pstr1_as_list_of_tuple):
        product = [(coeff_1*coeff_2*get_tot_phase(pauli_1,pauli_2),get_pauli_product(pauli_1,pauli_2))for coeff_2, pauli_2 in zip(coeff_Pstr2,Pstr2_as_list_of_tuple)]
        # below we are removing all terms with imaginary overall coefficients--considering 
        # only terms in pauli for a hermitian operator like hamiltonian
        pdt_list.extend([pauli for pauli in product if isinstance(pauli[0],float)])

    paulis_to_label = [''.join(ref_map[pauli] for pauli in pauli_lt[1]) for pauli_lt in pdt_list]
    paulis_to_label_w_coeff = [f'{pauli_lt[0]}*'+''.join(ref_map[pauli] for pauli in pauli_lt[1]) for pauli_lt in pdt_list]

    #return paulis_to_label_w_coeff
    # make every terms coeff as 1 and remove the duplicated no matter what their coeff is..!!
    return [f"1.0*{x}" for x in sorted(list(set(paulis_to_label)))] 

def powers_of_H(H,power):
    # calculates the square of H first
    hpow= [H,pauli_strings_list_multiplication(H,H)]
    
    # Now we calculate the higher powers here 
    for _ in range(2,power):
        hpow.append(pauli_strings_list_multiplication(hpow[-1],H))
    return hpow

def get_powers_of_h_wout_duplicates(H, power):
    ### copied from Yongxin's code but re-do it
    ref = ['I', 'X', 'Y', 'Z']
    nq = len(H[0].split('*')[1])
    
    H_powers_str = powers_of_H(H, power)

    Id = "1.0*" + nq * 'I'
    lst = [[f"1.0*{x.split('*')[1]}" for x in H], H_powers_str[0]]
    lst[1].remove(Id)

    tot = lst[0] + lst[1] + [Id] # used for removing duplicates
    for Hpow in H_powers_str[1:]:
        tmp = [f"1.0*{x.split('*')[1]}" for x in Hpow]
        lst.extend([list(set(tmp) - set(tot))])
        tot.extend(lst[-1])
    return lst


#["1.0*ZZ","1.0*ZI",'1.0*XI']# replace this with a hamitonian extracted as pauli strings 
#print('pwoers of h',powers_of_H(H,power))

def qubitwise_commuting_strings_u_qiskit(paulistrings):
    #print(paulistrings)
    pstrings = [pstr.split('*')[1] for pstr in paulistrings]
    op_pstr = qiskit.quantum_info.PauliList(pstrings)
    op_group = op_pstr.group_commuting(qubit_wise=True)
    return [i.to_labels() for i in op_group]

def qubitwise_commuting_strings_u_pennylane(paulistrings):
    wire_map = {'0' : 0, '1' : 1, '2' : 2,'3':3,'4':4}
    pauli_word_str = [qml.pauli.string_to_pauli_word(pstr.split('*')[1], wire_map=wire_map) for pstr in paulistrings]
    coeffs = [float(pstr.split('*')[0]) for pstr in paulistrings]
    obs_groupings, coeffs_groupings = qml.pauli.group_observables(pauli_word_str, coeffs, 'commuting', 'lf')
    return [[qml.pauli.pauli_word_to_string(i,wire_map=wire_map) for i in obs_grouping] for obs_grouping in obs_groupings]



def run ():
    "simplest way of using the code to generate commuting pauli groups for KSE"
    ham_p_strings = []
    for i in kse_pauli_strings(H,3):
        ham_p_strings.extend([j for j in i])
#print("pinted h3 = ",h3)
#print("hamiltonian = ",[i.split('*')[1] for i in H])


    print(len(ham_p_strings))


 
    qwc = qubitwise_commuting_strings_u_qiskit(ham_p_strings)
    print(len(qwc))

    qwc_penny = qubitwise_commuting_strings_u_pennylane(ham_p_strings)
    print(len(qwc_penny))





