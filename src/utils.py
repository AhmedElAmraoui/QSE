import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeAthensV2, FakeQuitoV2
from qiskit.visualization import plot_gate_map
from scipy.optimize import minimize
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap

import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.linalg import fractional_matrix_power
import networkx as nx






    











