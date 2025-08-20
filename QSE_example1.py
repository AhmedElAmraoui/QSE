import numpy as np
from scipy.linalg import eigh

# Einheitsmatrix (2x2)
I = np.eye(2)

# Pauli-Matrizen
X = np.array([[0, 1],
              [1, 0]])

Y = np.array([[0, -1j],
              [1j,  0]])

Z = np.array([[1,  0],
              [0, -1]])

H_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])


def tensor(*ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

P1 = tensor(Z,Z,I)
P2 = tensor(I,Z,Z)
P3 = tensor(X,I,I)
P4 = tensor(I,Y,I)

H = P1 + P2 + P3 + P4

zero = np.array([[1], [0]])
psi_000 = tensor(zero, zero, zero)
psi_plus = tensor(H_gate, H_gate, H_gate) @ psi_000


V0 = psi_plus
V1 = P1 @ psi_plus
V2 = P2 @ psi_plus
V3 = P3 @ psi_plus
V4 = P4 @ psi_plus

V = [V0, V1, V2, V3, V4]

Q, R = np.linalg.qr(V)  # Q ist orthonormale Basis

# Neue Overlap-Matrix = I
S_reduced = np.eye(Q.shape[1])

# Neue Hamilton-Matrix
H_reduced = np.array([[np.vdot(qi, H @ qj) for qj in Q.T] for qi in Q.T])

# Standard-Eigenwertproblem lösen
eigvals_approx, eigvecs = eigh(H_reduced)

# Overlap-Matrix S[i][j] = <V_i | V_j>
"""N = len(V)
S_ij = np.zeros((N, N), dtype=complex)
H_ij = np.zeros((N, N), dtype=complex)

for i in range(N):
    for j in range(N):
        S_ij[i, j] = np.vdot(V[i], V[j])
        H_ij[i, j] = np.vdot(V[i], H @ V[j])
        

eigvals, eigvecs = eigh(H_ij, S_ij)"""
