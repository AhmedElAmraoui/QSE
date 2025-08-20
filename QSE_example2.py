import numpy as np
from scipy.linalg import eigh

# Pauli-Matrizen
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

# Tensorprodukt
def tensor(*ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# Baue Hamiltonian: Z0Z1 + Z1Z2 + X0 + Y1
P1 = tensor(Z, Z, I)
P2 = tensor(I, Z, Z)
P3 = tensor(X, I, I)
P4 = tensor(I, Y, I)
pauli_terms = [P1, P2, P3, P4]
H = sum(pauli_terms)

# Initialzustand: |+++>
zero = np.array([[1], [0]])
psi_000 = tensor(zero, zero, zero)
psi_plus = tensor(H_gate, H_gate, H_gate) @ psi_000

# Krylov-Vektoren erzeugen (Moment k = 0 bis K)
def generate_krylov_vectors(psi, pauli_terms, max_k=1):
    basis_ops = [np.eye(8)]  # Start mit Identität (Moment 0)
    for k in range(1, max_k + 1):
        new_ops = []
        for a in basis_ops:
            for p in pauli_terms:
                op = p @ a
                if not any(np.allclose(op, existing) for existing in basis_ops + new_ops):
                    new_ops.append(op)
        basis_ops.extend(new_ops)
    V = [op @ psi for op in basis_ops]
    return V, basis_ops

# Simulation für gegebenes K
def iqae_run(K):
    V, ops = generate_krylov_vectors(psi_plus, pauli_terms, max_k=K)
    V_matrix = np.column_stack(V)
    Q, R = np.linalg.qr(V_matrix)

    H_reduced = np.array([[np.vdot(Q[:, i], H @ Q[:, j]) for j in range(Q.shape[1])]
                          for i in range(Q.shape[1])])
    eigvals_approx, eigvecs_approx = eigh(H_reduced)
    ground_energy_approx = eigvals_approx[0]
    ground_state_approx = Q @ eigvecs_approx[:, 0]

    eigvals_exact, eigvecs_exact = eigh(H)
    ground_energy_exact = eigvals_exact[0]
    ground_state_exact = eigvecs_exact[:, 0]

    energy_error = np.abs(ground_energy_exact - ground_energy_approx)
    fidelity = np.abs(np.vdot(ground_state_exact, ground_state_approx)) ** 2

    return {
        "K": K,
        "approx_energy": ground_energy_approx,
        "exact_energy": ground_energy_exact,
        "energy_error": energy_error,
        "fidelity": fidelity,
        "dim_reduced": Q.shape[1]
    }

# Beispiel: adaptive Analyse für K = 0 bis 3
results = [iqae_run(K) for K in range(0, 4)]
for r in results:
    print(f"K={r['K']} | approx E={r['approx_energy']:.4f} | error={r['energy_error']:.4f} | fidelity={r['fidelity']:.4f} | dim={r['dim_reduced']}")
