from qiskit.quantum_info import random_statevector
import numpy as np
import scipy as sp

psi = random_statevector(4).data.reshape(2, 2)

U, S, Vh = np.linalg.svd(psi)

node = (np.diag(S[:2]) @ Vh[:2, :])

print(node)

v1 = node.reshape(4, -1)
v2 = sp.linalg.null_space(v1.conj().T)
U = np.concatenate([v1, v2], axis=1)
print(U)
import pdb
pdb.set_trace()