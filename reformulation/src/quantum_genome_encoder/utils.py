"""
utils.py
---
This code contains utility functions for the quantum encoding library.

Mainly this source provides functionality for different types of
quantum state which could be used for benchmarking purposes.

Author: Floyd Creevey
Email: fc309@sanger.ac.uk
"""
import .encoder

def generate_w_state(num_qubits):
    """
    Generates an entangled quantum W state for a given number of
    qubits.

    Parameters:
    -----------
    num_qubits : int
        Qubits for required W state.

    Returns:
    --------
    statevector : np.typing.ArrayLike
        Corresponding state vector to W state.
    """
    statevector = np.zeros(2**num_qubits, dtype=np.complex128)
    bit = 1
    for i in range(2**num_qubits):
        if i == bit:
            statevector[i] = 1 + 0j
            bit *= 2
        else:
            statevector[i] = 0 + 0j

    statevector /= np.sqrt(sum(statevector))

    return statevector
