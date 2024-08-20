"""
basic_states.py
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

def generate_ghz_state(num_qubits):
    """
    Generates an entangled quantum GHZ (Greenberger-Horne-Zellinger)
    state for a given number ofqubits.

    Parameters:
    -----------
    num_qubits : int
        Qubits for required GHZ state.

    Returns:
    --------
    statevector : np.typing.ArrayLike
        Corresponding state vector to GHZ state.
    """
    statevector = np.zeros(2**num_qubits, dtype=np.complex128)
    statevector[0] = 1 / np.sqrt(2)
    statevector[-1] = 1 / np.sqrt(2)

    return statevector

def generate_gaussian_state(num_qubits):
    """
    Generates a Gaussian state for a given number of qubits.

    Parameters:
    -----------
    num_qubits : int
        Qubits for required Gaussian state.

    Returns:
    --------
    statevector : np.typing.ArrayLike
        Corresponding state vector to Gaussian state.
    """
    mu, sigma = (2**num_qubits) / 2, (2**num_qubits) / 4

    gaus = np.array(
        [np.sqrt(norm(mu, sigma).pdf(i)) for i in range(2**num_qubits)]
    )
    statevector = gaus / np.linalg.norm(gaus)

    return statevector

def generate_random_state(num_qubits):
    """
    Generates a random state for a given number of qubits (currently wrapping
    around QisKit).

    Parameters:
    -----------
    num_qubits : int
        Qubits for required Gaussian state.

    Returns:
    --------
    statevector : np.typing.ArrayLike
        Corresponding random state vector.
    """
    return random_statevector(2**num_qubits).data
