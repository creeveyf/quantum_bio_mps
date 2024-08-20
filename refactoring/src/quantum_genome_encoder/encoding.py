"""
This code is intended to create efficient gate based quantum circuits from 
arbitrary quantum states.

The code takes a given normalised arbitrary quantum state, converts it into a 
left canonicalised MPS through a series of reshapes and SVDs, and then converts
that produced MPS into an efficient gate based quantum circuit.

TODO:
    1) Implement splitting two-qubit unitaries generated at each step of circuit
    generation such that they can be directly multiplied into the existing MPS
    nodes, increasing efficiency by removing the need to multiply
    Operator(circuit.inverse()).data @ statevector and also removing the need
    for qiskit from the codebase entirely.

    2) Implement sweep algorithm (DMRG style) to improve efficiency of truncation
    to bond dimension 2 at each layer of circuit generation in
    convert_mps_to_circuit().


Author: Floyd Creevey
Email: fc309@sanger.ac.uk
"""
import re
import time
import copy
import numpy as np
import scipy as sp
import tensornetwork as tn
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import random_statevector
from qiskit.compiler import transpile
from scipy.stats import norm
#from statevector_simulator_tensor import Reg


BASE_MAP = {"A" : "00", "T" : "01", "G" : "10", "C" : "11"}

REV_BASE_MAP = dict((reversed(item) for item in BASE_MAP.items()))

TOL_DEFAULT_ERR = 1e-3

def generate_genome(length):
    """
    Generates a genome of length 'length', using the nucleobase set of
    {A, T, G, C}.

    Parameters:
    -----------
    lenght : int
        Length of required genome.

    Returns:
    --------
    genome : np.ndarray
        An array of nucleobases forming the genome.
    """

    alphabet = ["A", "T", "G", "C"]

    genome = np.random.choice(alphabet, length)

    return genome

def read_genome(file, length):
    """
    Reads length of a genome from given file.

    Parameters:
    ----------
    file : str
        Location of data file where genome is stored.
    length : int
        Length of genome to be read.

    Returns:
    --------
    genome : np.ndarray
        An array of nucleobases forming the genome
    """

    with open(file, "r") as filehandler:
        genome = filehandler.read().replace("\n", "")

    genome = [
        re.sub(r"[^ATGC]", "", genome)[i % len(genome)]
        for i in range(length, length + length)
    ]

    return genome

def encode_genome(genome):
    """
    Encodes the given genome into a binary representation with BASE_MAP

    Parameters:
    ----------
    genome : np.ndarray
        Genome array comprised of nucleobases from {A, T, G, C}.

    Returns:
    -------
    encoded_genome : np.ndarray
        Normalised binary encoding of the given genome.
    num_qubits : int
        Total number of qubits required for encoding.
    """

    pos_qubits = int(np.ceil(np.log2(len(genome))))
    data_qubits = 2

    num_qubits = pos_qubits + data_qubits

    encoded_genome = np.zeros(int(2**num_qubits), dtype=np.complex128)

    for idx, base in enumerate(genome):
        idx_mod = int(bin(idx)[2:].zfill(pos_qubits) + BASE_MAP[base], 2)
        encoded_genome[idx_mod] = 1.0

    return encoded_genome / np.sqrt(np.sum(encoded_genome)), num_qubits


def decode_state(statevector, num_qubits, tol=TOL_DEFAULT_ERR):
    """
    Decodes state vector into a genome, the tolerance parameter is heuristic
    and can be refined.

    Parameters:
    -----------
    statevector : np.typing.ArrayLike
        Normalised quantum state vector to decode.
    num_qubits : int
        Total number of qubits used in the encoding.
    tol : float
        Tolerance of decoding, by default set to TOL_DEFAULT_ERR

    Returns:
    --------
    decoded : str
        Genome as reconstructed from inverse of encode_genome
    """
    decoded = ""

    for idx, element in enumerate(statevector):
        if np.abs(element) > tol:
            # Extract the position revister and base register
            pos = bin(idx)[2:]
            base = pos[-2:].zfill(2)
            # Pad with zeroes if the position is too short
            if len(pos) < 4:
                pos = pos.zfill(4)
            decoded += REV_BASE_MAP[str(base)]

    return decoded


def create_mps(statevector, physical_dim, num_nodes, bond_dim):
    """
    Creates an MPS representation of the given statevector.

    Parameters:
    -----------
    statevector : np.typing.ArrayLike
        Arbitrary normalised quantum state of
    physical_dim : int
        Physical dimension of the given quantum state.
    num_nodes : int
        Number of particles in the given quantum state.
    bond_dim : int
        Number of singlar values kept in the SVD.

    Returns:
    --------
    a_matrices : list[tn.Node]
        MPS nodes containing tensors of the MPS representation
        of the given quantum state.
    connected_edges : list[tn.Edge]
        List of the connected edges in the MPS reresentation.
    """

    if bond_dim == 0:
        bond_dim = 10*10**16

    # First SVD decomposition
    a_matrices = []
    ranks = []
    u_temp, s_temp, vh_temp = np.linalg.svd(
        statevector.reshape(physical_dim, -1), full_matrices=False
    )
    ranks.append(np.linalg.matrix_rank(u_temp[:, :bond_dim]))
    a_matrices.append(
        tn.Node(u_temp[:, :bond_dim].reshape(physical_dim, -1), name="Node 1")
    )
    statevector = np.diag(s_temp[:bond_dim]) @ vh_temp[:bond_dim, :]

    # Subsequent SVD decompositions
    for node in range(2, num_nodes):
        u_temp, s_temp, vh_temp = np.linalg.svd(
            statevector.reshape(ranks[-1] * physical_dim, -1), full_matrices=False
        )
        ranks.append(np.linalg.matrix_rank(u_temp[:, :bond_dim]))
        a_matrices.append(
            tn.Node(
                u_temp[:, :bond_dim].reshape(ranks[-2], physical_dim, ranks[-1]),
                name=f"Node {node}",
            )
        )
        statevector = np.diag(s_temp[:bond_dim]) @ vh_temp[:bond_dim, :]

    # Special treatment for final node
    final_node = np.diag(s_temp[:bond_dim]) @ vh_temp[:bond_dim, :]
    final_node /= np.linalg.norm(final_node)
    a_matrices.append(
        tn.Node(final_node.reshape(-1, physical_dim), name=f"Node {num_nodes}")
    )

    # Connect edges to build mps
    connected_edges = []
    conn = a_matrices[0][1] ^ a_matrices[1][0]
    connected_edges.append(conn)

    for k in range(1, num_nodes - 1):
        conn = a_matrices[k][-1] ^ a_matrices[k + 1][0]
        connected_edges.append(conn)

    return a_matrices, connected_edges


def analyse_required_bond_dim(statevector, num_nodes, physical_dim, make_plots=False, tol=TOL_DEFAULT_ERR):
    """
    Analyse the required bond dimension of a given MPS to reproduce a state vector
    within required fidelity.

    Parameters:
    -----------
    statevector : str
        The quantum state to be analysed.
    num_nodes : int
        Number of particles in the generated quantum state.
    physical_dim : int
        Physical dimension of the generated quantum state.
    make_plots : bool
        Plot reconstruction error against bond dimension.
    tol : float
        Default tolerance on reconstruction err, TOL_DEFAULT_ERR by default.

    Returns:
    -------
    bond_dim : int
        The required bond dimension for the given tolerance.
    """

    bond_dim = 1
    fidelity = 0.

    # Parameter to estimate steps in bond dimension required.
    # Must be > 0.5 otherwise convergence is not guaranteed.
    # Smaller -> slower convergence, smaller overestimate.
    # Larger -> faster convergence, larger overestimate.
    DESCENT_PARAM = 0.75

    if make_plots:
        bond_dim_diffs = []

    while fidelity < 1 - tol:
        _, mps_edges = create_mps(statevector, physical_dim, num_nodes, bond_dim)

        for edge in mps_edges:
            mps_tensor = tn.contract(edge)

        fidelity = np.linalg.norm(np.dot(statevector.conj(), mps_tensor.tensor.flatten()) ** 2)

        if make_plots:
            bond_dim_diffs.append((bond_dim, np.round(1 - fidelity, 10)))

        if 1 - fidelity > tol:
            grad = np.log10( (1 - fidelity) / tol )
            # Use a descent parameter to ensure we do not wildly
            # overestimate bond dimension, will require tuning.
            grad = int(grad * DESCENT_PARAM)
            bond_dim += grad

    if make_plots:
        bond_dim_diffs.pop(0)
        x, y = zip(*bond_dim_diffs)
        plt.plot(x, y, "x")
        plt.title(
            f"Reconstruction error of state vector for given bond dimension"
        )
        plt.xlabel("Bond Dimension")
        plt.ylabel("Reconstruction Error")
        plt.show()

    return bond_dim


def convert_mps_to_circuit(statevector, num_nodes, physical_dim, plot_each_iter, tol=TOL_DEFAULT_ERR):
    """
    Converts a given arbitrary normalised quantum state into an MPS
    representation, and then use that MPS representation to generate a
    gate-based quantum circuit.

    Parameters:
    ----------
    statevector : np.typing.ArrayLike
        List of number of MPS nodes to be generated for comparison.
    num_nodes : int
        Number of nodes in the MPS.
    physical_dim : int
        Physical dimension of the generated quantum states.
    plot_each_iter : bool
        Show the plot of the circuit and target vector each iteration
        in main loop, defaults to False.
    tol : float
        fidelity of MPS representation required to original vector,
        defaults to TOL_DEFAULT_ERROR

    Returns:
    -------
    produced_circuit : qiskit.QuantumCircuit
        Quantum circuit produced from MPS representation.
    gates : int
        Number of gates in the produced quantum circuit.
    entropy : float
        Entropy of given representation.
    """

    target = copy.copy(statevector)
    _, mps_edges = create_mps(statevector, physical_dim, num_nodes, 0)

    for edge in mps_edges:
        mps_tensor = tn.contract(edge)

    _, s_temp, _ = np.linalg.svd(mps_tensor.tensor)
    s_temp = s_temp[s_temp > 1e-16]
    entropy = -np.sum(s_temp**2 * np.log(s_temp**2))

    produced_circuit = QuantumCircuit(num_nodes)

    zero_state = np.zeros(statevector.shape[0], dtype=np.complex128)
    zero_state[0] = 1

    fidelity = 0
    layer_unitaries = []
    validation_list = []
    REQ_SIZE = 15
    iterations = 0 
    while fidelity < 1 - tol:
        # If the last REQ_SIZE sequences decoded have been identical, break.
        # Idea: break when decoded genome is stable, rather than waiting
        # on fidelity.
        # We can tune this parameter for efficiency.
        if len(validation_list) == REQ_SIZE:
            break

        iterations += 1

        unitaries = []
        mps_nodes, mps_edges = create_mps(statevector, physical_dim, num_nodes, 2)

        circuit = QuantumCircuit(num_nodes)
        v1 = mps_nodes[num_nodes - 1].tensor.reshape(4, -1)
        v2 = sp.linalg.null_space(v1.conj().T)
        u_matrix = np.concatenate([v1, v2], axis=1)
        circuit.unitary(u_matrix, [0, 1])
        unitaries.append(u_matrix)

        for i in range(1, num_nodes - 1)[::-1]:
            v1 = mps_nodes[i].tensor.reshape(4, -1)
            v2 = sp.linalg.null_space(v1.conj().T)
            u_matrix = np.concatenate([v1, v2], axis=1)
            circuit.unitary(u_matrix, [(num_nodes - i) - 1, (num_nodes - i)])
            unitaries.append(u_matrix)

        circuit.unitary(mps_nodes[0].tensor, [num_nodes - 1])
        unitaries.append(mps_nodes[0].tensor)

        produced_circuit.compose(circuit, range(num_nodes), inplace=True)
        layer_unitaries.append(Operator(circuit).data)

        statevector = Operator(circuit.inverse()).data @ statevector

        fidelity = np.abs(np.dot(zero_state.conj(), statevector)) ** 2
        trial = copy.copy(zero_state)
        for unitary in layer_unitaries[::-1]:
            trial @= unitary.T

        decoded_seq = decode_state(trial, num_nodes)
        if len(validation_list) == 0:
            validation_list.append(decoded_seq)

        if decoded_seq != validation_list[-1]:
            validation_list.clear()

        validation_list.append(decoded_seq)

        if plot_each_iter:            
            plt.figure()
            plt.bar(range(len(trial)), np.abs(trial)**2, label=r"$|\psi_k\rangle$", alpha=0.5)
            plt.bar(range(len(target)), np.abs(target)**2, label=r"$|\psi_{\rm{target}}\rangle$", alpha=0.5)
            plt.ylabel("Probability")
            plt.legend()

            produced_circuit.draw('mpl')
            plt.show()

    gates = 0
    for unitary in layer_unitaries[::-1]:
        zero_state @= unitary.T
        gates += 1

    # Qiskit benchmarking checks - not required for functionality
    # qis_circ = QuantumCircuit(num_nodes)
    # qis_circ.initialize(target)
    # qis_trans_circ = transpile(
    #     qis_circ, basis_gates=["x", "sx", "rz", "cx"], optimization_level=3
    # )

    return produced_circuit, gates, entropy
