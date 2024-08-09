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
from statevector_simulator_tensor import Reg


BASE_MAP = {"A" : "00", "T" : "01", "G" : "10", "C" : "11"}
REV_BASE_MAP = dict((reversed(item) for item in base_dict.items()))

def generate_genome(length: int) -> np.ndarray:
    """
    Generates a genome of length 'length', using the nucleobase set of
    {A, T, G, C}.

    Args:
        length (int): number of nucleobase in the genome.

    Returns:
        np.ndarray: the generated genome.
    """

    alphabet = ["A", "T", "G", "C"]

    genome = np.random.choice(alphabet, length)

    return genome


def encode_genome(genome: np.typing.ArrayLike) -> np.ndarray:
    """
    Encodes the given genome into a binary representation where "A" = [0, 0],
        "T" = [0, 1], "G" = [1, 0], and "C" = [1, 1].

    Args:
        genome (np.typing.ArrayLike): genome comprised of nucleobases from {A, T, G, C}.

    Returns:
        np.ndarray: binary encoding of the given genome where "A" = [0, 0],
        "T" = [0, 1], "G" = [1, 0], and "C" = [1, 1].
    """

    pos_qubits = int(np.ceil(np.log2(len(genome))))
    data_qubits = 2

    num_qubits = pos_qubits + data_qubits

    encoded_genome = np.zeros(int(2**num_qubits), dtype=np.complex128)
    for idx, base in enumerate(genome):
        idx_mod = int(bin(idx)[2:].zfill(pos_qubits) + BASE_MAP[base], 2)
        encoded_genome[idx_mod] = 1.0

    return encoded_genome / np.sqrt(np.sum(encoded_genome)), num_qubits


def decode_state(statevector: np.typing.ArrayLike, num_qubits: int, tol = 1e-3) -> str:
    """
    Decodes state vector into a genome, the tolerance parameter is heuristic
    and can be refined.

    Args:
        statevector (np.typing.ArrayLike): normalised quantum state vector to decode.
        num_qubits (int): number of qubits used in the encoding.
        tol (float): tolerance of decoding, by default set to 1e-3

    Returns:
        decoded (str): genome as reconstructed from inverse of encode_genome
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


def create_mps(
    statevector: np.typing.ArrayLike, physical_dim: int, num_nodes: int, bond_dim: int
) -> tuple[list[tn.Node], list[tn.Edge]]:
    """
    Creates an MPS representation of the given statevector.

    Args:
        statevector (np.typing.ArrayLike): arbitrary normalised quantum state of
        the form |\psi\rangle = \sum_{i=0}^{N-1}a_i|i\rangle.
        physical_dim (int): Physical dimension of the given quantum state.
        num_nodes (int): Number of particles in the given quantum state.
        bond_dim (int): Number of singlar values kept in the SVD.

    Returns:
        tuple[list[tn.Node], list[tn.Edge]]: MPS nodes containing tensors for
        the MPS representation of the given quantum state, MPS edges which are
        connected in the generated MPS for ease of truncation.
    """

    if bond_dim == 0:
        bond_dim = 10*10**16

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
    final_node = np.diag(s_temp[:bond_dim]) @ vh_temp[:bond_dim, :]
    final_node /= np.linalg.norm(final_node)
    a_matrices.append(
        tn.Node(final_node.reshape(-1, physical_dim), name=f"Node {num_nodes}")
    )

    # connect edges to build mps
    connected_edges = []
    conn = a_matrices[0][1] ^ a_matrices[1][0]
    connected_edges.append(conn)
    for k in range(1, num_nodes - 1):
        conn = a_matrices[k][-1] ^ a_matrices[k + 1][0]
        connected_edges.append(conn)

    return a_matrices, connected_edges


def analyse_required_bond_dim(
    state_type: str, num_nodes: int, physical_dim: int
) -> None:
    """
    Script to analyse the required bond dimension of a given quantum state such
    that the reproduction is within infidelity of 10^-10.

    Args:
        state_type (str): Type of quantum state to be analysed, currently
        supported states are "genome", "phi", "wstate", "ghz", "gaussian", and
        "random".
        num_nodes (int): Number of particles in the generated quantum state.
        physical_dim (int): Physical dimension of the generated quantum state.
    """

    if state_type == "genome":
        genome = generate_genome(2 ** (num_nodes - 2))
        statevector, num_nodes = encode_genome(genome)
    elif state_type == "phi":
        with open("data/mini_phi.fa", "r") as filehandler:
            genome = filehandler.read().replace("\n", "")
        genome = [
            re.sub(r"[^ATGC]", "", genome)[i % len(genome)]
            for i in range(
                2 ** (num_nodes - 2), (2 ** (num_nodes - 2) + 2 ** (num_nodes - 2))
            )
        ]
        statevector, num_nodes = encode_genome(genome)
    elif state_type == "wstate":
        statevector = np.zeros(2**num_nodes, dtype=np.complex128)
        bit = 1
        for i in range(2**num_nodes):
            if i == bit:
                statevector[i] = 1 + 0j
                bit *= 2
            else:
                statevector[i] = 0 + 0j

        statevector /= np.sqrt(sum(statevector))
    elif state_type == "ghz":
        statevector = np.zeros(2**num_nodes, dtype=np.complex128)
        statevector[0] = 1 / np.sqrt(2)
        statevector[-1] = 1 / np.sqrt(2)
    elif state_type == "gaussian":
        mu, sigma = (2**num_nodes) / 2, (2**num_nodes) / 4

        gaus = np.array([np.sqrt(norm(mu, sigma).pdf(i)) for i in range(2**num_nodes)])
        statevector = gaus / np.linalg.norm(gaus)
    elif state_type == "random":
        statevector = random_statevector(2**num_nodes).data

    bond_dim_diffs = [np.inf]
    bond_dim = 1
    while bond_dim_diffs[-1] > 0.5:
        _, mps_edges = create_mps(statevector, physical_dim, num_nodes, bond_dim)

        for edge in mps_edges:
            mps_tensor = tn.contract(edge)

        bond_dim_diffs.append(
            np.round(
                1
                - np.linalg.norm(
                    np.dot(statevector.conj(), mps_tensor.tensor.flatten())
                )
                ** 2,
                10,
            )
        )
        bond_dim += 1

    bond_dim_diffs.pop(0)

    plt.plot(bond_dim_diffs, "x")
    plt.title(
        f"Reconstruction Error of {state_type} Statevector for given Bond Dimension"
    )
    plt.xlabel("Bond Dimension")
    plt.ylabel("Reconstruction Error")
    plt.savefig(f"results/{state_type}_reconstruction_error.pdf")


def analyse_genome_entropy(physical_dim: int, lengths: np.typing.ArrayLike) -> None:
    """
    Script to analyse the entropy of a given genome such that the reproduction
    is within infidelity of 10^-10.

    Args:
        physical_dim (int): Physical dimension of the generated quantum state.
        lengths (np.typing.ArrayLike): Lengths of the genomes to be generated.
    """

    t1 = time.time()
    bond_dims = []
    entropies = []
    max_bond_dim = []
    num_nodes_for_plot = []
    for length in lengths:
        # genome = generate_genome(length)
        with open("data/mini_phi.fa", "r") as filehandler:
            genome = filehandler.read().replace("\n", "")
        genome = [
            re.sub(r"[^ATGC]", "", genome)[i % len(genome)]
            for i in range(length, (length + length))
        ]
        statevector, num_nodes = encode_genome(genome)

        bond_dim = 1.0
        minimum = 1.0
        maximum = np.inf
        has_decreased = False
        opt_found = False
        while not opt_found:
            if has_decreased:
                bond_dim = np.average([minimum, maximum])
            else:
                bond_dim *= 2
            _, mps_edges = create_mps(
                statevector, physical_dim, num_nodes, int(bond_dim)
            )

            for edge in mps_edges:
                mps_tensor = tn.contract(edge)

            bond_dim_diff = np.round(
                1
                - np.linalg.norm(np.dot(statevector, mps_tensor.tensor.flatten())) ** 2,
                10,
            )

            _, s_temp, _ = np.linalg.svd(mps_tensor.tensor)
            s_temp = s_temp[s_temp > 10e-16]
            entropy = -np.sum(s_temp**2 * np.log(s_temp**2))

            if bond_dim_diff > 0:
                minimum = bond_dim
            else:
                maximum = bond_dim
                has_decreased = True

            if (maximum - minimum) <= 1:
                opt_found = True

        bond_dims.append(bond_dim)
        entropies.append(entropy)
        num_nodes_for_plot.append(num_nodes)
        max_bond_dim.append(num_nodes**physical_dim)
    print(f"Time taken: {time.time() - t1}")

    plt.plot(lengths, bond_dims, "o", label="Required Bond Dimension")
    plt.plot(
        lengths,
        physical_dim ** np.array(num_nodes_for_plot),
        "o",
        label=r"Maximum Bond Dimension $d^{L}$",
    )
    plt.yscale("log")
    plt.title("Genome Bond Dimension Analysis")
    plt.xlabel("Length")
    plt.ylabel("Bond Dimension")
    plt.legend()
    plt.savefig("results/genome_bond_dim_analysis.pdf")

    plt.figure()
    plt.plot(lengths, entropies, "o", label=r"Entropy ($\sum_iS_i^2\log(S_i^2)$)")
    plt.plot(lengths, bond_dims, "o", label="Required Bond dimension")
    plt.title("Genome Required Bond Dimension Compared to Entropy")
    plt.xlabel("Length")
    plt.ylabel("Bond Dimension\Entropy")
    plt.legend()
    plt.savefig("results/bond_dim_entropy_analysis.pdf")


def convert_mps_to_circuit(
    state_type: str, lengths: list[int], physical_dim: int, display_results: bool
) -> None:
    """
    Script to convert a given arbitrary normalised quantum state into an MPS
    representation, and then use that MPS representation to generate a gate
    based quantum circuit.
    Args:
        state_type (str): Type of quantum state to be analysed, currently
        supported states are "genome", "phi", "wstate", "ghz", "gaussian", and
        "random".
        lengths (list[int]): List of number of MPS nodes to be generated for
        comparison.
        physical_dim (int): Physical dimension of the generated quantum states.
    """

    qiskit_times = []
    mps_times = []
    qiskit_gates = []
    mps_gates = []
    entropies = []
    num_nodes_for_plot = []
    for length in lengths:
        print(f"State type: {state_type}, length: {length}")
        if state_type == "genome":
            genome = generate_genome(length)
            statevector, num_nodes = encode_genome(genome)
        elif state_type == "phi":
            with open("data/mini_phi.fa", "r") as filehandler:
                genome = filehandler.read().replace("\n", "")
            genome = [
                re.sub(r"[^ATGC]", "", genome)[i % len(genome)]
                for i in range(length, length + length)
            ]
            statevector, num_nodes = encode_genome(genome)
        elif state_type == "wstate":
            num_nodes = length
            statevector = np.zeros(2**num_nodes, dtype=np.complex128)
            bit = 1
            for i in range(2**num_nodes):
                if i == bit:
                    statevector[i] = 1 + 0j
                    bit *= 2
                else:
                    statevector[i] = 0 + 0j

            statevector /= np.sqrt(sum(statevector))
        elif state_type == "ghz":
            num_nodes = length
            statevector = np.zeros(2**num_nodes, dtype=np.complex128)
            statevector[0] = 1 / np.sqrt(2)
            statevector[-1] = 1 / np.sqrt(2)
        elif state_type == "gaussian":
            num_nodes = length
            mu, sigma = (2**num_nodes) / 2, (2**num_nodes) / 4

            gaus = np.array(
                [np.sqrt(norm(mu, sigma).pdf(i)) for i in range(2**num_nodes)]
            )
            statevector = gaus / np.linalg.norm(gaus)
        elif state_type == "random":
            num_nodes = length
            statevector = random_statevector(2**num_nodes).data

        num_nodes_for_plot.append(num_nodes)

        target = copy.copy(statevector)
        _, mps_edges = create_mps(statevector, physical_dim, num_nodes, 0)

        for edge in mps_edges:
            mps_tensor = tn.contract(edge)

        _, s_temp, _ = np.linalg.svd(mps_tensor.tensor)
        s_temp = s_temp[s_temp > 10e-16]
        entropy = -np.sum(s_temp**2 * np.log(s_temp**2))
        entropies.append(entropy)

        produced_circuit = QuantumCircuit(num_nodes)

        zero_state = np.zeros(statevector.shape[0], dtype=np.complex128)
        zero_state[0] = 1

        t1 = time.time()

        fidelity = 0
        layer_unitaries = []
        while fidelity < 0.99:
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

        mps_times.append(time.time() - t1)

        t1 = time.time()
        qis_circ = QuantumCircuit(num_nodes)
        qis_circ.initialize(target)
        qiskit_times.append(time.time() - t1)

        qis_trans_circ = transpile(
            qis_circ, basis_gates=["x", "sx", "rz", "cx"], optimization_level=3
        )
        if display_results:
            plt.figure()
            plt.bar(range(len(zero_state)), np.abs(zero_state) ** 2, alpha=0.5)
            plt.bar(range(len(target)), np.abs(target) ** 2, alpha=0.5)
            plt.title("Comparison of Target Statevector to Produced Statevector")
            plt.xlabel("Statevector Element")
            plt.ylabel("Probability")

            qis_trans_circ.draw("mpl")
            plt.title("Qiskit")

            produced_circuit.draw("mpl")
            plt.title("MPS")
            plt.show()

        qiskit_gates.append(len(qis_trans_circ))
        mps_gates.append(gates * (num_nodes - 1))

    plt.figure()
    plt.title("Correlation")
    plt.plot(entropies, label="State Entropy", alpha=0.5)
    plt.plot(mps_gates, label="Required Gates", alpha=0.5)
    if state_type in ("phi", "genome"):
        plt.xlabel("Genome Length")
    else:
        plt.xlabel("Number of Qubits")
    plt.ylabel("Entropy/Gates")
    plt.legend()
    plt.savefig(f"results/{state_type}_entropy_gates_correlation.pdf")

    plt.figure()
    plt.plot(mps_gates, entropies, "o", alpha=0.5)
    plt.title("Comparison")
    plt.xlabel("Required Gates")
    plt.ylabel("State Entropy")
    plt.savefig(f"results/{state_type}_entropy_gates_comparison.pdf")

    plt.figure()
    plt.title("Correlation")
    plt.plot(np.array(entropies) / np.max(entropies), label="State Entropy", alpha=0.5)
    plt.plot(np.array(mps_gates) / np.max(mps_gates), label="Required Gates", alpha=0.5)
    if state_type in ("phi", "genome"):
        plt.xlabel("Genome Length")
    else:
        plt.xlabel("Number of Qubits")
    plt.ylabel("Normalised Entropy/Gates")
    plt.legend()
    plt.savefig(f"results/{state_type}_normalised_entropy_gates_correlation.pdf")

    plt.figure()
    plt.title("Gates")
    plt.plot(lengths, qiskit_gates, label="qiskit")
    plt.plot(lengths, mps_gates, label="MPS")
    plt.plot(lengths, 2 ** np.array(num_nodes_for_plot), "--", c="tab:red")
    if state_type in ("phi", "genome"):
        plt.xlabel("Genome Length")
    else:
        plt.xlabel("Number of Qubits")
    plt.ylabel("Gates")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"results/{state_type}_gate_comparison.pdf")

    plt.figure()
    plt.title("Times")
    plt.plot(lengths, qiskit_times, label="qiskit")
    plt.plot(lengths, mps_times, label="MPS")
    if state_type in ("phi", "genome"):
        plt.xlabel("Genome Length")
    else:
        plt.xlabel("Number of Qubits")
    plt.ylabel("Seconds")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"results/{state_type}_time_comparison.pdf")

    plt.close("all")


if __name__ == "__main__":

    # analyse_required_bond_dim("genome", 15, 2)
    # analyse_genome_entropy(2, np.random.randint(1, 1001, 100))
    # for target_state_type in ["genome", "phi", "wstate", "gaussian", "ghz", "random"]:
    for target_state_type in ["genome"]:
        if target_state_type in ("phi", "genome"):
            convert_mps_to_circuit(target_state_type, range(5, 20), 2, False)
        else:
            convert_mps_to_circuit(target_state_type, range(2, 11), 2, False)
