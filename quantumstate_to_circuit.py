"""
_summary_

_extended_summary_

Returns:
    _type_: _description_
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
from qiskit.quantum_info import Statevector, random_statevector
from qiskit.compiler import transpile
from statevector_simulator_tensor import Reg
from scipy.stats import norm


def generate_genome(length):
    """
    Generates a genome of length 'length', using the nucleobase set of
    {A, T, G, C}.

    Args:
        length (int): number of nucleobase in the genome.

    Returns:
        np.array: the generated genome.
    """

    alphabet = ["A", "T", "G", "C"]

    genome = np.random.choice(alphabet, length)

    return genome


def encode_genome(genome):
    """
    _summary_

    _extended_summary_

    Args:
        genome (_type_): _description_
    """

    pos_qubits = int(np.ceil(np.log2(len(genome))))
    data_qubits = 2

    num_qubits = pos_qubits + data_qubits

    encoded_genome = np.zeros(int(2**num_qubits), dtype=np.complex128)
    for i, base in enumerate(genome):
        if base == "A":
            encoded_genome[int(bin(i)[2:].zfill(pos_qubits) + "00", 2)] = 1.0
        if base == "T":
            encoded_genome[int(bin(i)[2:].zfill(pos_qubits) + "01", 2)] = 1.0
        if base == "G":
            encoded_genome[int(bin(i)[2:].zfill(pos_qubits) + "10", 2)] = 1.0
        if base == "C":
            encoded_genome[int(bin(i)[2:].zfill(pos_qubits) + "11", 2)] = 1.0

    return encoded_genome / np.sqrt(np.sum(encoded_genome)), num_qubits


def create_mps(statevector, d, L, bond_dim=2):
    """
    _summary_

    _extended_summary_

    Args:
        statevector (_type_): _description_
        d (_type_): _description_
        L (_type_): _description_
        bond_dim (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    if bond_dim == 0:
        bond_dim = 10**16

    As = []
    rs = []
    Utemp, Stemp, Vhtemp = np.linalg.svd(
        statevector.reshape(d, -1), full_matrices=False
    )
    rs.append(np.linalg.matrix_rank(Utemp[:, :bond_dim]))
    As.append(tn.Node(Utemp[:, :bond_dim].reshape(d, -1), name="Node 1"))
    statevector = np.diag(Stemp[:bond_dim]) @ Vhtemp[:bond_dim, :]
    for l in range(2, L):
        Utemp, Stemp, Vhtemp = np.linalg.svd(
            statevector.reshape(rs[-1] * d, -1), full_matrices=False
        )
        rs.append(np.linalg.matrix_rank(Utemp[:, :bond_dim]))
        As.append(
            tn.Node(Utemp[:, :bond_dim].reshape(rs[-2], d, rs[-1]), name=f"Node {l}")
        )
        statevector = np.diag(Stemp[:bond_dim]) @ Vhtemp[:bond_dim, :]
    final_node = np.diag(Stemp[:bond_dim]) @ Vhtemp[:bond_dim, :]
    final_node /= np.linalg.norm(final_node)
    As.append(tn.Node(final_node.reshape(-1, d), name=f"Node {L}"))

    # connect edges to build mps
    connected_edges = []
    conn = As[0][1] ^ As[1][0]
    connected_edges.append(conn)
    for k in range(1, L - 1):
        conn = As[k][-1] ^ As[k + 1][0]
        connected_edges.append(conn)

    return As, connected_edges


def analyse_phi_mps():
    """
    _summary_

    _extended_summary_
    """

    d = 2
    with open("data/mini_phi.fa", "r") as filehandler:
        genome = filehandler.read().replace("\n", "")
    genome = re.sub(r"[^ATGC]", "", genome)
    statevector, L = encode_genome(genome)

    bond_dim_diffs = [np.inf]
    bond_dim = 1
    while bond_dim_diffs[-1] > 0:
        mps_nodes, mps_edges = create_mps(statevector, d, L, bond_dim)

        for edge in mps_edges:
            A = tn.contract(edge)

        bond_dim_diffs.append(
            np.round(
                1 - np.linalg.norm(np.dot(statevector, A.tensor.flatten())) ** 2, 10
            )
        )
        bond_dim += 1

    bond_dim_diffs.pop(0)

    plt.plot(bond_dim_diffs, "x")
    plt.title(r"Reconstruction Error of $\Phi X174$ for given Bond Dimension")
    plt.xlabel("Bond Dimension")
    plt.ylabel("Reconstruction Error")
    plt.savefig("results/phix174_reconstruction_error.pdf")


def analyse_genome_entropy():
    """
    _summary_

    _extended_summary_
    """

    t1 = time.time()
    d = 2
    lengths = np.random.randint(1, 1001, 100)
    bond_dims = []
    entropies = []
    max_bond_dim = []
    Ls = []
    for length in lengths:
        # genome = generate_genome(length)
        with open("data/mini_phi.fa", "r") as filehandler:
            genome = filehandler.read().replace("\n", "")
        genome = [
            re.sub(r"[^ATGC]", "", genome)[i % len(genome)]
            for i in range(length, (length + length))
        ]
        statevector, L = encode_genome(genome)

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
            mps_nodes, mps_edges = create_mps(statevector, d, L, int(bond_dim))

            for k in range(len(mps_edges)):
                A = tn.contract(mps_edges[k])

            bond_dim_diff = np.round(
                1 - np.linalg.norm(np.dot(statevector, A.tensor.flatten())) ** 2, 10
            )

            U, S, Vh = np.linalg.svd(A.tensor)
            entropy = -np.sum(S**2 * np.log(S**2))

            if bond_dim_diff > 0:
                minimum = bond_dim
            else:
                maximum = bond_dim
                has_decreased = True

            if (maximum - minimum) <= 1:
                opt_found = True

        bond_dims.append(bond_dim)
        entropies.append(entropy)
        Ls.append(L)
        max_bond_dim.append(L**d)
    print(f"Time take: {time.time() - t1}")

    plt.plot(lengths, bond_dims, "o", label="Required Bond Dimension")
    plt.plot(lengths, d ** np.array(Ls), "o", label=r"Maximum Bond Dimension $d^{L}$")
    plt.yscale('log')
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


def convert_mps_to_circuit(state_type, lengths):
    """
    _summary_

    _extended_summary_
    """

    d = 2
    qiskit_times = []
    mps_times = []
    qiskit_gates = []
    mps_gates = []
    for length in lengths:
        print(f"Genome Length: {length}")
        if state_type == "genome":
            genome = generate_genome(length)
            statevector, L = encode_genome(genome)
            print(L)
        elif state_type == "wstate":
            L = length
            statevector = np.zeros(2 ** L, dtype=np.complex128)
            bit = 1
            for i in range(2 ** L):
                if i == bit:
                    statevector[i] = 1 + 0j
                    bit *= 2
                else:
                    statevector[i] = 0 + 0j

            statevector /= np.sqrt(sum(statevector))
        elif state_type == "ghz":
            L = length
            statevector = np.zeros(2**L, dtype=np.complex128)
            statevector[0] = 1/np.sqrt(2)
            statevector[-1] = 1/np.sqrt(2)
        elif state_type == "gaussian":
            L = length
            mu, sigma = (2 ** L) / 2, (2 ** L) / 4

            gaus = np.array(
                [np.sqrt(norm(mu, sigma).pdf(i)) for i in range(2 ** L)]
            )
            statevector = gaus / np.linalg.norm(gaus)
        elif state_type == "random":
            L = length
            statevector = random_statevector(2 ** L).data

        target = copy.copy(statevector)
        psi_0, mps_edges = create_mps(statevector, d, L, 0)
            
        # for k in range(len(mps_edges)):
        #     A = tn.contract(mps_edges[k])

        # U, S, Vh = np.linalg.svd(A.tensor)
        # entropy = -np.sum(S**2 * np.log(S**2))
        # print(f"Number of Qubits: {L}")
        # print(f"Entropy of state: {entropy}")

        produced_circuit = QuantumCircuit(L)

        zero_state = np.zeros(statevector.shape[0], dtype=np.complex128)
        zero_state[0] = 1

        t1 = time.time()

        fidelity = 0
        layer_unitaries = []
        while fidelity < 0.999:
            unitaries = []
            mps_nodes, mps_edges = create_mps(statevector, d, L, d)

            circuit = QuantumCircuit(L)
            v1 = mps_nodes[L - 1].tensor.reshape(4, -1)
            v2 = sp.linalg.null_space(v1.conj().T)
            U = np.concatenate([v1, v2], axis=1)
            circuit.unitary(U, [0, 1])
            unitaries.append(U)


            for i in range(1, L - 1)[::-1]:
                v1 = mps_nodes[i].tensor.reshape(4, -1)
                v2 = sp.linalg.null_space(v1.conj().T)
                U = np.concatenate([v1, v2], axis=1)
                circuit.unitary(U, [(L - i) - 1, (L - i)])
                unitaries.append(U)

            circuit.unitary(mps_nodes[0].tensor, [L-1])
            unitaries.append(mps_nodes[0].tensor)

            produced_circuit.compose(circuit, range(L), inplace=True)
            layer_unitaries.append(Operator(circuit).data)

            statevector = (Operator(circuit.inverse()).data @ statevector)

            fidelity = np.abs(np.dot(zero_state.conj(), statevector)) ** 2

        for unitary in layer_unitaries[::-1]:
            zero_state @= unitary.T
        
        # print(f"Time taken for MPS: {time.time() - t1}")
        mps_times.append(time.time() - t1)

        t1 = time.time()
        qis_circ = QuantumCircuit(L)
        qis_circ.initialize(target)
        # print(f"Time taken for qiskit: {time.time() - t1}")
        qiskit_times.append(time.time() - t1)

        qis_trans_circ = transpile(qis_circ, basis_gates=["x", "sx", "rz", "cx"], optimization_level=3)
        # qis_trans_circ.draw('mpl')
        # plt.title("Qiskit")

        # plt.figure()
        # plt.bar(range(len(zero_state)), np.abs(zero_state)**2, alpha=0.5)
        # plt.bar(range(len(target)), np.abs(target)**2, alpha=0.5)

        # produced_circuit.draw('mpl')
        trans_circ = transpile(circuit, basis_gates=["x", "sx", "rz", "cx"], optimization_level=3)
        # trans_circ.draw('mpl')
        # plt.title("MPS")
        # plt.show()

        # print(f"Number of gates qiskit: {len(qis_trans_circ)}")
        qiskit_gates.append(len(qis_trans_circ))
        # print(f"Number of gates MPS: {len(trans_circ)}")
        mps_gates.append(len(trans_circ))
    
    print(mps_gates)
    import pdb
    pdb.set_trace()

    # plt.figure()
    # plt.title("Gates")
    # plt.plot(lengths, qiskit_gates, label="qiskit")
    # plt.plot(lengths, mps_gates, label="MPS")
    # plt.xlabel("Qubits")
    # plt.ylabel("Gates")
    # plt.yscale("log")
    # plt.legend()
    # plt.savefig(f"results/{state_type}_gate_comparison.pdf")

    # plt.figure()
    # plt.title("Times")
    # plt.plot(lengths, qiskit_times, label="qiskit")
    # plt.plot(lengths, mps_times, label="MPS")
    # plt.xlabel("Qubits")
    # plt.ylabel("Seconds")
    # plt.yscale("log")
    # plt.legend()
    # plt.savefig(f"results/{state_type}_time_comparison.pdf")



if __name__ == "__main__":

    # analyse_phi_mps()
    # analyse_genome_entropy()
    convert_mps_to_circuit("genome", range(700, 701))
