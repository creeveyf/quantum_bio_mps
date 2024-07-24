import numpy as np
import scipy as sp
import tensornetwork as tn
import matplotlib.pyplot as plt
import re
import time
import copy
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector
from qiskit.compiler import transpile
from statevector_simulator_tensor import Reg


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


def create_mps_old(statevector, d, L, bond_dim=False):
    rs = []
    As = []
    psi = statevector.reshape(d, -1)
    U, S, Vh = np.linalg.svd(psi, full_matrices=False)
    if bond_dim:
        U = U[:, :bond_dim]
        S = S[:bond_dim]
        Vh = Vh[:bond_dim]
    C = np.matmul(np.diag(S), Vh)
    rs.append(np.linalg.matrix_rank(U))
    As += [tn.Node(U.reshape(d, -1), name="Node 1")]
    for l in range(2, L):
        psi = C.reshape(rs[-1] * d, -1)
        U, S, Vh = np.linalg.svd(psi, full_matrices=False)
        if bond_dim:
            U = U[:, :bond_dim]
            S = S[:bond_dim]
            Vh = Vh[:bond_dim]
        C = np.matmul(np.diag(S), Vh)
        rs.append(np.linalg.matrix_rank(U))
        As += [tn.Node(U.reshape(rs[-2], d, rs[-1]), name=f"Node {l}")]
    As += [tn.Node(C.reshape(d, -1), name=f"Node {L}")]

    # connect edges to build mps
    connected_edges = []
    conn = As[0][1] ^ As[1][0]
    connected_edges.append(conn)
    for k in range(1, L - 1):
        conn = As[k][-1] ^ As[k + 1][0]
        connected_edges.append(conn)

    return As, connected_edges


def create_mps_new(statevector, d, L, bond_dim=False):
    rs = []
    As = []
    psi = statevector.reshape(d, -1)
    Q, C = np.linalg.qr(psi)
    rs.append(np.linalg.matrix_rank(Q))
    As += [tn.Node(Q.reshape(d, -1), name="Node 1")]
    for l in range(2, L):
        psi = C.reshape(rs[-1] * d, -1)
        Q, C = np.linalg.qr(psi)
        rs.append(np.linalg.matrix_rank(Q))
        As += [tn.Node(Q.reshape(rs[-2], d, rs[-1]), name=f"Node {l}")]
    As += [tn.Node(psi.reshape(rs[-1], -1), name=f"Node {L}")]

    # connect edges to build mps
    connected_edges = []
    conn = As[0][1] ^ As[1][0]
    connected_edges.append(conn)
    for k in range(1, L - 1):
        conn = As[k][-1] ^ As[k + 1][0]
        connected_edges.append(conn)

    return As, connected_edges


def analyse_phi_mps():

    d = 2
    genome = generate_genome(10000)
    # with open("mini_phi.fa", "r") as filehandler:
    #     genome = filehandler.read().replace("\n", "")
    # genome = re.sub(r"[^ATGC]", "", genome)
    statevector, L = encode_genome(genome)

    bond_dim_diffs = [np.inf]
    bond_dim = 1
    while bond_dim_diffs[-1] > 0:  # in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        mps_nodes, mps_edges = create_mps(statevector, d, L, bond_dim)

        for k in range(len(mps_edges)):
            A = tn.contract(mps_edges[k])

        bond_dim_diffs.append(
            np.round(
                1 - np.linalg.norm(np.dot(statevector, A.tensor.flatten())) ** 2, 10
            )
        )
        bond_dim += 1

    bond_dim_diffs.pop(0)

    plt.plot(bond_dim_diffs, "x-")
    plt.show()


def analyse_genome_entropy():

    t1 = time.time()
    d = 2
    lengths = np.random.randint(0, 1001, 10)
    # lengths = [16]
    bond_dims = []
    entropies = []
    max_bond_dim = []
    Ls = []
    for length in lengths:
        genome = generate_genome(length)
        # with open("mini_phi.fa", "r") as filehandler:
        #     genome = filehandler.read().replace("\n", "")
        # genome = [re.sub(r"[^ATGC]", "", genome)[i % len(genome)] for i in range(length, (length + length))]
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
            # print(bond_dim)
            mps_nodes, mps_edges = create_mps(statevector, d, L, int(bond_dim))

            for k in range(len(mps_edges)):
                A = tn.contract(mps_edges[k])

            bond_dim_diff = np.round(
                1 - np.linalg.norm(np.dot(statevector, A.tensor.flatten())) ** 2, 10
            )

            U, S, Vh = np.linalg.svd(A.tensor)
            # entropy_1 = -np.sum(
            #     (np.abs(S) ** 2) * np.log(np.abs(S) ** 2)
            # )
            # entropy = -np.sum(S * np.log(S))
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

    # plt.plot(lengths, bond_dims, "o", label="Bond Dimension")
    # plt.plot(lengths, 2 ** np.array(Ls), "o")
    # plt.title("Relationship between genome length, bond dimension, and entropy")
    # plt.xlabel("Length")
    # plt.ylabel("Bond Dimension/Entropy")
    # plt.figure()
    # print(entropies)
    plt.plot(lengths, entropies, "o", label=r"Entropy ($\sum_iS_i^2\log(S_i^2)$)")
    plt.plot(lengths, max_bond_dim, "o", label="Maximum bond dimension")
    # plt.figure()
    # plt.title('Number of parameters comparison')
    # plt.plot(lengths, L * d * np.array(bond_dims)**2, "o", label='Number of parameters')
    # plt.plot(lengths, L * d * np.array(max_bond_dim)**2, "o", label='Maximum number of parameters')
    plt.legend()
    plt.show()
    import pdb

    pdb.set_trace()


def convert_mps_to_circuit():
    d = 2
    # with open("mini_phi.fa", "r") as filehandler:
    #     genome = filehandler.read().replace("\n", "")
    # genome = re.sub(r"[^ATGC]", "", genome)[:10]
    # genome = generate_genome(4)
    # statevector, L = encode_genome(genome)
    # target = copy.copy(statevector)
    statevector = (
        1 / np.sqrt(2) * np.array(([1, 0, 0, 0, 0, 0, 0, 1]), dtype=np.complex128)
    )
    target = 1 / np.sqrt(2) * np.array(([1, 0, 0, 0, 0, 0, 0, 1]), dtype=np.complex128)
    L = 3
    mps_nodes, mps_edges = create_mps(statevector, d, L, 0)
    target_mps_nodes, target_mps_edges = create_mps(target, d, L, 0)
    circuit1 = QuantumCircuit(L)
    circuit3 = Reg(L)
    layers = 1
    fidelity = 0
    unitaries = []
    circ_ops = []
    k = 0
    fidelities = []
    while fidelity < 0.9:

        circuit2 = QuantumCircuit(L)

        mps_nodes, mps_edges = create_mps(statevector, d, L, 2)

        v1 = np.kron(mps_nodes[L - 1].tensor, [1, 0]).reshape(4, -1)
        v2 = sp.linalg.null_space(v1.conj().T)
        idx = np.argwhere(np.all(v1[..., :] == 0, axis=0))
        v1 = np.delete(v1, idx, axis=1)
        U = np.concatenate([v1, v2], axis=1)
        circuit1.unitary(U, [L - 1, L - 2])
        circuit2.unitary(U, [L - 1, L - 2])
        unitaries.append(U)
        new_as = []

        for i in range(1, L - 1)[::-1]:
            v1 = mps_nodes[i].tensor.reshape(4, -1)
            v2 = sp.linalg.null_space(v1.conj().T)
            # idx = np.argwhere(np.all(v1[..., :] == 0, axis=0))
            # v1 = np.delete(v1, idx, axis=1)
            U = np.concatenate([v1, v2], axis=1)
            circuit1.unitary(U, [i, i - 1])
            circuit2.unitary(U, [i, i - 1])
            unitaries.append(U)

        circuit1.unitary(mps_nodes[0].tensor, [0])
        circuit2.unitary(mps_nodes[0].tensor, [0])
        unitaries.append(mps_nodes[0].tensor)

        circuit1.draw("mpl")
        plt.show()

        trial_statevector = Statevector.from_int(0, 2**L).evolve(circuit1).data

        print(np.round(trial_statevector, 10))
        print(np.round(target, 10))

        # print(Operator(circuit2).data.conj().T)

        for k in range(len(mps_edges)):
            A = tn.contract(mps_edges[k])

        # statevector = Operator(circuit2).data.conj().T @ A.tensor.flatten()
        print(statevector)
        fidelity = np.abs(np.dot(target.conj(), statevector)) ** 2
        print(fidelity)

        k += 1

        # # print(target)
        # # print(statevector)
        # import pdb
        # pdb.set_trace()

        # new_mps_nodes = []
        # Us = []
        # S_Vhs = []
        # plt.show()
        # U, S, Vh = np.linalg.svd(unitaries[-2])
        # # Us.append(U)
        # S_Vhs.append((np.diag(S) @ Vh))
        # new_mps_nodes.append(tn.Node(target_mps_nodes[-1].tensor @ unitaries[-1] @ U.reshape(4, 2, 2)))
        # circuit2.unitary(unitaries[-1], [0])
        # for i in range(1, len(target_mps_nodes) - 1):
        #     U, S, Vh = np.linalg.svd(unitaries[len(unitaries) - 1 - i])
        #     if i == 1:
        #         new_mps_nodes.append(tn.Node(target_mps_nodes[i].tensor @ S_Vhs[-1] @ U.reshape(S_Vhs[-1].shape[-1], -1)))
        #     elif i == len(target_mps_nodes) - 2:
        #         new_mps_nodes.append(tn.Node((target_mps_nodes[i].tensor.reshape(-1, S_Vhs[-1].shape[0]) @ S_Vhs[-1] @ U).reshape(4, 2, 2)))
        #     else:
        #         new_mps_nodes.append(tn.Node(target_mps_nodes[i].tensor @ S_Vhs[-1] @ U))
        #     S_Vhs.append(np.diag(S) @ Vh)
        #     circuit2.unitary(unitaries[len(unitaries) - 1 - i], [i, i-1])

        # U, S, Vh = np.linalg.svd(unitaries[0])
        # new_mps_nodes.append(tn.Node(target_mps_nodes[-1].tensor @ S_Vhs[-1].reshape(2, 2, 4)))
        # circuit2.unitary(unitaries[0], [len(target_mps_nodes) - 2, len(target_mps_nodes) - 1])

        # # trans_circ = transpile(
        # #     circuit, basis_gates=["x", "sx", "rz", "cx"], optimization_level=3
        # # )
        # # trans_circ.draw("mpl")
        # # circuit.draw('mpl')
        # # plt.show()

        # statevector = Statevector.from_int(0, 2**L).evolve(circuit1).data
        # # statevector = circuit3.psi.conj().T.flatten()
        # print(f"target: {target}")
        # print(f"statevector:{statevector}")
        # fidelity = np.abs(np.dot(target.conj(), statevector)) ** 2
        # print(fidelity)
        # circuit1.draw('mpl')
        # circuit2.draw('mpl')
        # plt.show()
        # mps_edges = []
        # conn = new_mps_nodes[0][1] ^ new_mps_nodes[1][0]
        # mps_edges.append(conn)
        # for k in range(1, L - 1):
        #     conn = new_mps_nodes[k][-1] ^ new_mps_nodes[k + 1][0]
        #     mps_edges.append(conn)
        # mps_nodes = new_mps_nodes
        # layers += 1
    # print(statevector.data)

    plt.plot(fidelities)
    plt.show()


def truncate_mps(mps, bond_dim):

    new_mps = []
    for node in mps:
        U, S, Vh = np.linalg.svd(node.tensor, full_matrices=False)
        U = U[:, bond_dim]
        S = S[:bond_dim]
        Vh = Vh[bond_dim, :]
        try:
            new_mps.append(tn.Node(np.matmul((U * np.diag(S)), Vh)))
        except:
            import pdb

            pdb.set_trace()
    return new_mps


def convert_mps_to_circuit_new():
    d = 2
    # genome = generate_genome(4)
    # statevector, L = encode_genome(genome)
    statevector = 1 / np.sqrt(4) * np.array([1, 1, 1, 1], dtype=np.complex128)
    target = copy.copy(statevector)
    L = 2
    psi_0, mps_edges = create_mps(statevector, d, L, 0)
    produced_circuit = QuantumCircuit(L)

    zero_state = np.zeros(statevector.shape[0], dtype=np.complex128)
    zero_state[0] = 1

    error = np.inf
    layer_unitaries = []
    while error > 0.01:
        unitaries = []
        mps_nodes, mps_edges = create_mps(statevector, d, L, d)

        circuit = QuantumCircuit(L)
        v1 = mps_nodes[L - 1].tensor.reshape(4, -1)
        v2 = sp.linalg.null_space(v1.conj().T)
        U = np.concatenate([v1, v2], axis=1)
        circuit.unitary(U, [L - 1, L - 2])
        unitaries.append(U)
        new_as = []

        for i in range(1, L - 1)[::-1]:
            v1 = mps_nodes[i].tensor.reshape(4, -1)
            v2 = sp.linalg.null_space(v1.conj().T)
            U = np.concatenate([v1, v2], axis=1)
            circuit.unitary(U, [i, i - 1])
            unitaries.append(U)

        circuit.unitary(mps_nodes[0].tensor, [0])
        unitaries.append(mps_nodes[0].tensor)

        for k in range(len(mps_edges)):
            A = tn.contract(mps_edges[k])
        # Operator(circuit).data.conj().T

        layer_unitaries.append(unitaries)
        circ_unitary = np.eye(L**2, dtype=np.complex128)
        for unitary in unitaries:
            if unitary.shape[0] <= 2:
                circ_unitary @= np.kron(unitary, np.eye(2))
            else:
                circ_unitary @= unitary
        statevector @= circ_unitary.conj().T

        error = 1 - np.abs(np.dot(zero_state.conj(), statevector)) ** 2
        print(error)
    circ_unitary = np.eye(L**2, dtype=np.complex128)
    for layer in layer_unitaries:
        for unitary in layer[::-1]:
            if unitary.shape[0] <= 2:
                circ_unitary @= np.kron(unitary, np.eye(2))
            else:
                circ_unitary @= unitary
    print(np.round(circ_unitary @ zero_state, 10))
    print(target)

    import pdb

    pdb.set_trace()

    print(len(unitaries))
    unitary = iter(unitaries[::-1])
    for layer in range(int(len(unitaries) / L)):
        for qubit in range(L - 1):
            if qubit == 0:
                produced_circuit.unitary(next(unitary).T, [qubit])
                produced_circuit.unitary(next(unitary).T, [qubit, qubit + 1])
            else:
                produced_circuit.unitary(next(unitary).T, [qubit, qubit + 1])

    print(f"Target Statevector: {np.abs(target)**2}")
    print(
        f"Produced Statevector: {np.abs(Statevector.from_int(0, 2**L).evolve(produced_circuit).data)**2}"
    )
    print(
        np.abs(
            np.dot(
                target.conj(),
                Statevector.from_int(0, 2**L).evolve(produced_circuit).data,
            )
        )
        ** 2
    )
    produced_circuit.draw("mpl")
    plt.show()


if __name__ == "__main__":

    # analyse_phi_mps()
    # analyse_genome_entropy()
    convert_mps_to_circuit_new()


# import pdb
# pdb.set_trace()

# #Left-canonical matrix product state
# #-----------------------------------

# #Step 1: reshape statevector with d^L components into a matrix of dimension (d x d^(L-1))
# psi1 = statevector.reshape(d, d**(L-1))
# print(psi1)
# #Step 2: Take an SVD of psi1
# U1, S1, Vh1 = np.linalg.svd(psi1)
# #Step 3: Multiply S1 and Vh1 and reshape to a vector to obtain C1
# C1 = np.matmul(np.diag(S1), Vh1).flatten()
# #Step 4: Obtain the rank, r1, of U1
# r1 = np.linalg.matrix_rank(U1)
# #Step 5: Decompose U1 into a collection of d row vectors A_i1
# As1 = [A for A in U1.reshape(r1, -1)]
# # for i in range(d):
# #     As1.append(U1[:r1])
# #Step 6: Reshape C1 into a matrix psi2 of dimension into a (r1*d x d^{L-2})
# psi2 = C1.reshape(r1*d, d**(L-2))
# #Step 7: Take an SVD of psi2
# U2, S2, Vh2 = np.linalg.svd(psi2)
# #Step 8: Multiply S2 and Vh2 and reshape to a vector to obtain C2
# C2 = np.matmul(np.diag(S2), Vh2).flatten()
# #Step 9: Obtain the rank, r2, of U2
# r2 = np.linalg.matrix_rank(U2)
# #Step 10: Decompose U2 into a collection of d matrices of dimension (r1 x r2)
# As2 = [A for A in U2.reshape(r1, r2, -1)]
# # for i in range(d):
# #     As2.append(U2[i, :])
# #Step 11: Reshape C2 into a matrix psi3 of dimension into a (r2*d x d^{L-3})
# # psi3 = C2.reshape(r2*d, d**(L-3))
