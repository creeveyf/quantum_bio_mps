
"""
Analyses the conversion of MPS to quantum circuits and produces correlation
and benchmarking plots.
"""
from quantum_genome_encoder import *

def analyse_convert_mps_to_circuit(
    state_type: str, lengths: list[int], physical_dim: int, display_results: bool, plot_each_iter: bool=False, fidelity_req: float=1e-6) -> None:
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
        plot_each_iter (bool): Show the plot of the circuit and target vector
        each iteration in main loop, defaults to False.
        fidelity_req (float): fidelity of MPS representation required to original
        vector, defaults to 1e-6
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
            with open("../data/mini_phi.fa", "r") as filehandler:
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

        target = copy.copy(statevector)

        # Benchmark the MPS representation
        num_nodes_for_plot.append(num_nodes)
        t_start = time.time()
        circuit, gates, entropy = convert_mps_to_circuit(statevector, num_nodes, 2, False, fidelity_req)
        t_end = time.time()
        mps_gates.append(gates * (num_nodes - 1))

        mps_times.append(t_end - t_start)
        entropies.append(entropy)

        # Benchmark the Qiskit representation
        t_start = time.time()
        qis_circ = QuantumCircuit(num_nodes)
        qis_circ.initialize(target)
        t_end = time.time()
        qiskit_times.append(t_end - t_start)

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
    plt.savefig(f"../results/{state_type}_entropy_gates_correlation.pdf")

    plt.figure()
    plt.plot(mps_gates, entropies, "o", alpha=0.5)
    plt.title("Comparison")
    plt.xlabel("Required Gates")
    plt.ylabel("State Entropy")
    plt.savefig(f"../results/{state_type}_entropy_gates_comparison.pdf")

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
    plt.savefig(f"../results/{state_type}_normalised_entropy_gates_correlation.pdf")

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
    plt.savefig(f"../results/{state_type}_gate_comparison.pdf")

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
    plt.savefig(f"../results/{state_type}_time_comparison.pdf")

    plt.close("all")

if __name__ == "__main__":

    # analyse_required_bond_dim("genome", 15, 2)
    # analyse_genome_entropy(2, np.random.randint(1, 1001, 100))
    # for target_state_type in ["genome", "phi", "wstate", "gaussian", "ghz", "random"]:
    #for target_state_type in ["genome"]:
    #    if target_state_type in ("phi", "genome"):
    #        convert_mps_to_circuit(target_state_type, range(5, 20), 2, False)
    #    else:
    #        convert_mps_to_circuit(target_state_type, range(2, 11), 2, False)


    # Scaling test for phi-x-174 to 1e-6 fidelity
    analyse_convert_mps_to_circuit("phi", range(50, 60), 2, False)
