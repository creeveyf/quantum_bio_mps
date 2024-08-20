"""
Analyses the entropy of genomes and compares the entropy to the length of the
genomic data, as well as relating to the bond dimension.
"""
from quantum_genome_encoder import *


def analyse_genome_entropy(physical_dim, lengths, read_from_file=None, tol=TOL_DEFAULT_ERR):
    """
    Plots the dependence of the entropy and bond dimension of an MPS representation
    on the length of a given genome such that the representation achieves required
    fidelity.

    Parameters:
    -----------
    physical_dim : int
        Physical dimension of the generated quantum state.
    lengths : np.typing.ArrayLike
        Lengths of the genomes to be generated.
    read_from_file : str
        Path to file to read genome data from if required.
    tol : float
        Default tolerance on reconstruction err, TOL_DEFAULT_ERR by default.
    """

    t1 = time.time()
    bond_dims = []
    entropies = []
    max_bond_dim = []
    num_nodes_for_plot = []

    for length in lengths:
        genome = generate_genome(length)
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
            s_temp = s_temp[s_temp > 1e-16]
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
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(lengths, entropies, "o", label=r"Entropy ($\sum_iS_i^2\log(S_i^2)$)")
    plt.plot(lengths, bond_dims, "o", label="Required Bond dimension")
    plt.title("Genome Required Bond Dimension Compared to Entropy")


if __name__ == "__main__":

    analyse_genome_entropy(2, np.random.randint(1, 1001, 100), read_from_file="../../data/mini_phi.fa")
