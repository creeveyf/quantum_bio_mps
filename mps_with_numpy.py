import numpy as np
import tensornetwork as tn
import matplotlib.pyplot as plt

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


def block(kmer):
    """

    _extended_summary_

    Args:
        kmer (_type_): _description_
    """
    binary_rep = []
    for nucleobase in kmer:
        if nucleobase == "A":
            binary_rep.append([0, 0])
        elif nucleobase == "T":
            binary_rep.append([0, 1])
        elif nucleobase == "G":
            binary_rep.append([1, 0])
        elif nucleobase == "C":
            binary_rep.append([1, 1])
    
    return np.array(binary_rep)


def create_mps(genome, window_size):
    """
    _summary_

    _extended_summary_

    Args:
        genome (_type_): _description_
        window_size (_type_): _description_
    """
    kmers = [genome[i:i+window_size] for i in range(genome.shape[0] - window_size + 1)]

    mps = (
        [tn.Node(block(kmers[0]))]
        + [
            tn.Node(block(kmer)) for kmer in kmers[1:-1]
        ]
        + [tn.Node(block(kmers[-1]))]
    )

    print(mps)

    # Connect edges to build mps
    connected_edges = []
    conn = mps[0][1] ^ mps[1][0]
    connected_edges.append(conn)
    for k in range(1, window_size - 1):
        conn = mps[k][2] ^ mps[k + 1][0]
        connected_edges.append(conn)

    return mps, connected_edges



if __name__ == "__main__":
    """
    _summary_

    _extended_summary_
    """

    lengths = range(20, 100, 10)

    memory = []
    for length in lengths:
        window_size = 2
        genome = generate_genome(length)
        mps_nodes, mps_edges = create_mps(genome, window_size)
        memory.append(np.sum([x.tensor.nbytes for x in mps_nodes]))

    #Plot Results
    plt.loglog(lengths, memory, 'o', ls=':')

    # plt.legend()
    plt.xlabel('Tensor Rank')
    plt.ylabel('MPS Memory')

    plt.show()