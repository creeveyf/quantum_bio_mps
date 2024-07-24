import tensornetwork as tn
import numpy as np
import matplotlib.pyplot as plt


def block(*dimensions):
    """Construct a new matrix for the MPS with random numbers from 0 to 1"""
    size = tuple([x for x in dimensions])
    return np.random.random_sample(size)


def create_MPS(rank, dimension, bond_dimension):
    """Build the MPS tensor"""
    mps = (
        [tn.Node(block(dim, bond_dim), name="Node 0")]
        + [
            tn.Node(block(bond_dim, dim, bond_dim), name=f"Node {i + 1}")
            for i in range(rank - 2)
        ]
        + [tn.Node(block(bond_dim, dim), name=f"Node {rank - 1}")]
    )

    import pdb

    pdb.set_trace()

    # connect edges to build mps
    connected_edges = []
    conn = mps[0][1] ^ mps[1][0]
    connected_edges.append(conn)
    for k in range(1, rank - 1):
        conn = mps[k][2] ^ mps[k + 1][0]
        connected_edges.append(conn)

    return mps, connected_edges


# dimensions = range(2, 9, 2)
# MPS_ranks = range(2, 150)
# MPS_memory = []

# for dim in dimensions:
#     bond_dim = 2
#     MPS_memory = []
#     for i in range(len(MPS_ranks)):
#         rank = MPS_ranks[i]

#         # Creating the MPS state:
#         ##################################################################
#         mps_nodes, mps_edges = create_MPS(rank, dim, bond_dim)
#         MPS_memory.append(np.sum([x.tensor.nbytes for x in mps_nodes]))

#     # Plot Results
#     plt.loglog(MPS_ranks, MPS_memory, "o", ls=":", label=f"d = {dim}")

# plt.legend()
# plt.xlabel("Tensor Rank")
# plt.ylabel("MPS memory")

# plt.show()

########################################################################
# ----- Retrieving a Component from an MPS by Contracting its edges-----#
########################################################################
dim = 2
bond_dim = 2
rank = 4
components = tuple(
    np.random.randint(0, dim, rank)
)  # select randomly the components that we will retrieve
print(f"components are: {components}")

mps_nodes, mps_edges = create_MPS(rank, dim, bond_dim)
for k in range(len(mps_edges)):
    A = tn.contract(mps_edges[k])

# the last node now has all the edges corresponding to the tensor components.

print(f"coefficient of the tensor at the selected components: {A.tensor[components]}")
