import numpy as np
import tensornetwork as tn

def encode_genome(genome):
    """
    _summary_

    _extended_summary_

    Args:
        genome (_type_): _description_
    """

    encoded_genome = np.zeros(int(2 ** len(genome)), dtype=np.complex128)
    for i, base in enumerate(genome):
        if base == 'A':
            encoded_genome[int(bin(i)[2:].zfill(int(np.log2(len(genome)))) + '00', 2)] = 1.0
        if base == 'T':
            encoded_genome[int(bin(i)[2:].zfill(int(np.log2(len(genome)))) + '01', 2)] = 1.0
        if base == 'G':
            encoded_genome[int(bin(i)[2:].zfill(int(np.log2(len(genome)))) + '10', 2)] = 1.0
        if base == 'C':
            encoded_genome[int(bin(i)[2:].zfill(int(np.log2(len(genome)))) + '11', 2)] = 1.0
        
    return encoded_genome / np.sqrt(np.sum(encoded_genome))


def create_MPS(psi):
    """
    _summary_

    _extended_summary_

    Args:
        genome (_type_): _description_
    """

    # import pdb
    # pdb.set_trace()

    mps = []
    # psi_matrix = psi.reshape([2] * len(genome))
    temp = psi.reshape(2, 8)
    print(temp)
    for i in range(0, len(genome)-1):
        U, S, Vh = np.linalg.svd(temp, full_matrices=False)
        if i == 0:
            mps += [tn.Node(U.reshape(2, -1), name=f'Node {i}')]
        else:
            mps += [tn.Node(U.reshape(2, 2, -1), name=f'Node {i}')]
        dims = temp.shape
        temp = np.dot(np.diag(S), Vh[:dims[0],:]).reshape(2, 8)
    mps += [tn.Node(U.reshape(-1, 2), name=f'Node {i+1}')]
    
        # import pdb
        # pdb.set_trace()

    #connect edges to build mps
    connected_edges=[]
    conn=mps[0][1]^mps[1][0]
    connected_edges.append(conn)
    for k in range(1, len(genome) - 1):
        # print(len(genome))
        conn=mps[k][1]^mps[k+1][0]
        # print()
        connected_edges.append(conn)
    print(connected_edges)
    return mps, connected_edges



genome = 'AAAA'

psi = encode_genome(genome)

# import pdb
# pdb.set_trace()

mps_nodes, mps_edges = create_MPS(psi)
# for k in range(len(mps_edges)):
#     A = tn.contract(mps_edges[k])

import pdb
pdb.set_trace()


def create_mps(statevector, d, L):
    rs = []
    As = []
    psi = statevector.reshape(d, d**(L-1))
    U, S, Vh = np.linalg.svd(psi)
    C = np.matmul(np.diag(S), Vh).flatten()
    rs.append(np.linalg.matrix_rank(U))
    As.append([A for A in U.reshape(rs + [-1])])
    for l in range(2, L+1):
        psi = C.reshape(rs[-1]*d, d**(L-l))
        U, S, Vh = np.linalg.svd(psi)
        C = np.matmul(np.diag(S), Vh).flatten()
        rs.append(np.linalg.matrix_rank(U2))
        As.append([A for A in U.reshape(rs + [-1])]
