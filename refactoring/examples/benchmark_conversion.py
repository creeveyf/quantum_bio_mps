"""
Benchmarking for a short genome read encoded as MPS and converted to
a quantum circuit with a reconstruction error of 1e-6
"""
from quantum_genome_encoder import *

def main():
    """
    Benchmarking test for random 100-length genome to 1e-6 fidelity
    """
    genome = generate_genome(100)
    tol = 1e-6
    statevector, num_nodes = encode_genome(genome)

    bond_dim = analyse_required_bond_dim(statevector, num_nodes, 2, False, tol)
    print(f"Bond dimension required: {bond_dim}")

    t_start = time.time()
    convert_mps_to_circuit(statevector, num_nodes, 2, False, tol)
    t_end = time.time()
    print(f"Time to decode (produce circuit): {t_end - t_start}s")

if __name__ == """__main__""":
    main()
