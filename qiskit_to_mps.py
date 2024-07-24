import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

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

d = 2
genome = generate_genome(3)
target, num_qubits = encode_genome(genome)

target = 1 / np.sqrt(5) * np.array([0, 1, 1, 0, 1, 1, 0, 1], dtype=np.complex128)
num_qubits = 3

print(target, num_qubits)

# Construct quantum circuit
circ = QuantumCircuit(num_qubits, num_qubits)

circ.initialize(target)

# Select the AerSimulator from the Aer provider
simulator = AerSimulator(method='matrix_product_state')

# Define a snapshot that shows the current state vector
circ.save_statevector(label='my_sv')
circ.save_matrix_product_state(label='my_mps')
circ.measure([0,1], [0,1])

# Execute and get saved data
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()
data = result.data(0)

import pdb
pdb.set_trace()

As = []
for i in range(len(data['my_mps']) - 1):
    A = np.eye(2, dtype=np.complex128)
    for gamma in data['my_mps'][i * 2]:
        print(np.vstack(gamma).reshape(2, -1))
        A @= np.vstack(gamma).reshape(2, -1)
    As.append(A)

#print the result data
# A = np.vstack(data['my_mps'][0][0])
print(As)