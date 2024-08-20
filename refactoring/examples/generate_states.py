"""
Showcases the generation of quantum states making use of the
`basic_states` included with the library.
"""
from quantum_genome_encoder import *

def main():
    for qubits in range(10):
        w_state = generate_w_state(qubits)
        print(f"W state with {qubits} qubits: {w_state}")
        ghz_state = generate_ghz_state(qubits)
        print(f"GHZ state with {qubits} qubits: {ghz_state}")
        gaussian_state = generate_gaussian_state(qubits)
        print(f"Gaussian state with {qubits} qubits: {gaussian_state}")
        random_state = generate_random_state(qubits)
        print(f"Random state with {qubits} qubits: {random_state}")

if __name__ == """__main__""":
    main()
