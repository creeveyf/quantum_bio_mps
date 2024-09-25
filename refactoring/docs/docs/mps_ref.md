# API Reference - MPS Construction

The MPS construction, conversion, and analysis methods (currently also
in `quantum_genome_encoder.encoding.py`) are documented in this section.

## MPS construction, analysis, and conversion

```
quantum_genome_encoder.create_mps(statevector, physical_dim, num_nodes, bond_dim)
```
::: quantum_genome_encoder.create_mps
    handler: python
---

```
analyse_required_bond_dim(statevector, num_nodes, physical_dim, make_plots=False, tol=TOL_DEFAULT_ERR, descent_param=DESCENT_PARAM)
```
::: quantum_genome_encoder.analyse_required_bond_dim
    handler: python
---

```
convert_mps_to_circuit(statevector, num_nodes, physical_dim, plot_each_iter, tol=TOL_DEFAULT_ERR)
```
::: quantum_genome_encoder.convert_mps_to_circuit
    handler: python
---
