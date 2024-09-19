# Quantum Genome Encoder

A Python package for encoding and decoding genomic data as matrix product
states (MPS), and converting to quantum circuits.

This is a proposed restructuring of the repository into a library format
for use on HPC at Sanger.

## Installation

You can install the package using pip:

```bash
pip install .
```

## Examples

See the `examples` directory for examples of implementation of the library
methods and functionality.


## Documentation

The documentation requires `python3` and the `mkdocs` package.

To build the documentation, you can us a virtual environment

```
cd docs
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements_doc.txt
mkdocs serve
```

By default, this will host the webpage on http://127.0.0.1:8000

You can also build the documentation with:

```
mkdocs build
```

which will create a directory called `public` with the build files
that may then be viewed in browser.
