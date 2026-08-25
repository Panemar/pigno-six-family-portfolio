# Portability notes

## Portable layer

`src/portfolio_operators` and `tests/test_portfolio_operators.py` form the portable code layer. They depend on Python, NumPy, h5py, and PyTorch, not on the original workstation directory tree.

## Historical pipeline layer

The numbered scripts and frozen JSON contracts are retained as scientific provenance. Some contain absolute paths from the machine on which the campaign was run. They are not silently rewritten because that would alter the exact historical record.

For a full rerun on another machine:

1. materialize the external artifacts listed in `data/external/EXTERNAL_BINARY_MANIFEST.csv`;
2. create a machine-local path mapping outside version control;
3. preserve the frozen grouped folds and case identities;
4. run preflight checks before launching training;
5. write new outputs under a new run identifier rather than overwriting historical evidence.

## Hardware

The recorded environment used Python 3.12 and a CUDA-capable PyTorch build. CPU execution is valid for unit tests and small audits; campaign runtimes and floating-point details can differ across devices.

## Scientific naming

The numerical authority is the FEM reference. “COMSOL” and “FEM” are not treated as independent models. The comparison is between learned operators and the FEM reference.
