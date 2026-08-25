# Six-family physics-informed operator portfolio for railway-bridge dynamics

This repository packages the code, compact scientific data, provenance, metrics, figures, tables, and logs from the final six-family operator campaign for a dynamic railway-bridge finite-element reference.

## Compared families

| ID | Family | Main structural idea |
|---|---|---|
| R1 | Bridge-PINO | Causal temporal spectral operator with bridge-state outputs |
| R2 | MO-PIGNO | Multiple-input graph neural operator |
| R3 | Graph Neural Galerkin | Graph operator with a compatible weak-form term |
| R4 | port-Hamiltonian OpInf | Reduced port-Hamiltonian operator inference and repaired effective model |
| R5 | Rotation-aware multiscale GNO | Multiscale graph messages conditioned by local rotations |
| R6 | Load-dependent Ritz/Krylov | Load-conditioned reduced basis with graph-temporal residual |

The campaign uses one FEM authority, one audited Beam/Timoshenko graph, common grouped splits, common computational budgets, and common metrics. B2 (POD + causal FIR + Ridge) is retained as a comparison reference; it is not a physics-informed family and is not the purpose of the campaign.

## Evidence boundary

- 68 historical physical trajectories.
- Nested grouped cross-validation and out-of-fold (OOF) evidence.
- Same case, saved time, node, component, unit, and axis convention for each comparison.
- The 68 trajectories are historically exposed; this is not a blind or external test.
- No sensor evidence, Rev7/Rev8 data, new FEM simulations, or commercial model files are included.

## Repository contents

- `src/portfolio_operators/`: portable implementations of the six families.
- `families/`: family-level definitions and campaign records.
- `scripts/`: training, audits, metrics, visualization, packaging, and verification code.
- `configs/`: frozen campaign contracts and configurations.
- `tests/`: unit and evidence-contract tests.
- `data/authority/`: compact FEM authority artifacts, exact graph, modal reference, and case-quality table.
- `data/campaign_metadata/`: grouped split and S10 reconstruction/protocol metadata.
- `metrics/`, `tables/`, `figures/`, `reports/`, `logs/`: derived evidence and interpretation artifacts.
- `predictions_oof/`, `checkpoints/`: pointer records for large external binaries.
- `manifests/`: historical artifact registries and provenance.
- `data/external/EXTERNAL_BINARY_MANIFEST.csv`: public, path-sanitized registry of excluded large binaries.

## Quick start

Python 3.12 is the recorded environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest tests\test_portfolio_operators.py -q
.\.venv\Scripts\python scripts\verify_repository.py
```

On Linux or macOS, replace the activation/executable path with `.venv/bin/python`.

## Data policy

Git contains the compact authority, all derived CSV/JSON/Parquet tables supplied by the final package, figures, reports, and logs. Full trajectory HDF5 files, full OOF fields, and large checkpoints are registered by SHA-256 but excluded from ordinary Git. See [DATA.md](DATA.md) and `data/external/EXTERNAL_BINARY_MANIFEST.csv`.

The large source table `tables/figure_data/F27.csv` is stored losslessly as `F27.csv.gz`. Recreate it with:

```powershell
python scripts\materialize_compressed_data.py
```

## Reproducibility and portability

The importable operator package and its unit tests are portable. Numbered historical scripts preserve the original campaign semantics and may contain local path assumptions. See [PORTABILITY.md](PORTABILITY.md) before attempting a full rerun.

Repository integrity is recorded in `MANIFEST.sha256` and checked by `scripts/verify_repository.py`.

## Scientific outputs

The final reports are in `reports/`. Figures F01-F45 and their source tables/captions are included. The repository presents the comparison among the six physics-informed families and the B2 reference under the frozen protocol; it does not reframe the evidence as an external validation.

## License

No public reuse license has been selected. See [LICENSE_PENDING.md](LICENSE_PENDING.md) before making the repository public.
