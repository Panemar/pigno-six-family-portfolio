# Data contract

## Included in Git

The repository includes compact artifacts required to inspect the campaign:

- exact audited structural graph (`data/authority/original_exact_timoshenko_graph.npz`);
- compact modal reference (`data/authority/modal_reference_original.h5`);
- FEM authority manifest and case-quality checks;
- grouped split assignments and fold definitions;
- campaign protocols, configurations, run registries, audit outputs, metrics, tables, logs, figures, and reports;
- losslessly compressed source data for figure F27.

These files support audit, result reproduction from stored derived data, figure inspection, code review, and low-cost operator tests.

## Registered outside Git

The following classes are too large for an ordinary Git repository and remain external:

- the complete 68-trajectory HDF5 dataset;
- fold-specific representation HDF5 files;
- full-resolution OOF prediction fields;
- aggregated OOF fields;
- checkpoints registered by the final campaign.

`data/external/EXTERNAL_BINARY_MANIFEST.csv` is the authoritative public registry. Each row records a logical path, SHA-256 when available, byte size, role, and recommended storage. Absolute workstation paths are not published.

Suitable publication backends are a DOI-bearing scientific repository (for frozen data releases), an institutional repository, or Git LFS when its quota is explicitly managed. GitHub Releases alone is not a substitute for a stable scientific data archive.

## Evidence label

The 68 trajectories are historically exposed and are evaluated with nested grouped cross-validation/OOF evidence. They are not new data and are not a blind test.

## Axes and comparisons

Project axes are transverse X, vertical Y, and longitudinal Z. Every reported supervised comparison must preserve case, saved time, node, component, unit, and response definition.

## F27 materialization

`tables/figure_data/F27.csv.gz` is a lossless Git-friendly representation of the original CSV. `data/external/COMPRESSED_DATA_MANIFEST.json` stores source and compressed hashes. `scripts/materialize_compressed_data.py` reconstructs and verifies the CSV.
