# Ruta R1

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

## Definición congelada

```json
{
  "id": "R1_BRIDGE_PINO",
  "name": "Bridge-PINO",
  "core": "multiple-input TFNO/TNO trajectory operator",
  "physics": "compatible BC, spectral, modal or residual terms after FEM-floor audit",
  "representation_repair": "input factorization or continuous/spectral temporal representation",
  "optimization_repair": "normalized losses or causal curriculum"
}
```

## Representación, física y control

| route_id | canonical_name | spatial_representation | temporal_representation | physics_mechanism | primary_state | mandatory_data_only_control | representation_repair | optimization_repair | capacity_exit_condition | not_a_new_family_modules |
|---|---|---|---|---|---|---|---|---|---|---|
| R1_BRIDGE_PINO | Bridge-PINO | observation grid plus audited load/graph descriptors | TFNO/TNO multiple-input trajectory operator | physics-guided initialization plus compatible loss | q first; specialized observation heads allowed | same capacity without physics | continuous-time/Fourier-mode or input-factorization repair | loss normalization or causal curriculum | passes one-case and 3-case micropanel after at most 2 repairs | TFNO; MIONet input branches |

## Panel factorial S8

| route | family | hard_seed_count | primary_seed_count | velocity_seed_count | strict_noninferiority_seed_count | worst_pooled_over_seeds | worst_P90_over_seeds | worst_case_over_seeds | worst_physical_residual_ratio_over_seeds | parameters | rank | promotion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | BRIDGE_PINO | 2 | 2 | 0 | 0 | 0.08261997349207761 | 0.11238323189823934 | 0.13289008667627866 | 0.04667683841859926 | 621216 | 1 | PROMOTE_TO_S9_BOUNDED_HPO |

## Búsqueda multifidelidad S9

| trial_id | route | noninferior_folds | physical_ratio_worst |
|---|---|---|---|
| R1_LHS_07 | R1 | 1 | 0.1724743404272367 |

## OOF S10

No hay filas admitidas.

## Puertas alcanzadas

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | R1_BRIDGE_PINO | EXECUTED_WITH_REPAIRS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |

No se infiere éxito de etapas no alcanzadas. Las reparaciones dirigidas conservan la identidad de la familia y su control data-only.
