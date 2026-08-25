# Ruta R2

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

## Definición congelada

```json
{
  "id": "R2_MO_PIGNO",
  "name": "MO-PIGNO",
  "core": "shared physical graph with specialized q/v/a operators or heads",
  "physics": "hard BC, modal structure, defect-aware/integral kinematics and compatible weak/tangent terms",
  "representation_repair": "task-specific basis, rank or context",
  "optimization_repair": "GradNorm or PCGrad only after measured conflict"
}
```

## Representación, física y control

| route_id | canonical_name | spatial_representation | temporal_representation | physics_mechanism | primary_state | mandatory_data_only_control | representation_repair | optimization_repair | capacity_exit_condition | not_a_new_family_modules |
|---|---|---|---|---|---|---|---|---|---|---|
| R2_MO_PIGNO | MO-PIGNO | active Beam graph shared encoder or specialized bases | separate q/v/a operators with defect-aware coupling | BC hard; modal; integral kinematics; compatible weak/tangent physics | q,v,a with rotations in six-DOF state | capacity-matched M0/M1/M2 data-only | basis/rank/context specialization | GradNorm or PCGrad after measured conflict | no single-channel closure; primary and physical gates evaluated independently | modal graph; generalized-alpha only if audited |

## Panel factorial S8

| route | family | hard_seed_count | primary_seed_count | velocity_seed_count | strict_noninferiority_seed_count | worst_pooled_over_seeds | worst_P90_over_seeds | worst_case_over_seeds | worst_physical_residual_ratio_over_seeds | parameters | rank | promotion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R2 | MO_PIGNO | 2 | 2 | 0 | 0 | 0.09829802614455775 | 0.1287568805902596 | 0.14757266037628716 | 0.08199323743640147 | 187552 | 4 | PROMOTE_TO_S9_BOUNDED_HPO |

## Búsqueda multifidelidad S9

| trial_id | route | noninferior_folds | physical_ratio_worst |
|---|---|---|---|
| R2_LHS_02 | R2 | 1 | 0.08081256115806888 |

## OOF S10

| trial_id | route | eligible_for_S11 | noninferior_to_B2_all_axes | noninferior_to_capacity_matched_control_all_axes | predictive_material_gain | physical_material_gain | bootstrap_positive_axes | median_equilibrium_residual_reduction |
|---|---|---|---|---|---|---|---|---|
| R2_LHS_02 | R2 | False | False | False | False | True | 1 | 0.9308924799342415 |

## Puertas alcanzadas

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R2 | R2_MO_PIGNO | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |

No se infiere éxito de etapas no alcanzadas. Las reparaciones dirigidas conservan la identidad de la familia y su control data-only.
