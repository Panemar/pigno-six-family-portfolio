# Ruta R6

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

## Definición congelada

```json
{
  "id": "R6_LOAD_DEPENDENT_RITZ_KRYLOV",
  "name": "load-dependent Ritz/Krylov residual",
  "core": "inner-fold load-conditioned Ritz/SOAR basis plus second-order propagator and graph residual",
  "physics": "second-order projection, force directions and residual flexibility",
  "representation_repair": "basis enrichment or POD-residual hybrid",
  "optimization_repair": "rank/regularization selection then staged residual fit"
}
```

## Representación, física y control

| route_id | canonical_name | spatial_representation | temporal_representation | physics_mechanism | primary_state | mandatory_data_only_control | representation_repair | optimization_repair | capacity_exit_condition | not_a_new_family_modules |
|---|---|---|---|---|---|---|---|---|---|---|
| R6_LOAD_DEPENDENT_RITZ_KRYLOV | Load-dependent Ritz/Krylov residual | load-conditioned ROM basis plus local graph residual | reduced second-order propagator | Ritz/Krylov/SOAR moment matching; residual flexibility | q primary; v/a through compatible propagator/heads | fixed modal/POD ROM plus same residual capacity | basis enrichment or hybrid POD-residual repair | rank/regularization or staged residual optimization | oracle floor, force projection and stable rollout pass | Craig-Bampton only if partition justified |

## Panel factorial S8

| route | family | hard_seed_count | primary_seed_count | velocity_seed_count | strict_noninferiority_seed_count | worst_pooled_over_seeds | worst_P90_over_seeds | worst_case_over_seeds | worst_physical_residual_ratio_over_seeds | parameters | rank | promotion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R6 | LOAD_DEPENDENT_RITZ_KRYLOV | 2 | 2 | 0 | 0 | 0.08734629126914625 | 0.12166024933918489 | 0.1421905191941265 | 0.0649448040114765 | 266081 | 3 | PROMOTE_TO_S9_BOUNDED_HPO |

## Búsqueda multifidelidad S9

| trial_id | route | noninferior_folds | physical_ratio_worst |
|---|---|---|---|
| R6_LHS_04 | R6 | 1 | 0.0038723897966941876 |

## OOF S10

| trial_id | route | eligible_for_S11 | noninferior_to_B2_all_axes | noninferior_to_capacity_matched_control_all_axes | predictive_material_gain | physical_material_gain | bootstrap_positive_axes | median_equilibrium_residual_reduction |
|---|---|---|---|---|---|---|---|---|
| R6_LHS_04 | R6 | False | False | False | False | True | 0 | 0.9961606875214561 |

## Puertas alcanzadas

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R6 | R6_LOAD_DEPENDENT_RITZ_KRYLOV | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |

No se infiere éxito de etapas no alcanzadas. Las reparaciones dirigidas conservan la identidad de la familia y su control data-only.
