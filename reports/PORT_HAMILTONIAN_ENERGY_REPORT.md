# Ruta R4

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

## Definición congelada

```json
{
  "id": "R4_PORT_HAMILTONIAN_OPINF",
  "name": "port-Hamiltonian OpInf",
  "core": "constrained reduced input-output dynamics plus zero-initialized graph residual",
  "physics": "J skew, R positive semidefinite, explicit energy and passivity balance",
  "representation_repair": "energy-coordinate or state scaling",
  "optimization_repair": "constrained linear solve followed by staged AdamW residual"
}
```

## Representación, física y control

| route_id | canonical_name | spatial_representation | temporal_representation | physics_mechanism | primary_state | mandatory_data_only_control | representation_repair | optimization_repair | capacity_exit_condition | not_a_new_family_modules |
|---|---|---|---|---|---|---|---|---|---|---|
| R4_PORT_HAMILTONIAN_OPINF | port-Hamiltonian OpInf | reduced physical state with graph residual | passive state-space flow | J skew; R positive semidefinite; energy balance; input port | reduced q,p then decoded fields | unconstrained OpInf/state-space control | state scaling or energy-coordinate repair | stable constrained solve then AdamW residual | FEM energy floor and stable forced rollout pass | energy-consistent neural residual |

## Panel factorial S8

| route | family | hard_seed_count | primary_seed_count | velocity_seed_count | strict_noninferiority_seed_count | worst_pooled_over_seeds | worst_P90_over_seeds | worst_case_over_seeds | worst_physical_residual_ratio_over_seeds | parameters | rank | promotion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R4 | PORT_HAMILTONIAN_OPINF | 2 | 2 | 0 | 0 | 0.08623590110952234 | 0.11075536875833572 | 0.12860599371419368 | 0.0064740394492456355 | 283792 | 2 | PROMOTE_TO_S9_BOUNDED_HPO |

## Búsqueda multifidelidad S9

| trial_id | route | noninferior_folds | physical_ratio_worst |
|---|---|---|---|
| R4_LHS_03 | R4 | 4 | 0.013549329668132616 |

## OOF S10

| trial_id | route | eligible_for_S11 | noninferior_to_B2_all_axes | noninferior_to_capacity_matched_control_all_axes | predictive_material_gain | physical_material_gain | bootstrap_positive_axes | median_equilibrium_residual_reduction |
|---|---|---|---|---|---|---|---|---|
| R4_LHS_03 | R4 | False | False | True | True | True | 1 | 0.9852573236016215 |

## Puertas alcanzadas

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R4 | R4_PORT_HAMILTONIAN_OPINF | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | FAIL | PASS | PASS | PASS_FUNCTIONAL | REPORTED_COMMON_AND_PROJECTED | FAIL_LIMITED | FAIL |

No se infiere éxito de etapas no alcanzadas. Las reparaciones dirigidas conservan la identidad de la familia y su control data-only.
