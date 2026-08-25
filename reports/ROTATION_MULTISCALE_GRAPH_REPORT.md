# Ruta R5

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

## Definición congelada

```json
{
  "id": "R5_ROTATION_MULTISCALE_GNO",
  "name": "rotation-aware multiscale GNO",
  "core": "Beam local-frame polar/axial messages with frozen fine-coarse-fine hierarchy",
  "physics": "directional Timoshenko properties, frame covariance and parity-aware rotations",
  "representation_repair": "hierarchy/coarsening or typed equivariant channels",
  "optimization_repair": "zero-start residual gating or layerwise schedule"
}
```

## Representación, física y control

| route_id | canonical_name | spatial_representation | temporal_representation | physics_mechanism | primary_state | mandatory_data_only_control | representation_repair | optimization_repair | capacity_exit_condition | not_a_new_family_modules |
|---|---|---|---|---|---|---|---|---|---|---|
| R5_ROTATION_MULTISCALE_GNO | Rotation-aware multiscale GNO | active Beam graph with local frames and hierarchy | multiscale causal message evolution | polar translations; axial rotations; local directional mechanics | six-DOF q and task-specific rates | same hierarchy with neutralized mechanics | hierarchy/coarsening or equivariant-channel repair | residual gating or layerwise schedule | orientation/invariance tests plus one-case and micropanel capacity | EGNN/MGKN/AMG mechanisms |

## Panel factorial S8

| route | family | hard_seed_count | primary_seed_count | velocity_seed_count | strict_noninferiority_seed_count | worst_pooled_over_seeds | worst_P90_over_seeds | worst_case_over_seeds | worst_physical_residual_ratio_over_seeds | parameters | rank | promotion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R5 | ROTATION_MULTISCALE_GNO | 2 | 2 | 0 | 0 | 0.09923924093992784 | 0.133738707804688 | 0.15562772186677493 | 0.10138571863431231 | 261045 | 5 | RETAIN_AS_S8_NEGATIVE_COMPARATOR |

## Búsqueda multifidelidad S9

No hay filas admitidas.

## OOF S10

No hay filas admitidas.

## Puertas alcanzadas

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R5 | R5_ROTATION_MULTISCALE_GNO | EXECUTED_WITH_REPAIRS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |

No se infiere éxito de etapas no alcanzadas. Las reparaciones dirigidas conservan la identidad de la familia y su control data-only.
