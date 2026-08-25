# Ruta R3

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

## Definición congelada

```json
{
  "id": "R3_GRAPH_NEURAL_GALERKIN",
  "name": "Graph Neural Galerkin",
  "core": "Beam-graph field operator plus elementwise variational assembly",
  "physics": "virtual work in a compatible six-DOF test space and hard essential BC",
  "representation_repair": "test-space or Petrov-Galerkin enrichment",
  "optimization_repair": "term normalization or adaptive quadrature/sampling"
}
```

## Representación, física y control

| route_id | canonical_name | spatial_representation | temporal_representation | physics_mechanism | primary_state | mandatory_data_only_control | representation_repair | optimization_repair | capacity_exit_condition | not_a_new_family_modules |
|---|---|---|---|---|---|---|---|---|---|---|
| R3_GRAPH_NEURAL_GALERKIN | Graph Neural Galerkin | elements/subdomains and Beam connectivity | causal latent evolution | virtual work/Galerkin test functions and BC hard | q and optional v | same graph operator without variational residual | test-space/rank or Petrov-Galerkin repair | normalized weak terms or adaptive quadrature/sampling | FEM weak floor pass plus memorization and multicaso graph utility | Graph Galerkin code; test-space modal basis |

## Panel factorial S8

| route | family | hard_seed_count | primary_seed_count | velocity_seed_count | strict_noninferiority_seed_count | worst_pooled_over_seeds | worst_P90_over_seeds | worst_case_over_seeds | worst_physical_residual_ratio_over_seeds | parameters | rank | promotion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3 | GRAPH_NEURAL_GALERKIN | 2 | 1 | 0 | 0 | 0.10175432732257358 | 0.12779853433438806 | 0.15173010918572774 | 0.054126261796813506 | 244080 | 6 | RETAIN_AS_S8_NEGATIVE_COMPARATOR |

## Búsqueda multifidelidad S9

No hay filas admitidas.

## OOF S10

No hay filas admitidas.

## Puertas alcanzadas

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3 | R3_GRAPH_NEURAL_GALERKIN | EXECUTED_WITH_REPAIRS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |

No se infiere éxito de etapas no alcanzadas. Las reparaciones dirigidas conservan la identidad de la familia y su control data-only.
