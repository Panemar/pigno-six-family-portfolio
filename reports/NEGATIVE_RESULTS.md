# Resultados negativos

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

Los fallos y rutas no promovidas se conservan como evidencia. No se permite que una reparación cambie de familia ni que un caso aislado cierre una ruta antes de su presupuesto.

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | R1_BRIDGE_PINO | EXECUTED_WITH_REPAIRS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R2 | R2_MO_PIGNO | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R3 | R3_GRAPH_NEURAL_GALERKIN | EXECUTED_WITH_REPAIRS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R4 | R4_PORT_HAMILTONIAN_OPINF | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | FAIL | PASS | PASS | PASS_FUNCTIONAL | REPORTED_COMMON_AND_PROJECTED | FAIL_LIMITED | FAIL |
| R5 | R5_ROTATION_MULTISCALE_GNO | EXECUTED_WITH_REPAIRS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R6 | R6_LOAD_DEPENDENT_RITZ_KRYLOV | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
