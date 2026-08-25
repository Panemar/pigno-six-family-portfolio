# Calibración hiperparamétrica

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

La criba y promoción multifidelidad conservan todos los ensayos, configuraciones, seeds, folds, stop reasons y fallos. S10/S11 no reutilizan objetivos externos para retunar el espacio de búsqueda.

## Ranking auditado S9

| trial_id | route | noninferior_folds | physical_ratio_worst |
|---|---|---|---|
| R4_LHS_03 | R4 | 4 | 0.013549329668132616 |
| R2_LHS_02 | R2 | 1 | 0.08081256115806888 |
| R6_LHS_04 | R6 | 1 | 0.0038723897966941876 |
| R1_LHS_07 | R1 | 1 | 0.1724743404272367 |

## Promoción OOF

| trial_id | route | eligible_for_S11 | noninferior_to_B2_all_axes | noninferior_to_capacity_matched_control_all_axes | predictive_material_gain | physical_material_gain | bootstrap_positive_axes | median_equilibrium_residual_reduction |
|---|---|---|---|---|---|---|---|---|
| R4_LHS_03 | R4 | False | False | True | True | True | 1 | 0.9852573236016215 |
| R2_LHS_02 | R2 | False | False | False | False | True | 1 | 0.9308924799342415 |
| R6_LHS_04 | R6 | False | False | False | False | True | 0 | 0.9961606875214561 |
