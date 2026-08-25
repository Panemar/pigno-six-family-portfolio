# Comparación de familias physics-informed

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

Las seis familias congeladas se comparan bajo la misma autoridad, split, presupuesto y puertas. 'Not reached' no se reetiqueta como fallo experimental y ninguna mejora física compensa una violación predictiva.

## Formulaciones

| route_id | canonical_name | spatial_representation | temporal_representation | physics_mechanism | primary_state | mandatory_data_only_control |
|---|---|---|---|---|---|---|
| R1_BRIDGE_PINO | Bridge-PINO | observation grid plus audited load/graph descriptors | TFNO/TNO multiple-input trajectory operator | physics-guided initialization plus compatible loss | q first; specialized observation heads allowed | same capacity without physics |
| R2_MO_PIGNO | MO-PIGNO | active Beam graph shared encoder or specialized bases | separate q/v/a operators with defect-aware coupling | BC hard; modal; integral kinematics; compatible weak/tangent physics | q,v,a with rotations in six-DOF state | capacity-matched M0/M1/M2 data-only |
| R3_GRAPH_NEURAL_GALERKIN | Graph Neural Galerkin | elements/subdomains and Beam connectivity | causal latent evolution | virtual work/Galerkin test functions and BC hard | q and optional v | same graph operator without variational residual |
| R4_PORT_HAMILTONIAN_OPINF | port-Hamiltonian OpInf | reduced physical state with graph residual | passive state-space flow | J skew; R positive semidefinite; energy balance; input port | reduced q,p then decoded fields | unconstrained OpInf/state-space control |
| R5_ROTATION_MULTISCALE_GNO | Rotation-aware multiscale GNO | active Beam graph with local frames and hierarchy | multiscale causal message evolution | polar translations; axial rotations; local directional mechanics | six-DOF q and task-specific rates | same hierarchy with neutralized mechanics |
| R6_LOAD_DEPENDENT_RITZ_KRYLOV | Load-dependent Ritz/Krylov residual | load-conditioned ROM basis plus local graph residual | reduced second-order propagator | Ritz/Krylov/SOAR moment matching; residual flexibility | q primary; v/a through compatible propagator/heads | fixed modal/POD ROM plus same residual capacity |

## Evidencia S8

| route | family | primary_seed_count | worst_pooled_over_seeds | worst_P90_over_seeds | worst_case_over_seeds | worst_physical_residual_ratio_over_seeds | parameters | promotion |
|---|---|---|---|---|---|---|---|---|
| R1 | BRIDGE_PINO | 2 | 0.08261997349207761 | 0.11238323189823934 | 0.13289008667627866 | 0.04667683841859926 | 621216 | PROMOTE_TO_S9_BOUNDED_HPO |
| R4 | PORT_HAMILTONIAN_OPINF | 2 | 0.08623590110952234 | 0.11075536875833572 | 0.12860599371419368 | 0.0064740394492456355 | 283792 | PROMOTE_TO_S9_BOUNDED_HPO |
| R6 | LOAD_DEPENDENT_RITZ_KRYLOV | 2 | 0.08734629126914625 | 0.12166024933918489 | 0.1421905191941265 | 0.0649448040114765 | 266081 | PROMOTE_TO_S9_BOUNDED_HPO |
| R2 | MO_PIGNO | 2 | 0.09829802614455775 | 0.1287568805902596 | 0.14757266037628716 | 0.08199323743640147 | 187552 | PROMOTE_TO_S9_BOUNDED_HPO |
| R5 | ROTATION_MULTISCALE_GNO | 2 | 0.09923924093992784 | 0.133738707804688 | 0.15562772186677493 | 0.10138571863431231 | 261045 | RETAIN_AS_S8_NEGATIVE_COMPARATOR |
| R3 | GRAPH_NEURAL_GALERKIN | 1 | 0.10175432732257358 | 0.12779853433438806 | 0.15173010918572774 | 0.054126261796813506 | 244080 | RETAIN_AS_S8_NEGATIVE_COMPARATOR |

## Evidencia S9

| trial_id | route | noninferior_folds | physical_ratio_worst |
|---|---|---|---|
| R4_LHS_03 | R4 | 4 | 0.013549329668132616 |
| R2_LHS_02 | R2 | 1 | 0.08081256115806888 |
| R6_LHS_04 | R6 | 1 | 0.0038723897966941876 |
| R1_LHS_07 | R1 | 1 | 0.1724743404272367 |

## Gates finales

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | R1_BRIDGE_PINO | EXECUTED_WITH_REPAIRS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R2 | R2_MO_PIGNO | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R3 | R3_GRAPH_NEURAL_GALERKIN | EXECUTED_WITH_REPAIRS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R4 | R4_PORT_HAMILTONIAN_OPINF | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | FAIL | PASS | PASS | PASS_FUNCTIONAL | REPORTED_COMMON_AND_PROJECTED | FAIL_LIMITED | FAIL |
| R5 | R5_ROTATION_MULTISCALE_GNO | EXECUTED_WITH_REPAIRS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R6 | R6_LOAD_DEPENDENT_RITZ_KRYLOV | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
