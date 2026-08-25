# Informe final del portafolio PIGNO

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

## Dictamen

`NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`

## Definición de las seis familias

| route_id | canonical_name | spatial_representation | temporal_representation | physics_mechanism | primary_state | mandatory_data_only_control |
|---|---|---|---|---|---|---|
| R1_BRIDGE_PINO | Bridge-PINO | observation grid plus audited load/graph descriptors | TFNO/TNO multiple-input trajectory operator | physics-guided initialization plus compatible loss | q first; specialized observation heads allowed | same capacity without physics |
| R2_MO_PIGNO | MO-PIGNO | active Beam graph shared encoder or specialized bases | separate q/v/a operators with defect-aware coupling | BC hard; modal; integral kinematics; compatible weak/tangent physics | q,v,a with rotations in six-DOF state | capacity-matched M0/M1/M2 data-only |
| R3_GRAPH_NEURAL_GALERKIN | Graph Neural Galerkin | elements/subdomains and Beam connectivity | causal latent evolution | virtual work/Galerkin test functions and BC hard | q and optional v | same graph operator without variational residual |
| R4_PORT_HAMILTONIAN_OPINF | port-Hamiltonian OpInf | reduced physical state with graph residual | passive state-space flow | J skew; R positive semidefinite; energy balance; input port | reduced q,p then decoded fields | unconstrained OpInf/state-space control |
| R5_ROTATION_MULTISCALE_GNO | Rotation-aware multiscale GNO | active Beam graph with local frames and hierarchy | multiscale causal message evolution | polar translations; axial rotations; local directional mechanics | six-DOF q and task-specific rates | same hierarchy with neutralized mechanics |
| R6_LOAD_DEPENDENT_RITZ_KRYLOV | Load-dependent Ritz/Krylov residual | load-conditioned ROM basis plus local graph residual | reduced second-order propagator | Ritz/Krylov/SOAR moment matching; residual flexibility | q primary; v/a through compatible propagator/heads | fixed modal/POD ROM plus same residual capacity |

## Panel factorial S8

| route | family | primary_seed_count | worst_pooled_over_seeds | worst_P90_over_seeds | worst_case_over_seeds | worst_physical_residual_ratio_over_seeds | parameters | promotion |
|---|---|---|---|---|---|---|---|---|
| R1 | BRIDGE_PINO | 2 | 0.08261997349207761 | 0.11238323189823934 | 0.13289008667627866 | 0.04667683841859926 | 621216 | PROMOTE_TO_S9_BOUNDED_HPO |
| R4 | PORT_HAMILTONIAN_OPINF | 2 | 0.08623590110952234 | 0.11075536875833572 | 0.12860599371419368 | 0.0064740394492456355 | 283792 | PROMOTE_TO_S9_BOUNDED_HPO |
| R6 | LOAD_DEPENDENT_RITZ_KRYLOV | 2 | 0.08734629126914625 | 0.12166024933918489 | 0.1421905191941265 | 0.0649448040114765 | 266081 | PROMOTE_TO_S9_BOUNDED_HPO |
| R2 | MO_PIGNO | 2 | 0.09829802614455775 | 0.1287568805902596 | 0.14757266037628716 | 0.08199323743640147 | 187552 | PROMOTE_TO_S9_BOUNDED_HPO |
| R5 | ROTATION_MULTISCALE_GNO | 2 | 0.09923924093992784 | 0.133738707804688 | 0.15562772186677493 | 0.10138571863431231 | 261045 | RETAIN_AS_S8_NEGATIVE_COMPARATOR |
| R3 | GRAPH_NEURAL_GALERKIN | 1 | 0.10175432732257358 | 0.12779853433438806 | 0.15173010918572774 | 0.054126261796813506 | 244080 | RETAIN_AS_S8_NEGATIVE_COMPARATOR |

## Selección S9

| trial_id | route | noninferior_folds | physical_ratio_worst |
|---|---|---|---|
| R4_LHS_03 | R4 | 4 | 0.013549329668132616 |
| R2_LHS_02 | R2 | 1 | 0.08081256115806888 |
| R6_LHS_04 | R6 | 1 | 0.0038723897966941876 |
| R1_LHS_07 | R1 | 1 | 0.1724743404272367 |

## Decisión S10→S11

| trial_id | route | eligible_for_S11 | noninferior_to_B2_all_axes | noninferior_to_capacity_matched_control_all_axes | predictive_material_gain | physical_material_gain | bootstrap_positive_axes | median_equilibrium_residual_reduction |
|---|---|---|---|---|---|---|---|---|
| R4_LHS_03 | R4 | False | False | True | True | True | 1 | 0.9852573236016215 |
| R2_LHS_02 | R2 | False | False | False | False | True | 1 | 0.9308924799342415 |
| R6_LHS_04 | R6 | False | False | False | False | True | 0 | 0.9961606875214561 |

## Evidencia OOF admitida

| trial_id | model | axis | pooled_relative_l2 | mean_relative_l2 | p90_relative_l2 | worst_relative_l2 |
|---|---|---|---|---|---|---|
| R4_LHS_03 | S10_HYBRID | X | 0.0199960605457061 | 0.0615005526941892 | 0.1265087644210564 | 0.3819923878453377 |
| R4_LHS_03 | B2 | X | 0.0181402658269503 | 0.0556123915459219 | 0.1224531596991992 | 0.3819923878453377 |
| R4_LHS_03 | S10_HYBRID | Y | 0.0342411899819672 | 0.0329324214834703 | 0.0428746528416704 | 0.0491593984228299 |
| R4_LHS_03 | B2 | Y | 0.024930806387929 | 0.0236225563443107 | 0.0372969676861317 | 0.0457175689909722 |
| R4_LHS_03 | S10_HYBRID | Z | 0.0288335960039827 | 0.032195385177254 | 0.047362187188576 | 0.0641828557423997 |
| R4_LHS_03 | B2 | Z | 0.0371812033501441 | 0.039953413459361 | 0.0551334748354795 | 0.0641828557423997 |
| R2_LHS_02 | S10_HYBRID | X | 0.0216877912118828 | 0.0641253202693331 | 0.1432208560010132 | 0.3819923878453377 |
| R2_LHS_02 | B2 | X | 0.0181402658269503 | 0.0556123915459219 | 0.1224531596991992 | 0.3819923878453377 |
| R2_LHS_02 | S10_HYBRID | Y | 0.0364933674367093 | 0.0347234972199813 | 0.0473609226208897 | 0.0567179432926939 |
| R2_LHS_02 | B2 | Y | 0.024930806387929 | 0.0236225563443107 | 0.0372969676861317 | 0.0457175689909722 |
| R2_LHS_02 | S10_HYBRID | Z | 0.0355892387911767 | 0.0368183455673379 | 0.0507374634532816 | 0.0644375566915637 |
| R2_LHS_02 | B2 | Z | 0.0371812033501441 | 0.039953413459361 | 0.0551334748354795 | 0.0641828557423997 |
| R6_LHS_04 | S10_HYBRID | X | 0.0199714028692268 | 0.0612743336459539 | 0.1278244948277508 | 0.3819923878453377 |
| R6_LHS_04 | B2 | X | 0.0181402658269503 | 0.0556123915459219 | 0.1224531596991992 | 0.3819923878453377 |
| R6_LHS_04 | S10_HYBRID | Y | 0.0355265636993568 | 0.0338811027923786 | 0.0462680732143365 | 0.0587080247973065 |
| R6_LHS_04 | B2 | Y | 0.024930806387929 | 0.0236225563443107 | 0.0372969676861317 | 0.0457175689909722 |
| R6_LHS_04 | S10_HYBRID | Z | 0.0408226824849148 | 0.0402747810397721 | 0.0628905805876814 | 0.0887203974764634 |
| R6_LHS_04 | B2 | Z | 0.0371812033501441 | 0.039953413459361 | 0.0551334748354795 | 0.0641828557423997 |

## Comparación no compensatoria final

| route | family | S6_capacity | S8_factorial_primary | S9_HPO | S10_nested_OOF | S11_five_seed | predictive_noninferiority | predictive_material_gain | physical_material_gain | graph_utility | modal_evidence | full_state | final_selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | R1_BRIDGE_PINO | EXECUTED_WITH_REPAIRS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R2 | R2_MO_PIGNO | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R3 | R3_GRAPH_NEURAL_GALERKIN | EXECUTED_WITH_REPAIRS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R4 | R4_PORT_HAMILTONIAN_OPINF | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | FAIL | PASS | PASS | PASS_FUNCTIONAL | REPORTED_COMMON_AND_PROJECTED | FAIL_LIMITED | FAIL |
| R5 | R5_ROTATION_MULTISCALE_GNO | EXECUTED_WITH_REPAIRS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
| R6 | R6_LOAD_DEPENDENT_RITZ_KRYLOV | EXECUTED_WITH_REPAIRS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | COMMON_REFERENCE_ONLY | NOT_REACHED | FAIL |
