# Interpretación de resultados

- Estado científico: `NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE`.
- Referencia: modelo FEM implementado y resuelto en COMSOL.
- Evidencia: nested grouped OOF on 68 historically exposed trajectories; not blind or external.
- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.

La selección es no compensatoria: una mejora física no compensa degradación predictiva fuera de tolerancia, y una media favorable no compensa colas, semillas inestables o pérdida de utilidad del grafo. F01–F45 constituyen el atlas reproducible.

## Evidencia emparejada

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

## Claims admitidos

| claim | observation | evidence | metric | FEM | family | case | worst_case | physical_mechanism | architectural_mechanism | figure | table | decision | limitation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| One numerical authority | All admitted comparisons preserve the original FEM/COMSOL identity contract. | S1 data authority and exact same-case/time/node/component contracts | identity counts, hashes, units and axis mapping | original FEM model implemented and solved in COMSOL | all six | 68 historically exposed trajectories | not applicable to an identity gate | common loads, supports, base state and observation map | common immutable data adapter | F02-F06 | DATA_ACCESS_REGISTRY.csv; S14_FAMILY_GATE_MATRIX.csv | PASS | historically exposed trajectories; not blind or external |
| Common structural graph/modal authority | The active Beam topology and first structural modes have an independent Timoshenko audit. | S2 Beam graph and first-12 FEM/COMSOL versus independent Timoshenko audit | frequency error, MAC, cluster-subspace similarity, topology and frame hashes | FEM/COMSOL modal reference | all graph routes | common structural reference | reported by the modal audit, not inferred from forced-response POD | Timoshenko beam stiffness, mass, supports and local frames | shared active Beam graph and fixed modal projection | F03-F05; F39-F41 | MODAL_REFERENCE_AUDIT.md; S12 modal source CSVs | PASS | independent matrices are auditor/regularizer, not COMSOL full transient M,C,K |
| Physics-informed family adds material primary-field value | Acceptance is determined by paired trajectory evidence and noncompensatory predictive/physical gates. | S10 single-seed diagnostic after no route qualified for S11 plus B2/control noninferiority and S12 diagnostics | relative L2, RMSE, MAE, NRMSE, P90, worst case, bootstrap gain and residual reduction | same-case/time/node/component FEM/COMSOL displacement fields | none accepted | all 68 OOF trajectories | explicitly reported in OOF tables and F20-F25 | route-compatible reduced, weak, energy, modal or Ritz physics | frozen route plus capacity-matched data-only control | F17-F45 | S10/S11 OOF metrics; S14_FAMILY_GATE_MATRIX.csv | NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE | full state, sensors and genuinely new FEM panel remain unvalidated |
| Graph is functionally beneficial | Frozen-checkpoint perturbations test whether topology, mechanics and frames contribute to admitted predictions. | S12 OOF graph corruption and consistent node-relabel tests | paired error change, prediction shift, bootstrap interval and permutation invariance | OOF FEM/COMSOL fields | best diagnostic route only | all applicable OOF trajectories | reported per perturbation and global axis | Beam connectivity, edge mechanics and local frames | message passing under frozen checkpoint | F42 | S12 graph-utility source CSV | False | no separately retrained graph-free causal comparator |
| Modal reproduction by PIGNO | Candidate forced responses are projected on fixed structural modes; no candidate eigenpair is manufactured. | OOF response projected on fixed FEM/COMSOL structural modes | response-coordinate peak frequency, temporal MAC, COMAC and subspace angles | first 12 FEM/COMSOL structural modes | best diagnostic route only | energetically admitted modes by trajectory | P90 response-frequency error and minimum admitted subspace MAC | fixed structural modal subspace | post hoc projection of OOF forced response | F39-F41 | S12 modal source CSVs | RESPONSE_MODAL_CONSISTENCY_ONLY | PIGNO does not output eigenpairs; response POD is not labeled structural mode |
