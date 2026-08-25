# Registro de decisiones

## 2026-08-11 — Pruebas unitarias del portafolio y auditoría maestra independiente

- Se añadieron nueve pruebas CPU reproducibles. Cubren: causalidad y BC hard de R1; separación de cabezas de seis GDL de R2; compatibilidad de residuos fuerte/débil de R3; antisimetría, positividad y balance pasivo de R4; transformación polar/axial y jerarquía de R5; ortonormalidad respecto de masa y arranque residual nulo de R6; y exclusión de semillas fabricadas en la rama negativa S10.
- `pytest` devolvió `9 passed`; el resultado se congeló como `audits/PORTFOLIO_OPERATOR_UNIT_TESTS.junit.xml` y será incluido en el paquete.
- `scripts/86_audit_final_master_completion.py` verificará independientemente el estado final, exactamente seis rutas, autoridad única, 68×1201×512×3, seeds realmente disponibles, HDF5 OOF completos, F01–F45 con hashes, 19 informes, binarios, código, pruebas, límites de claims y ausencia de ArrowPoint.
- El watcher S14 fue reemplazado únicamente mientras estaba en `WAITING_FOR_ADMITTED_S12_PATH` para cargar la nueva etapa de auditoría. S10 y los watchers científicos no fueron detenidos ni modificados.

## 2026-08-11 — Rama negativa S10→S12 y cierre S14 automatizados sin fabricar evidencia

- Si S10 no promueve ninguna ruta, no se duplicará la semilla `20260813` ni se denominará finalista a la mejor ruta. `scripts/s12_evidence_context.py` habilita exclusivamente un diagnóstico OOF de una semilla sobre la ruta S10 mejor clasificada, con `five_seed_claim_allowed=false`.
- `scripts/83_run_s12_negative_result_fallback.py` permanece inactivo si existe una ruta S11 normal. Solo ante `NO_S11_POSTCAMPAIGN_NO_PROMOTED_ROUTE` ejecutará los diagnósticos de campo, dinámica, modalidad y utilidad funcional del grafo con el alcance negativo explícito.
- `scripts/81_decide_s14_final_portfolio.py` admite dos rutas de evidencia mutuamente excluyentes. En la rama negativa, la estabilidad multisemilla y la no inferioridad S11 quedan en `NOT_REACHED/FAIL`, el ganador es nulo y el estado científico no puede convertirse en aceptación positiva.
- `scripts/84_assemble_final_portfolio_package.py` crea el paquete contractual, 19 informes, F01–F45 y registros binarios con hardlinks cuando sean posibles o punteros hash explícitos cuando no lo sean. No duplica silenciosamente grandes HDF5/checkpoints.
- `scripts/85_run_final_decision_and_packaging_pipeline.py` espera una ruta S12 admitida y no puede entrenar ni retunar. El heartbeat `monitoreo-portafolio-pigno-s10-s14` supervisa cada 15 minutos sin duplicar los watchers 61, 64, 68, 80, 83 y 85.

## 2026-08-11 — Diagnósticos S12 serializados y alcance de utilidad del grafo

- Los diagnósticos S12 de campos emparejados, modal y utilidad del grafo se ejecutan secuencialmente mediante `scripts/80_run_s12_sequential_extension_pipeline.py` después del núcleo S12. Los watchers independientes 74, 76 y 79 quedaron sustituidos y no deben relanzarse concurrentemente porque el equipo de 32 GiB no dispone de RAM libre suficiente para auditorías HDF5 de campo completo simultáneas.
- El valor histórico `graph_load_branch_sensitivity_relative_l2` se excluye de la decisión sobre utilidad del grafo. Esa intervención eliminó conjuntamente las entradas temporales y de carga, por lo que no aisló topología, atributos mecánicos ni marcos locales.
- F42 utiliza perturbaciones de inferencia con checkpoint congelado y correspondencia mismo caso–tiempo–nodo–componente: reetiquetado consistente de nodos como prueba de invariancia y corrupciones separadas de conectividad, atributos mecánicos y marcos locales como pruebas de dependencia funcional. No se presenta como superioridad causal frente a un modelo sin grafo reentrenado por separado.
- Los scripts S14 81–82 están bloqueados por las puertas completas S11/S12. Una ejecución de prepuerta del script 81 se negó por ausencia del estado del núcleo S12 y no creó una decisión final.

## 2026-08-11 — Recuperación operacional R6-control admitida

- La reanudación conservadora completó `S10_OUTER_R6_LHS_04_OUTER_0_OUTER_OOF_CONTROL_SEED_20260813` en la época 95 con métricas finitas, violación de BC hard exactamente cero e informe admitido; todas las corridas previas completas fueron omitidas.
- En el fold externo 0, R6-physics obtuvo L2 agrupado de desplazamiento X/Y/Z `0.08657/0.07222/0.08349` y residuo reducido mediano `0.008697`; su control de capacidad equivalente obtuvo `0.07840/0.05109/0.04922` y residuo `2.95905`. Se registra como conflicto predictivo–físico de un fold, no como conclusión OOF del portafolio.
- La campaña avanzó después a la selección interna de R4 en el fold externo 1. S11 continúa sin autorización hasta terminar los folds externos, la auditoría independiente y la regla congelada de promoción S10.

## 2026-08-10 — Cierre S6 y promoción condicional a S8

- Evidencia: micropanel común de seis trayectorias históricamente expuestas, una semilla y 150 épocas; no es OOF, generalización ni test ciego.
- Representación admitida: estado `Physical32` separado de cabezas de observación especializadas, con rango 64 para desplazamiento y 128 para velocidad.
- Resultado primario: cinco variantes physics-informed pasaron el umbral de desplazamiento; R3 quedó fuera por X. Ninguna pasó la puerta secundaria completa de velocidad.
- No inferioridad: ninguna variante physics-informed cumplió simultáneamente el margen estricto de 2% para pooled, P90 y peor caso de desplazamiento frente al mejor control de su familia.
- Física: todas las variantes redujeron materialmente el residual mediano frente al control data-only; esta ganancia no compensa por sí sola la no inferioridad fallida.
- Optimización: la única reparación coseno fue rechazada en las seis familias.
- Decisión: promover condicionalmente las seis familias al panel factorial S8 porque el contrato prohíbe cerrar una ruta por una sola salida/caso y P4 admite exactamente seis candidatos. Esta promoción no autoriza HPO ni OOF.
- Artefactos: `s7_directed_repairs/S6_S7_MICROPANEL_PORTFOLIO_DECISION.{json,md}`, `S6_S7_MICROPANEL_RUN_REGISTRY.csv` y `S6_S7_PAIRED_NONINFERIORITY.csv`.

## 2026-08-11 — S10 incremental/total comparison contract frozen during inner selection

- The active S10 trainers predict the train-induced displacement and velocity increments relative to the environment-matched zero-train case.
- Frozen B2 historical OOF fields and metrics describe total response; raw S10 incremental L2 values therefore cannot be compared directly with B2 total-response L2 values.
- Before the first S10 outer OOF result, `s10_nested_grouped_oof/S10_INCREMENTAL_TOTAL_RECONSTRUCTION_CONTRACT_V1.json` froze two auditable views: incremental FEM/PIGNO/control/B2 comparisons and a total cross-fitted hybrid `B2 OOF base + S10 OOF increment` comparison against B2 total OOF.
- Adding the observed FEM base response is prohibited because it would leak the numerical target.
- The B2 and S10 stores contain the same 68 case identifiers and time grid; sampled same-case FEM total values agree exactly. Full chunkwise identity and all reconstruction metrics remain pending the independent S10 audit.
- This clarification does not alter any active trainer, split, target, epoch, or source hash, and it does not authorize S11.

## 2026-08-11 — V1 reconstruction superseded by target-fold-clean B2 refit contract

- Further audit established that the historical B2 and S10 outer folds differ. Therefore, an OOF prediction of a zero-train base case does not by itself prove that the B2 model excluded the loaded S10 target trajectory.
- `S10_INCREMENTAL_TOTAL_RECONSTRUCTION_CONTRACT_V2.json` supersedes V1 before any S10 outer OOF result. Historical B2 evidence remains immutable but is no longer the primary common-split comparator.
- The frozen B2 algorithm will be refitted on the exact S10 outer and inner folds after the active single-GPU campaign. Each S10 target will receive both a B2 total prediction and its environment-matched base prediction from the same B2 outer model whose fit excluded that target.
- The strict total reconstruction is `B2 target-fold-clean base + S10 incremental prediction`; observed FEM base fields remain prohibited.
- `S10_B2_COMMON_SPLIT_FOLDS.csv` and `S10_B2_COMMON_SPLIT_ASSIGNMENTS.csv` were independently checked against all five outer and four inner S10 partitions; every comparison was exact.
- This correction changes no active S10 training artifact and does not authorize S11.

## 2026-08-11 — S10-to-S11 promotion gate frozen before OOF evidence

- `scripts/55_decide_s10_promotion.py` was created and syntax-checked while S10 remained active; its execution guard correctly rejected promotion because the independent S10 audit did not yet exist.
- Only physics-informed variants may be promoted. Their capacity-matched data-only variants remain mandatory noninferiority and attribution controls.
- Each promoted route must be noninferior within 2% to both the common-split B2 comparator and its data-only control on mean total-response relative L2 for X, Y and Z.
- Material gain requires either at least 5% mean relative-L2 reduction in two axes, at least 10% P90/worst-case reduction in one axis, or at least 25% equilibrium-residual reduction relative to the matched control.
- Paired complete-case bootstrap must assign at least 0.95 probability of improvement over B2 in at least one axis. At most two routes may enter S11.
- The frozen rule cannot be evaluated from training loss or inner-fold metrics and does not authorize S11 before the independent OOF audit.

## 2026-08-11 — Direct OOF velocity preservation added before the independent audit

- Every S10 outer prediction already contains `velocity_mps`, and the frozen FEM/COMSOL authority contains `response/delta_velocity_mps` on the same case, saved-time, node and axis grid.
- The future independent audit was extended, before its first execution, to preserve `delta_velocity_mps` and compute direct incremental velocity metrics for physics and capacity-matched control variants.
- No B2 velocity is created by differentiating its displacement prediction. Consequently, B2 remains the common comparator for displacement, while S10 physics-versus-control provides the admissible velocity comparison at this stage.
- This change affects only post-campaign aggregation. It changes no active trainer, fold, checkpoint, epoch, target or FEM/COMSOL artifact and does not authorize S11.

## 2026-08-11 — Partial paired outer-fold monitoring separated from promotion

- `scripts/56_monitor_s10_completed_outer_pairs.py` verifies complete physics/control outer pairs, identical validation trajectories, finiteness, hard BC, causality and absence of outer-target selection leakage.
- Its CSV and JSON outputs are explicitly diagnostic and non-promotional. No partial fold statistic can authorize S11.
- The first completed pair, R4 outer fold 0, showed equilibrium-residual reduction of 0.9999999703 relative to the data-only control but failed the 2% displacement noninferiority check in that fold.
- This single-fold tension is retained as evidence; it neither closes R4 nor establishes benefit. Final attribution remains contingent on exact-once 68-case OOF aggregation and common-split B2 comparison.

## 2026-08-11 — Robust paired-fold physical-gain criterion frozen

- The preliminary promotion script originally compared the ratio of mean outer-fold equilibrium residuals. That statistic could be dominated by a single fold with a larger residual scale.
- Before completion of any second physics/control outer pair, the physical-gain gate was replaced by a paired-fold rule: median equilibrium-residual reduction at least 25% and positive reduction in at least four of five outer folds.
- The predictive noninferiority, common-split B2 comparison, complete-case bootstrap and maximum-two promotion rules are unchanged.
- The guard still rejects execution before the independent OOF audit, and S11 remains unauthorized.

## 2026-08-11 — Conservative storage-headroom gate added to the live paired monitor

- G: free space fluctuated independently of S10 output growth, so raw free-space observations were insufficient for an operational decision.
- The monitor now compares free space with: remaining outer predictions at the maximum observed file size, uncompressed sizes for all six independent-audit field stores, uncompressed common-split B2 fields, and a fixed 25 GiB safety margin.
- At implementation, free space was 161.90 GiB and the conservative requirement was 39.47 GiB, yielding `PASS_STORAGE_HEADROOM`.
- The storage gate is diagnostic and may stop future progression only if headroom actually fails; it changes no scientific threshold or active training artifact.

## 2026-08-11 — Post-OOF dynamic and spatial auditor frozen without touching S10

- `scripts/57_audit_s10_oof_dynamic_spatial_metrics.py` was created while S10 remained active, but its hard guard refuses execution until the independent OOF audit has passed; the refusal was tested and produced no output artifact.
- The auditor preserves the same case, saved time, observation node and global component. It evaluates total displacement against FEM/COMSOL and common-split target-clean B2, and direct incremental velocity against the direct FEM/COMSOL extraction. It never fabricates B2 velocity or acceleration by differentiation.
- Spectral evaluation retains the complete resolvable 0–20 Hz band and reports 0–0.5, 0.5–1, 1–2, 2–5, 5–10 and 10–20 Hz separately; no 5 Hz low-pass filter is used to hide error.
- Kinematic differentiation is only a saved-grid consistency diagnostic measured against the corresponding FEM/COMSOL saved-grid floor. It is not treated as the COMSOL integration scheme or as a replacement for directly extracted velocity.
- A deterministic synthetic identity test passed for PSD distance, dominant frequency, coherence, phase, hotspot location and kinematic consistency. A zero-energy test first exposed an undefined-coherence/phase edge case; the auditor was corrected and the complete synthetic test then passed.
- A full FEM/COMSOL trajectory benchmark exposed zero-energy observation nodes inside SciPy's coherence calculation. Coherence and phase now use only nodes whose target and prediction energies exceed `1e-12` of their respective maximum node energies (with an absolute `1e-30` floor), while global, PSD and spatial errors still retain all 512 nodes. Identity coherence returned exactly 1.0 over 496 energetic nodes, and the projected complete audit cost is approximately ten minutes.
- These additions are postprocessing only. They do not modify the trainer, folds, targets, graph, checkpoints, promotion thresholds or S11 authorization.

## 2026-08-11 — Post-S10 scientific-decision pipeline armed without S11 execution authority

- `scripts/58_run_s10_scientific_decision_pipeline.py` waits for the existing postcampaign pipeline to admit B2 common-split fields and the independent OOF audit.
- It then runs the dynamic/spatial audit followed by the already frozen S10-to-S11 promotion decision. Any upstream or postprocessing failure stops the watcher and preserves S11 as not started.
- The watcher cannot launch training. Even when the promotion decision sets `S11_authorized=true`, its terminal artifact explicitly records `S11_training_started=false`.
- The live watcher was verified through PIDs 38456/39548 and `S10_SCIENTIFIC_DECISION_PIPELINE_STATUS.json`; the active S10 trainer remained unchanged.

## 2026-08-11 — S11 five-seed protocol frozen before completion of S10

- `s11_five_seed_confirmation/S11_FIVE_SEED_CONFIRMATION_PROTOCOL_V1.json` fixes seeds `[0,1,2,3,4]`, the same five grouped outer folds, and both physics and capacity-matched control variants for at most two promoted finalists.
- Each fold reuses only its epoch selected from S10 inner folds; seed-specific or outer-target checkpoint selection is forbidden. Every fit starts from a new initialization.
- The maximum confirmation budget is 100 runs: 2 finalists × 5 folds × 5 seeds × 2 variants. If S10 promotes fewer candidates, unused runs are not reassigned.
- Stability requires at least four of five seeds. The representative checkpoint is the median admitted seed, never the most favorable seed.
- The protocol hash is `61CBD620448C875061372A0C134F36B93B68A7B6BEAD14C8A7BCBFF317B0F85A`. Execution remains blocked until the S10 scientific-decision pipeline passes and promotes at least one candidate.

## 2026-08-11 — S11 execution path prepared and gated before finalist identities are known

- `scripts/59_run_s11_fold_seed_confirmation.py` reuses the immutable S10 fold worker under a separate S11 run root, checks promotion, seed, fold and inner-selected epoch, and writes a canonical S11 alias without altering the worker report.
- `scripts/60_run_s11_five_seed_campaign.py` executes a resumable, sequential single-GPU plan for every promoted finalist, five folds, five seeds and both physics/control variants. A failed run stops the campaign before the next run.
- `scripts/61_run_s11_after_s10_gate.py` waits for the S10 scientific-decision pipeline. It exits without training if no route is promoted; otherwise it starts only the frozen S11 campaign.
- The watcher was syntax-checked and verified alive under PIDs 34972/21408 with status `WAITING_FOR_S10_PROMOTION_GATE`. No S11 training has begun.

## 2026-08-11 — Independent S11 OOF auditor prepared with pre-execution refusal tests

- `scripts/62_audit_s11_five_seed_oof.py` requires a completed S11 campaign, independently checks every run, split, epoch, causality and hard BC, and enforces exact-once 68-case coverage for every finalist, variant and seed.
- It preserves full OOF incremental displacement, direct incremental velocity and target-clean hybrid total displacement fields, plus per-case and per-seed pooled/P90/P95/worst metrics.
- It does not authorize S12; a separate paired five-seed decision remains mandatory.
- The worker, campaign and audit entry points were each executed before their upstream gate. All three refused explicitly and created no training or audit evidence.

## 2026-08-11 — Second S10 outer physics/control pair retained as non-promotional evidence

- R2 outer fold 0 completed with physics/control displacement-X pooled relative L2 of `0.1154847/0.0655312` and equilibrium residual medians of `0.1961190/2.1872098`.
- The physics variant reduced the residual by `91.03%` but failed the fold-level 2% displacement noninferiority check. R4 fold 0 shows the same qualitative conflict with a stronger residual reduction.
- This is evidence of a data-physics tradeoff in one outer partition, not a route-level verdict. R2 and R4 remain active until all five folds, common-split B2, exact-once OOF aggregation and paired bootstrap are available.
- The frozen promotion gate is unchanged: physical gain cannot compensate predictive inferiority.

## 2026-08-11 — S11-to-S12 decision and postcampaign automation frozen before S10 completion

- `S11_TO_S12_DECISION_PROTOCOL_V1.json` requires four of five admitted seeds and noncompensatory 2% noninferiority in pooled, P90 and worst total-displacement errors for X/Y/Z against both common-split B2 and the matched control.
- `scripts/63_decide_s11_to_s12_full_diagnostics.py` computes five-seed medians, paired trajectory bootstrap and 25 paired seed-fold residual reductions. Preliminary final eligibility requires stability, noninferiority and material predictive or physical gain.
- Every independently admitted S11 finalist still enters S12 diagnostics even if preliminary eligibility fails, because S12 performs no fitting. This preserves negative evidence and prevents closing a route from a single output while also preventing diagnostic metrics from compensating a failed predictive gate.
- `scripts/64_run_s11_postcampaign_pipeline.py` waits for S11, executes the independent OOF audit and decision once, and never authorizes training or tuning in S12. The watcher is alive and reports `WAITING_FOR_S11_AUTORUN`.

## 2026-08-11 — Forty-five-figure S12 visualization contract frozen before final OOF evidence

- `s12_final_diagnostics/S12_VISUALIZATION_CONTRACT_V1.json` defines 45 unique thesis figures spanning authority, graph, loads, capacity, repairs, HPO, convergence, complete OOF distributions, tails, time histories, fields, spectra, physics, modality, graph utility, cost and final decision.
- Every figure requires PNG 300 dpi, vector PDF, source CSV/Parquet, caption JSON and a figure manifest containing run, case, fold, seed, units and hashes.
- The coordinate convention is fixed as X transverse, Y vertical and Z longitudinal. Three-dimensional plots use `[X,Z,Y]`, orthographic projection and true coordinate-span aspect; any readability scaling must be separately and explicitly labeled.
- ArrowPoint figures, hidden 5 Hz filtering, silent longitudinal compression, snapshot bootstrap and modal claims from response POD modes are prohibited.
- The contract contains 45/45 unique figure identifiers and has SHA-256 `CE98D05A648F3D0C730BCBDF1FB18DF4F7D91F67DA33262540014337CA065942`. Execution remains blocked until admitted S11 OOF evidence exists.

## 2026-08-11 — First executable S12 figure batch prepared and refusal-tested

- `scripts/65_generate_s12_core_oof_figures.py` implements F17, F18, F20, F21, F23, F37, F38 and F43: ECDF, error distributions, tail metrics, case-model-axis heatmap, five-seed stability, paired residual reduction, hard-BC compliance and error-capacity-VRAM comparison.
- Candidate case-level plots use the median across five seeds before treating 68 trajectories as paired observational units; seeds and snapshots are not falsely counted as independent cases.
- Every implemented figure writes PNG 300 dpi, vector PDF, source CSV, caption JSON and a hash-bearing manifest. Exact-zero BC values remain zero in source data and are displayed at `1e-16 m` only for a logarithmic plot.
- The script passed AST/import checks and its pre-S12 execution test refused without creating figures. It performs no training or tuning.

## 2026-08-11 — Full saved-grid S12 dynamic/spatial multiseed auditor prepared

- `scripts/66_audit_s12_dynamic_spatial_multiseed.py` evaluates all admitted finalist and matched-control seeds over all 68 OOF trajectories, 512 observation nodes, 1201 saved times and X/Y/Z components.
- It computes PSD distance, dominant frequency, coherence, phase, six energy bands through the 20 Hz Nyquist limit, nodewise spatial errors, hotspot distance and saved-grid displacement-to-direct-velocity consistency.
- B2 is evaluated only for total displacement. No B2 velocity or acceleration is manufactured; candidate velocity is compared directly with FEM/COMSOL velocity, and acceleration remains outside this auditor.
- The script passed syntax and refusal tests. It performs no filtering, fitting, checkpoint selection or tuning and remains blocked by the S11-to-S12 decision.

## 2026-08-11 — Executable S12 hotspot, spatial, spectral and kinematic figures prepared

- `scripts/67_generate_s12_dynamic_spatial_figures.py` implements F30–F36 from admitted multiseed OOF fields: hotspot distance, true-aspect node error, PSD, coherence, phase, band energy and saved-grid kinematic consistency.
- Representative cases and seeds are selected by a frozen median-performance rule, not by visual attractiveness. Their identities are serialized in the output report.
- F31 uses the actual 512 FEM/COMSOL observation coordinates and plots `[X,Z,Y]` with orthographic true-span aspect. F32–F34 retain 0–20 Hz; phase is left undefined where target spectral energy is below the declared floor.
- Every figure inherits the PNG/PDF/source-data/caption/hash-manifest contract. The script passed syntax/import and pre-gate refusal tests and produced no premature figure.

## 2026-08-11 — Partial S12 diagnostics pipeline armed behind the full S11 gate

- `scripts/68_run_s12_diagnostics_pipeline.py` waits for the admitted S11 postcampaign decision and then executes core OOF figures, multiseed dynamic/spatial audit and F30–F36 generation in a fixed order.
- It validates existing report statuses before skipping any step, stops on the first failure and cannot train, tune or authorize a final decision.
- The current implemented batch covers 15 contract figures: F17, F18, F20, F21, F23, F30–F38 and F43. The terminal status explicitly remains `PARTIAL_FIGURE_SET` until the other required figures and modal/graph/final-decision evidence exist.
- The watcher is alive under its Python process chain and reports `WAITING_FOR_S11_POSTCAMPAIGN`; no S12 computation has begun.

## 2026-08-11 — F01–F06 authority, graph, load and FEM-reference figures accepted after visual QA

- `scripts/69_generate_s12_authority_graph_figures.py` generated F01–F06 from the frozen campaign ledger, active 22,164-node/48,430-edge Beam graph, causal axle histories and original FEM/COMSOL fields. No training, tuning, FEM recomputation or ArrowPoint output occurred.
- Original-resolution inspection rejected compressed three-dimensional versions of F03/F04/F06. Rejected attempts remain preserved under `s12_final_diagnostics/rejected_visual_qa_v1` through `rejected_visual_qa_v4_stale_generator_hash`; none is treated as an accepted thesis product.
- F04 now uses plan, elevation and cross-section projections with the true physical aspect of each displayed coordinate pair. F06 uses true-aspect plan and elevation projections with one symmetric color scale across base, incremental and total vertical displacement fields.
- The common figure writer now records the actual calling generator and its SHA-256 rather than incorrectly attributing imported generators to script 65.
- `scripts/70_audit_s12_authority_graph_figures.py` returned `PASS_S12_AUTHORITY_GRAPH_VISUAL_QA_V1`: all six PNG/PDF/CSV/caption/manifest sets are nonempty, hashes and generator provenance match, and ArrowPoint count is zero. This admits only F01–F06; it does not authorize S12 final decision.

## 2026-08-11 — S10 continues without intervention after the R6 inner-fold transition

- R6 outer-fold-0 inner folds 0 and 1 completed; inner fold 1 selected epoch 90 with displacement pooled L2 X/Y/Z `0.06449/0.04230/0.03145`, hard-BC violation `0`, finite residual `0.01255` and no outer-target checkpoint selection.
- The campaign advanced normally to `S10_INNER_R6_LHS_04_OUTER_0_INNER_2_PHYSICS_SEED_20260813`, observed at epoch 37/100 with a finite best selection key. No process was restarted or modified.
- The two-fold evidence remains internal model selection only. It neither promotes R6 nor changes the frozen S10 gate.

## 2026-08-11 — F07–F16 historical experiment figures accepted with declared limitations

- `scripts/71_generate_s12_historical_experiment_figures.py` generated F07–F16 from frozen S5 oracle floors, S6 capacity records, S7 directed repairs, S8 factorial results and S9 multifidelity reports. It performed no operator fitting, hyperparameter adjustment or OOF selection.
- F07–F16 cover oracle rank floors, six-family capacity, directed repairs, factorial physics/control errors, successive halving, parallel coordinates, exploratory permutation importance, error–physics–cost Pareto structure, validation convergence and recorded data/physics losses.
- Visual QA rejected the first color-rich version because it violated the frozen palette and F10 had low annotation contrast. The rejected package is preserved under `s12_final_diagnostics/rejected_visual_qa_historical_v1`.
- The accepted version uses marker and line style for route identity, blue/orange for physics/control where applicable, and contrast-aware heatmap labels. F13 is explicitly an exploratory four-fold permutation association with only 32 trials, not a causal ranking. F16 explicitly records that per-term gradient norms and gradient cosines were unavailable in S9 logs.
- `scripts/72_audit_s12_historical_experiment_figures.py` returned `PASS_S12_HISTORICAL_EXPERIMENT_VISUAL_QA_V1`; every PNG/PDF/CSV/caption/manifest set and generator hash passed, with zero ArrowPoint files. This does not authorize OOF claims or the final decision.

## 2026-08-11 — S10 R6 outer-fold-0 inner selection continues normally

- The campaign advanced from R6 inner fold 2 to inner fold 3 without intervention and was observed at epoch 60/100 with a finite best selection key, hard primary tier `0` and best worst-axis pooled displacement `0.06853`.
- S11 remains explicitly unauthorized. Inner-fold progress is not OOF evidence and is not compared to B2 for promotion.
# 2026-08-11 — Evidence-preserving S10 recovery

- S10 was not treated as slow after its runner and worker vanished with stale epoch-0 state and zero GPU activity.
- The incomplete R6-control directory contained no checkpoint, prediction or report and was quarantined rather than deleted.
- Restart used the existing `run_one` admission rule: every completed run with a valid `PASS_S10_FOLD_TRIAL_EXECUTION` report is skipped; only the incomplete R6-control run is repeated.
- This is an operational recovery, not a scientific rerun, protocol change, hyperparameter change or new seed.

## 2026-08-11 — Corrección predecisional de la puerta S10 pooled/P90/worst

- Evidencia: `scripts/51_audit_s10_nested_oof_independent.py` agregaba la media de errores L2 por trayectoria, P90 y peor caso, pero no el L2 pooled exacto; `scripts/55_decide_s10_promotion.py` aplicaba no inferioridad solo a la media.
- Riesgo evitado: promover a S11 una ruta con colas o error pooled degradados y consumir hasta 100 corridas de cinco semillas bajo una puerta más débil que `ACCEPTANCE_GATES.json` y `S11_TO_S12_DECISION_PROTOCOL_V1.json`.
- Decisión: conservar estadísticas aditivas por campo, calcular `pooled_relative_l2` exactamente y exigir no inferioridad no compensatoria de pooled, P90 y worst, por eje, frente a B2 y al control de capacidad equivalente.
- Alcance: postprocesamiento y promoción futuros. No se modificaron datos FEM/COMSOL, splits, representaciones, checkpoints, resultados terminados ni el entrenamiento S10 activo.

## 2026-08-11 — Reapertura dirigida de R4 por incumplimiento arquitectónico

- La interrupción no responde al error de una salida o un caso, sino a que la familia ejecutada no corresponde a la definición congelada.
- Se conserva toda evidencia y se mantiene cerrado el portafolio en seis rutas: la reparación no crea una séptima familia.
- R2 y R6 no se reinician. R4 debe volver a la última puerta compatible, demostrar núcleo pH/OpInf efectivo, gradiente/embedding físico y control state-space equivalente antes de reingresar al panel factorial.
- No se modifican FEM/COMSOL, grafo, cargas, splits, autoridad de datos ni resultados previos.

## 2026-08-11 - Reparacion R4: puerta de capacidad antes de E30

- La puerta de representacion fold-clean aprobo cuatro folds con error directo peor menor de 5,5 %, pasividad y balance energetico.
- Antes de ejecutar E30 se congelan para capacidad: desplazamiento por eje <=10 %, estado q <=10 %, estado v <=20 %, mediana/peor velocidad de campo <=70 %/85 %, BC exactas, energia relativa <=1e-5, gradiente residual no nulo y mejora frente a epoca 0.
- E30 es la reparacion de optimizacion de R4; un incumplimiento no autoriza otra extension ad hoc.

## 2026-08-11 - R4 reparado admite reingreso limitado al panel factorial

- Capacity E150 paso todos los gates congelados; no se interpreta como generalizacion.
- El micropanel de seis casos paso campo primario: pooled X/Y/Z 9,29/5,99/4,75 %, P90 10,84/9,56/5,65 %, BC=0, causalidad=0 y rama grafica-carga no nula.
- Fallo estado completo: velocidad pooled X/Y/Z 76,49/64,80/52,47 % y P90 78,99/75,33/59,25 %. El residual de equilibrio P90 fue aproximadamente 1,00 y se conserva como limitacion.
- El ajuste pH S6 tuvo rango energetico 63/64; por ello no autoriza HPO. La ruta reingresa solo a S8, donde el panel de 12 casos y la auditoria fold-clean demostraron rango 64/64.
# 2026-08-11 — Transición a registro paralelo de seis rutas

- La instrucción de portafolio congela exactamente R1–R6 y revoca el desarrollo secuencial de una única arquitectura.
- Se detuvo de forma acotada la corrida `S9_MEDIUM_R4_LHS_02_FOLD_2_PHYSICS_REPAIRED_EFFECTIVE_PH_OPINF_SEED_20260812`; se preservaron logs, estados y checkpoints parciales.
- La primera entrega S0 ya existía y fue revalidada: 14 artefactos nominales no vacíos, decisión `GO_PORTFOLIO_DESIGN`, JSON/CSV estructuralmente válidos y seis rutas únicas.
- La auditoría S9 histórica permanece como evidencia, pero no gobierna decisiones futuras porque su R4 usó una implementación posteriormente invalidada.
- R1, R2 y R6 conservan su evidencia S9 válida; R3 y R5 permanecen como comparadores negativos S8; R4 debe completar solo la reparación acotada ya iniciada.
- OOF continúa bloqueada hasta una nueva auditoría común S9 con R4 reparada.
# 2026-08-11 — S9 common closure and repaired-R4 S10 re-entry

- The independent common S9 audit passed all gates and promoted exactly `R4_LHS_03`, `R2_LHS_02`, and `R6_LHS_04` to S10 preparation. This is fold-clean development evidence, not OOF or external validation.
- Historical S10 R4 physics runs are excluded because they used `Physical32_Newmark`, not effective port-Hamiltonian OpInf. Dataset, frozen nested splits, fold-local representations, and admitted R2/R6 outer-fold-0 pairs are preserved.
- The first repaired-R4 S10 smoke exposed rank deficiency in the full 64-state Hamiltonian gradient. The one representation repair selects the largest training-only shared generalized-coordinate subspace with full Hamiltonian-gradient rank and projects M/C/K/forces congruently.
- The repaired smoke passed with generalized rank 19, Hamiltonian-gradient rank 38/38, exact causal witness and hard BC, finite metrics, and a dissipative operator. Its one-epoch predictive error is not a utility result.
- Full S10 nested grouped OOF execution is authorized; S11 remains blocked until the independently audited paired S10 decision.
# 2026-08-11 — Partial S10 paired evidence and downstream R4 identity repair

- The read-only paired monitor recognizes two completed historical outer-fold-0 pairs, R2 and R6, and excludes every legacy R4 physics run lacking `REPAIRED_EFFECTIVE_PH_OPINF`.
- Both existing pairs show strong equilibrium-residual reduction but fail the 2% displacement noninferiority diagnostic in that fold. This is retained as adverse partial evidence and is not used to close either route before exact-once 68-case OOF aggregation.
- R4 repaired inner-fold-0 reached a zero-violation inner selection key during training; this remains inner-development evidence only.
- S10/S11 run resolution, five-seed confirmation, historical convergence figures, and the S12 graph-utility reconstruction now resolve repaired R4 identities explicitly. The S12 R4 reconstruction refits the same training-only identifiable pH-OpInf subspace rather than substituting the historical Newmark anchor.

# 2026-08-11 — S12 visualization coverage frozen without premature generation

- The independent coverage audit maps all F01-F45 contract figures to an existing generator; this establishes design readiness only.
- Only F01-F06 currently have complete active PNG/PDF/source/caption/manifest bundles. F07-F16 remain archived as invalid pre-effective-pH-OpInf visuals, and F17-F45 remain blocked by their repaired S10/S11/S14 evidence gates.
- The F07-F16 generator now reads the repaired S8 registry and repaired common S9 audit, excludes every legacy R4 physics report lacking `REPAIRED_EFFECTIVE_PH_OPINF`, and labels S6/S7 R4 points as historical pre-repair evidence rather than forward evidence.
- No training, tuning, promotion, figure generation, or scientific decision was performed by this audit.
- The future S14 decision and final-package assembler were also redirected from the superseded S8/S9 audits to the repaired-R4 V3/V1 authorities; historical files remain preserved but cannot govern the final gate matrix.
- The S12 execution chain now regenerates F07-F16 after S11, performs a hash/source structural audit, and prepares F01-F43 contact sheets. Structural QA cannot assert visual readability; S14 now requires a separately recorded agent visual review.
- F44-F45 have an equivalent post-decision readiness stage. Final packaging is blocked until their separate manual visual-QA record passes.
- S10 progress accounting now excludes every legacy R4 physics report. At correction time the forward-valid counts were 9/60 inner and 5/30 outer; the previous 17/60 and 6/30 values were contaminated by eight legacy inner reports and one legacy outer physics report.
- Reuse of legacy-hash non-R4 physics and capacity-matched controls is governed by `S10_LEGACY_TRAINER_PATH_EQUIVALENCE_AUDIT_V1.json`: the repair flag defaults false and every new computational branch is unreachable outside repaired R4 physics. This is a bounded static path argument, not a formal program-equivalence proof, and all legacy R4 physics remains excluded.
- The final completion audit now requires the admitted S10 exact-once audit, complete finite OOF fields for both physics and controls, exact 68-case ordering, frozen saved-time shape, five distinct protocol seeds for any positive finalist, and cryptographically bound manual visual-QA records for F01-F45.
- Manual visual QA cannot be satisfied by a status-only JSON: each figure must have an ordered finding, six explicit readability/semantics checks, the reviewed PNG hash, and the readiness-artifact hash.

# 2026-08-11 — S11 repaired-R4 execution gate audited end to end

- The first inspection of the top-level S11 runner incorrectly suggested that the repair flag was missing. End-to-end inspection showed that `59_run_s11_fold_seed_confirmation.py` is the actual wrapper: it reads the frozen fold epoch, fixes `phase=outer`, and already appends `--r4-repaired` for `R4_LHS_03/physics` before invoking the S10 worker.
- A transient redundant top-level flag was removed before S11 began; it affected no S10 run, metric, checkpoint, split, or scientific decision.
- `62_audit_s11_five_seed_oof.py` was validly strengthened: repaired-R4 evidence is rejected unless effective pH-OpInf diagnostics are present, finite, converged, full rank in the identifiable Hamiltonian-gradient space, and dissipative within numerical tolerance.
- Tests now assert the real two-level execution path, frozen epoch/outer phase, repaired flag, and independent diagnostic gate. S11 remains blocked by the S10 promotion gate.
- The resumable S11 orchestrator now validates every existing alias against its frozen trial/fold/seed/variant identity, selected epoch, complete report and prediction artifacts, leakage flag, finiteness, BC and causality before skipping it. Repaired R4 additionally requires converged, full-rank, dissipative pH-OpInf diagnostics. This prevents a stale status-only alias from surviving until the end-of-campaign audit.
- The final completion audit now requires the exact frozen route identifiers `R1_BRIDGE_PINO` through `R6_LOAD_DEPENDENT_RITZ_KRYLOV` and the exact six packaged implementation modules. A count of six alone is no longer accepted as proof that the frozen nonredundant portfolio was preserved.
- The package assembler applies the same exact-module-set rule before committing the staging directory, so a renamed, duplicated or missing family fails structural QA before final publication rather than only in the terminal audit.

# 2026-08-11 - S10 matched-budget repair for R4 outer controls

- Repaired R4 outer-fold-0 physics completed at the four-inner-fold selected epoch 95, while the preserved control report had been trained for epoch 85.
- The independent S10 audit requires both variants to match the frozen inner-selected epoch. Therefore, the epoch-85 control cannot be used in the paired comparison even though its report is otherwise finite, causal and complete.
- The S10 runner now validates resumable report identity, gates, prediction presence and selected epoch. It archives only an incompatible outer report recoverably, then recomputes that exact variant at the frozen epoch; compatible physics and other families are preserved.
- The read-only monitor excludes outer reports whose epoch differs from their corresponding inner-selection artifact. The corrected forward-valid count is 12/60 inner and 5/30 outer before recomputing the R4 outer-fold-0 control.
- A 25-test suite passes, including mismatch detection, recoverable archival, matching-epoch reuse and rejection of identity drift. S11 remains blocked.

# 2026-08-11 - First matched repaired-R4 outer pair (diagnostic only)

- R4 outer-fold-0 physics and control both completed at the frozen epoch 95 on the same 14 trajectories.
- Physics reduced pooled displacement L2 versus control by 20.21%/28.84%/44.76% on X/Y/Z, reduced P90 and worst-case displacement on all axes, and reduced median equilibrium residual by 98.10%.
- Velocity changed by -3.73%/-6.32%/+1.34% improvement on X/Y/Z; the X/Y degradation is retained and must be judged by the complete non-compensatory OOF contract.
- This is one historically exposed grouped outer fold. It is neither promotion, complete OOF evidence nor external validation; R4 remains active because the master protocol forbids closure from one case, output or fold.

# 2026-08-11 - R4 outer-fold-1 four-inner selection

- All four repaired R4 inner partitions completed 100 epochs and passed execution, finiteness, exact hard BC, causality, converged full-rank identifiable pH-OpInf, and numerical dissipativity gates.
- Their individually selected epochs were 85, 90, 90 and 85; these are not averaged or reused as the outer budget.
- An independent recomputation of the frozen common-epoch rule confirmed epoch 100 as the lexicographic minimizer of the componentwise maximum across all four inner curves. Its aggregate key begins with zero violations and worst pooled displacement-X L2 0.068269, versus 0.070281 at the second-ranked epoch 85.
- The outer-fold-1 physics and control runs therefore both use exactly 100 epochs. This choice uses inner-training evidence only; outer targets remain excluded from checkpoint and hyperparameter selection.

# 2026-08-12 - Second matched repaired-R4 outer pair (diagnostic only)

- R4 outer-fold-1 physics and capacity-matched control both completed at the frozen common epoch 100 on the same 14 external-fold trajectories; the physics prediction artifact contains finite displacement and velocity fields with shape `14 x 1201 x 512 x 3` plus finite reduced `q`, `qdot` and `qddot` states.
- Physics worsened pooled displacement L2 versus control by 37.22%/51.92%/30.90% on X/Y/Z, and worsened displacement P90 and worst case on every axis. Velocity also worsened by 8.04%/1.07%/1.15% on X/Y/Z.
- Physics reduced the median equilibrium residual from 1.651921 to 0.024354 (98.53%) while retaining exact hard BC and zero measured causality violation. This is a real physical-gain/predictive-loss conflict, not a promotion result.
- Outer fold 0 and outer fold 1 give opposite conclusions on the 2% displacement non-inferiority gate. R4 therefore remains open until all five grouped outer folds and the exact-once 68-case OOF aggregation are complete; neither fold is allowed to close or promote the family alone.

# 2026-08-12 - Second matched R2 outer pair (diagnostic only)

- Four inner partitions for R2 outer fold 1 individually selected epochs 85, 90, 70 and 100. Independent componentwise-max recomputation of the frozen common-epoch rule selected epoch 70; epochs 80 and 100 ranked second and third. Outer targets were not used.
- At the matched epoch 70, R2 physics reduced pooled displacement L2 versus its capacity-matched control by 6.34%/10.47%/13.69% on X/Y/Z. Displacement P90 and worst case also improved on all three axes.
- The median equilibrium residual decreased from 2.433731 to 0.168189 (93.09%). Exact hard BC and zero measured causality violation were retained.
- Velocity improved by 1.20% on Z but degraded by 3.57% and 2.13% on X and Y. The pair is therefore a positive displacement/physics diagnostic with a non-compensable velocity warning, not a family promotion.

# 2026-08-12 - Second matched R6 outer pair (diagnostic only)

- R6 outer-fold-1 used common epoch 80, independently confirmed as the componentwise-max lexicographic optimum across its four inner curves; epochs 70 and 90 ranked second and third.
- Physics reduced pooled displacement L2 versus the capacity-matched control by 2.99%/19.32%/21.71% on X/Y/Z and reduced median equilibrium residual from 2.901290 to 0.012398 (99.57%).
- Displacement P90 and worst case improved materially on Y/Z but worsened slightly on X. Velocity improved by 1.17%/6.60% on X/Z and worsened by 0.68% on Y.
- The pair supports a physical and mostly predictive gain but does not satisfy every non-compensatory tail/output gate. R6 remains active until complete 68-case exact-once OOF aggregation and comparison with B2.

# 2026-08-12 - Third matched repaired-R4 outer pair (diagnostic only)

- The four repaired R4 inner partitions for outer fold 2 completed the frozen 100-epoch budget. The common-epoch componentwise-maximum lexicographic rule selected epoch 85 with zero violations and aggregate displacement-X L2 0.068443; only inner-training evidence was used.
- Physics and its capacity-matched control both completed at epoch 85 on the same 14 outer-fold trajectories with 377,760 parameters, finite `14 x 1201 x 512 x 3` displacement and velocity fields, exact hard BC, zero measured causality violation, and no outer-target checkpoint or hyperparameter selection.
- Physics reduced pooled displacement L2 versus control by 9.72%/25.35%/32.94% on X/Y/Z and reduced the median equilibrium residual from 1.902432 to 0.027899 (98.53%). The repaired pH-OpInf fit was finite and converged, with gradient rank 64/64, identifiable generalized rank 32/32, and maximum symmetric eigenvalue 6.34e-16.
- The displacement P90 and worst case improved on Y/Z but worsened slightly on X. Velocity worsened by 3.72%/5.27%/1.67% on X/Y/Z. The fold therefore supplies a positive displacement/physics diagnostic with non-compensable tail-X and velocity warnings; it is not promotion or complete OOF evidence.
- Across the first three repaired R4 outer folds, the 2% all-axis displacement non-inferiority indicator is true in folds 0 and 2 and false in fold 1. R4 remains open until five-fold exact-once 68-case OOF aggregation, common-split B2 comparison, and paired bootstrap are complete.

# 2026-08-12 - Third matched R2 outer pair (diagnostic only)

- R2 outer fold 2 selected common epoch 95 from the four inner-training curves using the frozen componentwise-maximum lexicographic rule. The aggregate key had zero violations and displacement-X L2 0.068421; outer targets were not used.
- Physics and its capacity-matched control both completed at epoch 95 with 396,512 parameters on the same 14 trajectories. The physics artifact contains finite displacement and velocity fields shaped `14 x 1201 x 512 x 3`; both variants retain exact hard BC and zero measured causality violation.
- Physics improved pooled displacement L2 by 38.45%/38.63%/54.09% and velocity L2 by 4.57%/10.47%/7.88% on X/Y/Z. Displacement P90 and worst case also improved on all axes.
- Median equilibrium residual decreased from 2.937758 to 0.206128, a 92.98% reduction. This is the cleanest R2 fold observed so far across displacement, velocity, tails, and physics.
- The result is still one historically exposed grouped outer fold and cannot promote R2. Promotion remains blocked until all five folds produce exact-once 68-case OOF fields, common-split B2 is evaluated, and the paired bootstrap and frozen non-compensatory gate are applied.

# 2026-08-12 - Third matched R6 outer pair (diagnostic only)

- R6 outer fold 2 selected common epoch 85 from four inner-training curves using the frozen componentwise-maximum lexicographic rule. The aggregate key had zero violations and displacement-X L2 0.069634; outer targets were not used.
- Physics and its capacity-matched control both completed at epoch 85 with 279,753 parameters on the same 14 trajectories. Both produced finite `14 x 1201 x 512 x 3` displacement and velocity fields, exact hard BC, and zero measured causality violation.
- Physics changed pooled displacement L2 by -2.79%/+5.04%/+17.36% improvement on X/Y/Z; the X degradation exceeds the 2% per-axis non-inferiority tolerance. Velocity changed by +1.64%/-2.04%/+5.76% improvement on X/Y/Z.
- Displacement P90 and worst case improved on Y/Z and worsened on X. Median equilibrium residual decreased from 3.152110 to 0.012102, a 99.62% reduction.
- This fold is a strong physical gain with mixed predictive effects and fails the all-axis displacement non-inferiority indicator. It is diagnostic only; R6 remains open until all five folds and the exact-once 68-case OOF comparison with B2 are complete.

# 2026-08-12 - Preserved operational interruption during repaired-R4 outer-fold-3 control

- Repaired R4 outer-fold-3 completed all four inner partitions and selected common epoch 85 solely from the four inner-training curves. The physics outer run then completed at that frozen epoch with finite outputs, exact hard BC, zero measured causality violation, 377,760 parameters, a converged 64/64 repaired pH-OpInf fit, maximum symmetric eigenvalue 5.61e-16, and no use of outer targets.
- The capacity-matched outer control subsequently stopped without `report.json` after persisting epoch 50/85. PID 33524 and both campaign-parent processes were absent, GPU usage returned to zero, no `run_finished` event existed, and no Windows shutdown or application-crash event explained the interruption. This is an operational interruption, not a scientific failure or completed comparison.
- Recovery is restricted to the existing audited resume mechanism in `scripts/49_run_s10_nested_oof_campaign.py`: preserve the report-less directory under `interrupted_partial_runs`, validate and skip every completed identity, and recompute only the exact missing R4 outer-fold-3 control at the already frozen common epoch 85. No S11, new family, new FEM run, or outer-target-based adjustment is authorized by this recovery.

# 2026-08-12 - Fourth matched repaired-R4 outer pair (diagnostic only)

- The interrupted outer control was recovered exactly as authorized: its epoch-50 partial directory was preserved under `interrupted_partial_runs`, every completed inner/physics identity was validated and skipped, and only the same capacity-matched control was recomputed at the already frozen common epoch 85. The recovered control completed with finite outputs, exact hard BC, zero measured causality violation, 377,760 parameters, and no use of outer targets.
- Against that control, repaired R4 physics changed pooled displacement L2 by -4.65%/-18.75%/+2.19% improvement on X/Y/Z and velocity L2 by -12.30%/-2.83%/+0.75%. Negative values denote degradation.
- Displacement P90 changed by -14.51%/-18.01%/+10.82% and worst-case error by -11.04%/-13.22%/+5.68% on X/Y/Z. Thus both ordinary and tail performance degrade materially on X/Y despite the Z improvement.
- Median equilibrium residual decreased from 1.514459 to 0.025197, a 98.34% reduction. This physical gain cannot compensate the non-inferiority failures; R4 outer-fold-3 is a negative predictive diagnostic and the family remains unpromoted pending complete five-fold OOF evidence.
- Ten of fifteen planned physics/control outer pairs are now complete. S11 remains blocked until exact-once 68-case aggregation, the common-split B2 comparison, paired bootstrap, and the frozen non-compensatory promotion gate are all complete.

# 2026-08-12 - Fourth matched R2 outer pair (diagnostic only)

- R2 outer-fold-3 selected common epoch 95 from the four inner-training curves using the frozen componentwise-maximum lexicographic rule. The aggregate key had zero violations; outer targets were not used.
- Physics and its capacity-matched control both completed at epoch 95 with 396,512 parameters, finite outputs, exact hard BC, zero measured causality violation, and no outer-target checkpoint or hyperparameter selection.
- Physics changed pooled displacement L2 by -8.54%/-2.94%/+3.17% improvement on X/Y/Z and velocity L2 by +0.70%/-0.83%/-0.92%. Negative values denote degradation.
- Displacement P90 improved by 2.28% on X and 3.69% on Z but worsened by 7.48% on Y. Worst-case error improved by 2.14%/4.69% on X/Z and worsened by 17.76% on Y.
- Median equilibrium residual decreased from 3.506577 to 0.222611, a 93.65% reduction. The physical gain cannot compensate the displacement non-inferiority failures on X/Y or the Y-tail degradation; this fold is a mixed-to-negative predictive diagnostic, not promotion evidence.
- Eleven of fifteen planned physics/control outer pairs are complete. R2 remains open until five-fold exact-once OOF aggregation, common-split B2 comparison, paired bootstrap, and the frozen non-compensatory promotion gate are complete.

# 2026-08-12 - Bounded transient-I/O repair after R6 inner-fold interruption

- `S10_INNER_R6_LHS_04_OUTER_3_INNER_3_PHYSICS_SEED_20260813` stopped at persisted epoch 29 with return code 1 because the synchronized G: drive raised `PermissionError: [Errno 13]` while appending `live_progress.csv`. Losses and previously evaluated keys were finite; this is an operational filesystem failure, not a scientific failure.
- `scripts/48_run_s10_fold_trial.py` was changed only to retry bounded `PermissionError`/`BlockingIOError` failures while initializing/appending progress and atomically replacing status JSON. Dataset, split, model, forward map, physics, loss, optimizer, scheduler, seed, epoch budget and selection key are unchanged. Persistent failures are reraised after eight attempts.
- The exact code-path equivalence boundary is recorded in `audits/S10_TRANSIENT_IO_RETRY_EQUIVALENCE_AUDIT_V1.json` and its hashed diff. The independent S10 auditor admits only the previous repaired trainer hash and the new I/O-retry hash under that audit; legacy R4 physics remains excluded.
- `py_compile`, bounded retry behavior, persistent-error propagation and trainer-provenance classification tests passed. Recovery may preserve the epoch-29 partial directory and recompute only the same R6 outer-fold-3 inner-fold-3 identity. S11 remains unauthorized.

# 2026-08-12 - Fourth matched R6 outer pair after bounded I/O recovery (diagnostic only)

- The exact interrupted R6 inner identity was recomputed after preserving its epoch-29 partial directory. The repeated loss at epoch 29 matched the interrupted run to the displayed precision, the run surpassed the former failure point, and inner fold 3 completed with a finite, zero-violation selection key. The four inner folds selected common epoch 85 using only outer-training partitions; outer targets were not used.
- Physics and its capacity-matched control both completed at epoch 85 with 279,753 parameters on the same 13 outer trajectories. Both produced finite outputs, exact hard BC, zero measured causality violation, and no outer-target checkpoint or hyperparameter selection.
- Physics reduced pooled displacement L2 by 11.54%/14.10%/21.10% and velocity L2 by 8.63%/0.78%/9.40% on X/Y/Z. Displacement P90 improved by 4.15%/14.92%/24.61% on X/Y/Z.
- Worst-case displacement improved by 12.79% on Y and 30.55% on Z, but worsened by 2.03% on X, just beyond the frozen 2% non-inferiority tolerance. Median equilibrium residual decreased from 3.094740 to 0.011497, a 99.63% reduction.
- This fold is a favorable but not uniformly non-inferior diagnostic. Across four completed R6 outer folds, the pooled all-axis displacement non-inferiority indicator passes folds 1 and 3 and fails folds 0 and 2. Twelve of fifteen planned physics/control pairs are complete; no promotion is authorized before fold 4, exact-once 68-case OOF aggregation, common-split B2, paired bootstrap, and the frozen non-compensatory gate.

# 2026-08-12 - Restoration of gated downstream watchers after the operational interruption

- A live-process audit found the S10 campaign parent and current fold child healthy, but none of the previously prepared downstream waiting processes remained alive. Without restoration, S10 could finish without automatically advancing through common-split B2, independent OOF audit, promotion, conditional S11, diagnostics, and final packaging.
- The existing audited watchers `54`, `58`, `61`, `64`, `68`, `80`, `83`, and `85` were relaunched hidden. No scientific computation was opened by this action: their status artifacts confirm they are waiting respectively for the admitted S10, promotion, S11, S12, or negative-result gates.
- The normal five-seed branch and the negative-result branch remain mutually gated by the S11 outcome. The superseded standalone S12 paired-field, modal, and graph-utility watchers were not relaunched because `80_run_s12_sequential_extension_pipeline.py` is their declared replacement.
- S11 remains unauthorized while S10 is incomplete. No sensor data, new FEM simulation, Rev7/Rev8 evidence, or seventh family was opened.

# 2026-08-12 - Fifth matched repaired-R4 outer pair (diagnostic only)

- All four outer-fold-4 inner partitions completed with finite results and zero-violation checkpoints. The frozen componentwise-maximum selection rule chose common epoch 80 using only the four outer-training partitions; its aggregate key had zero violations and displacement-X L2 0.088625.
- Repaired R4 physics and its capacity-matched control both completed at epoch 80 on the same 13 outer trajectories with 377,760 parameters, finite outputs, exact hard BC, zero measured causality violation, and no use of outer targets for checkpoint or hyperparameter selection.
- Physics reduced pooled displacement L2 by 7.63%/10.49%/30.42% on X/Y/Z. Displacement P90 and worst-case error improved on all three axes.
- Velocity changed by -0.56%/-1.45%/+5.89% improvement on X/Y/Z; negative values denote degradation. The X/Y velocity degradations remain within 2%, while Z improves.
- Median equilibrium residual decreased from 1.771745 to 0.025822, a 98.54% reduction. The repaired pH-OpInf fit remained converged and structurally admissible.
- R4 therefore supplies a favorable fifth-fold diagnostic. Across its five outer folds, the all-axis pooled-displacement non-inferiority indicator passes folds 0, 2, and 4 and fails folds 1 and 3. Thirteen of fifteen planned pairs are complete. Promotion remains blocked until R2/R6 fold 4, exact-once 68-case OOF aggregation, common-split B2, paired bootstrap, and the frozen non-compensatory gate are complete.

# 2026-08-12 - Fifth matched R2 outer pair (diagnostic only)

- All four outer-fold-4 inner partitions completed with finite, zero-violation checkpoints. The componentwise-maximum rule selected common epoch 70 using only outer-training partitions; its aggregate displacement-X L2 was 0.075517. The individual best epochs were 90/90/85/75, confirming that the common epoch was not selected by majority or by the easiest fold.
- R2 physics and its capacity-matched control both completed at epoch 70 on the same 13 outer trajectories with 396,512 parameters, finite outputs, exact hard BC, zero measured causality violation, and no use of outer targets for selection.
- Physics degraded pooled displacement L2 by 22.10%/10.63%/33.57% on X/Y/Z and degraded velocity by 0.50%/0.84%/1.92%. Displacement P90 worsened on all axes; worst-case error worsened on X/Y and improved 4.97% on Z.
- Median equilibrium residual decreased from 1.944182 to 0.130832, a 93.27% reduction. That physical gain cannot compensate the non-inferiority failures on all three primary displacement axes.
- R2 fold 4 is a negative predictive diagnostic. Across five outer folds, the all-axis pooled-displacement non-inferiority indicator passes folds 1 and 2 and fails folds 0, 3, and 4. Fourteen of fifteen planned pairs are complete; R2 remains unpromoted pending the complete exact-once OOF/B2/bootstrap decision.
