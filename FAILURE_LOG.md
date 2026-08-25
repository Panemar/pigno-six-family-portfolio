# Registro de fallos

## 2026-08-11 — S12 orchestration omitted F07-F16 and explicit visual review

- The existing S12 core and extension pipelines covered F17-F43 but never invoked the regenerated historical F07-F16 generator, so the S14 completeness gate would inevitably fail.
- The historical visual-audit script also hard-coded a manual-review claim; automated hash checks alone cannot prove readability, clipping, or visual-semantic correctness.
- The orchestration now generates F07-F16, performs structural source/hash QA, creates contact sheets for F01-F43, and leaves S14 blocked until an explicit agent visual-review artifact exists. F44-F45 have the same separation before packaging.

## 2026-08-11 — S10 generic monitor counted invalid legacy R4 physics

- The read-only monitor counted all report files and therefore reported 17 completed inner runs and 6 outer runs, despite ten R4 physics reports belonging to the superseded Newmark implementation.
- The monitor now requires repaired R4 identity, finite/converged pH fit, full gradient rank and dissipativity. The corrected counts were 9 valid inner runs and 5 valid outer runs; only two complete physics/control outer pairs existed.
- No trainer, checkpoint, prediction or active process was modified.

## 2026-08-11 — Stale R4 sources detected in the future F07-F16 generator

- Pre-execution inspection found that F10 still referenced `S8_RUN_REGISTRY_V2.csv`, F11 referenced the superseded S9 final audit, and broad S9 globs could include both legacy R4/Newmark and repaired pH-OpInf physics reports.
- No stale figure was generated. The generator was corrected to use `S8_RUN_REGISTRY_V3_REPAIRED_R4.csv`, `S9_MULTIFIDELITY_FINAL_AUDIT_V2_REPAIRED_R4.json`, and an explicit R4 repaired-run identity filter.
- A read-only source check returned 64 low-fidelity physics reports and 32 high-fidelity paired reports with zero legacy R4 physics records.

## S5-INPUT-001 — broadcasting incorrecto en el auditor VDS

- Estado: `AUDITOR_IMPLEMENTATION_FAILURE_PRESERVED_AND_REPAIRED`.
- Artefacto fallido: `failed_attempts/causal_inputs_68_branch_o_v1_MASK_BROADCAST_FAILURE.h5`.
- SHA-256: `223B2DD68280A03650EF68E1E825D80F2E7CC2163B83908D971E5767805C81D9`.
- Causa: una máscara `(caso,vía)` fue indexada directamente sobre datos `(caso,tiempo,vía,carga,momento)`.
- Impacto científico: ninguno; el fallo ocurrió después de construir el VDS y antes de admitirlo. Las fuentes no fueron modificadas.
- Reparación: broadcasting explícito sobre tiempo, carga y momento; repetición de las mismas pruebas.

## S2-QA-001 — igualdad flotante exacta aplicada al mapeo de sensores

- Estado inicial: `FAIL_AUDITOR_GATE_TOO_STRICT`.
- Evidencia: error máximo de correspondencia geométrica `7.105427357601002e-15 m`; todas las identidades de entidad `S1=212, S2=57, S3=99, S4=146` y las restantes pruebas del grafo pasaron.
- Causa: el auditor v1 exigió `error == 0.0` en punto flotante.
- Corrección: tolerancia absoluta `1e-10 m`, sin modificar grafo, datos, nodos, coordenadas ni hashes.
- Evidencia preservada: `audits/S2_ACTIVE_BEAM_GRAPH_NUMERICAL_QA_v1_STRICT_ZERO_FAILURE.json`, SHA-256 `EC7878469359B930DD483054BBC9CF602E3273731D93D96D993A6B627E7FACC6`.
- Impacto científico: ninguno; el error observado es cinco órdenes de magnitud inferior a la tolerancia corregida y compatible con redondeo numérico.

## 2026-08-10 — S6 R1 smoke preflight used the CPU-only system interpreter

- Attempt: `S6_R1_BRIDGE_PINO_CAPACITY_DATA_ONLY_SMOKE_E1_V1`.
- Failure: the system `python` exposed a PyTorch build with `torch.cuda.is_available() == False`; the frozen capacity contract requires `cuda:0`.
- Scientific work performed: none. The process stopped before model construction, optimization, checkpointing or prediction.
- Preserved evidence: the empty run directory remains under `s6_capacity_runs`; it is not a scientific trial and will not be overwritten.
- Repair: use the previously installed environment at `C:\Users\yunim\Documents\BRIDGE\pigno_dynamic_vscode_pipeline_v1_2\.venv\Scripts\python.exe`, verified as PyTorch `2.11.0+cu128` on the NVIDIA GeForce RTX 5050 Laptop GPU. The repeated smoke test has a new run revision.

## 2026-08-10 — S6 R1 CUDA memory-counter API preflight incompatibility

- Attempt: `S6_R1_BRIDGE_PINO_CAPACITY_DATA_ONLY_SMOKE_E1_V2`.
- Failure: PyTorch CUDA was available, but `torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))` raised `RuntimeError: Invalid device argument` in this installed build.
- Scientific work performed: none. The process stopped before loading the capacity data and before optimization.
- Preserved evidence: the empty V2 run directory is retained and not reclassified as a trial.
- Repair: pass the CUDA device index `0` to the memory-counter API; model tensors remain assigned to `cuda:0`. A new V3 smoke identity is required.

## 2026-08-10 — S6 R1 peak-memory reset remains unsupported with integer device

- Attempt: `S6_R1_BRIDGE_PINO_CAPACITY_DATA_ONLY_SMOKE_E1_V3`.
- Failure: `torch.cuda.reset_peak_memory_stats(0)` produced the same API error before data loading.
- Independent GPU witness: a 2048 x 2048 matrix multiplication completed on `cuda:0`; PyTorch reported one device, current device 0 and finite output. `nvidia-smi` identified the RTX 5050 Laptop GPU.
- Repair: remove only the unsupported reset call. Continue to record `torch.cuda.max_memory_allocated()` without an explicit device; this API was independently verified in the same CUDA witness. V4 is a new smoke identity.

## 2026-08-10 — S6 R1 progress schema omitted the finite flag

- Attempt: `S6_R1_BRIDGE_PINO_CAPACITY_DATA_ONLY_SMOKE_E1_V4`.
- Failure: the epoch-0 evaluation completed on GPU, but CSV writing rejected the `finite` metric because it was missing from the declared columns.
- Scientific boundary: no optimizer step occurred. The epoch-0 checkpoint and partial log are preserved, but V4 is an implementation preflight failure rather than a capacity trial.
- Repair: add the existing `finite` diagnostic to the immutable row schema and rerun under V5 without overwriting V4.

## 2026-08-10 — S6 R3 temporal adapter retained NumPy float64

- Attempt: `S6_R3_GRAPH_GALERKIN_CAPACITY_DATA_ONLY_REP_PETROV_PHYSICAL32_SMOKE_E1_V1`.
- Failure: the normalized temporal input inherited NumPy `float64`, while the PyTorch model parameters and graph tensors use `float32`; the first linear layer stopped with `mat1 and mat2 must have the same dtype`.
- Scientific work performed: none. The process stopped during the first forward pass, before an optimizer step, checkpoint or prediction.
- Preserved evidence: the V1 run directory and traceback remain as an operational preflight failure and are not a capacity trial.
- Repair: cast the normalized temporal array explicitly to `float32` without changing values, architecture, data, physics or budget; rerun under revision V2.

## 2026-08-10 — S6 R4 decision auditor encountered a historical schema field

- Attempt: first execution of `scripts/16_audit_s6_r4_capacity_family.py`.
- Failure: the two base R4 reports predate the `optimization_repair` field and raised `KeyError` during registry construction.
- Scientific work affected: none; all checkpoints, predictions, metrics and run reports remain unchanged.
- Repair: interpret a missing field as the historically correct value `none`; rerun the auditor without retraining or rewriting source reports.

## 2026-08-10 — S6 micropanel HDF5 unordered fancy-index preflight

- Attempt: first execution of `scripts/26_build_s6_six_case_micropanel_dataset.py`.
- Failure: h5py rejected the selected six case indices because fancy indices were not monotonically increasing.
- Scientific work affected: none; the process stopped before creating the temporary output HDF5 and before training.
- Repair: read the immutable 68-case causal arrays and apply the frozen case order in memory. The repeated builder completed under the same protocol.

## 2026-08-10 — S6 one-case residual basis failed the six-case representation floor

- Attempt: reuse the one-case full-grid rank-224 basis as the common micropanel decoder.
- Failure: the transverse displacement oracle floor reached `0.1463553656`; direct versus observation-inferred q errors exceeded `1.94` in every selected 2T case.
- Scientific impact: material. Training on that decoder would misclassify a basis limitation as an architecture failure.
- Evidence: `s6_micropanel_common/S6_SIX_CASE_MICROPANEL_DATASET_REPORT.json`.
- Decision: preserve the failed dataset and prohibit training with its inferred q/qdot as physical targets.

## 2026-08-10 — S6 coupled Physical32 plus orthogonal residual representation failed

- Attempt: `scripts/27_build_s6_multifield_observation_basis.py`.
- Failure: forcing the three observed translation axes to identify one common Physical32 state yielded displacement oracle error up to `0.3974331529` and q32 errors above `4.96`.
- Scientific impact: demonstrates non-identifiability of a full six-DOF state from the translation-only observation operator; it does not invalidate Physical32 itself.
- Preserved evidence: `s6_micropanel_common/S6_MULTIFIELD_OBSERVATION_BASIS_REPORT.json` and its HDF5.
- Repair: separate the sparse audited physical state from specialized observation-field heads; do not impose latent identity.

## 2026-08-10 — S6 dual representation R64 exposed velocity-only rank limitation

- Attempt: `scripts/28_build_s6_dual_state_field_representation.py`.
- Result: displacement passed with maximum oracle error `0.003126`, but velocity R64 reached `0.092708` in X and `0.054655` in Y.
- Scientific impact: none on the frozen displacement anchor; the failure is isolated to the secondary velocity head.
- Repair: retain displacement R64 and Physical32 unchanged; increase only the velocity observation rank to 128. The resulting artifact passed with maximum velocity oracle errors `[0.012205, 0.012981, 0.001282]`.

## 2026-08-10 — S6 R2 causal audit was nondeterministic on CUDA

- Attempt: `S6_MICROPANEL_R2_MO_PIGNO_DATA_ONLY_CONTROL_V1`.
- Failure: repeated scatter accumulation on CUDA changed the recomputed prefix by approximately `7.6e-6`, exceeding the frozen `1e-7` causality tolerance even though no future sample entered the causal convolution.
- Scientific impact: the V1 result was not admitted; no gate was relaxed.
- Repair: enable deterministic PyTorch algorithms and deterministic cuDNN settings, then rerun under `V2_DETERMINISTIC`. The repeated audit returned zero prefix change.

## 2026-08-10 — S6 R4 fixed physical anchor has no trainable physics gradient

- Attempt: `S6_MICROPANEL_R4_PORT_HAMILTONIAN_OPINF_PHYSICS_INFORMED_V1_DETERMINISTIC`.
- Failure: the Newmark/port-Hamiltonian-compatible Physical32 anchor is fixed by construction, so its physics loss does not require gradients with respect to the observation-head parameters; the diagnostic auditor incorrectly assumed otherwise.
- Scientific impact: no optimizer step was accepted from the failed preflight.
- Repair: record an exact zero physics-gradient norm for architecture-imposed fixed anchors and retain the data-gradient audit. The rerun is `S6_MICROPANEL_R4_PORT_HAMILTONIAN_OPINF_PHYSICS_INFORMED_V2_FIXED_ANCHOR_GRADIENT_AUDIT`.

## 2026-08-10 — S7 cosine optimization repair was rejected in all six families

- Intervention: identical five-epoch warm-up followed by cosine learning-rate decay to 5% of the base rate, without changing architecture, data, seed or 150-epoch budget.
- Result: no repaired run satisfied both a primary pass and at least 2% improvement in the pooled displacement sum over its fixed-rate physics counterpart.
- Material observations: R2-R5 lost the primary X gate; R1 degraded pooled displacement sum by 8.45%; R6 improved it by only 1.05%; R3 improved it by 7.15% but still failed the absolute primary gate.
- Decision: preserve all six negative runs, adopt none, and retain the fixed-rate checkpoints for S8. Do not infer that every scheduler is ineffective; only this frozen cosine intervention was tested.

## 2026-08-11 — S8 V1 used a compensatory checkpoint score

- Detection: during `S8_FACTORIAL_R2_MO_PIGNO_PHYSICS_INFORMED_SEED_20260810_V1`, epoch 120 had pooled displacement X/Y/Z near `0.0992/0.0638/0.0517`, whereas epoch 150 degraded X to `0.1102` but was selected because velocity and Physical32 terms reduced the summed score.
- Violation: a checkpoint failing a primary displacement gate could outrank a passing checkpoint, contrary to the frozen non-compensatory contract.
- Action: stop only the S8 V1 runner after three complete trials and one incomplete fourth trial; preserve all V1 artifacts and logs.
- Repair: lexicographic checkpoint selection. Primary-passing checkpoints dominate failing checkpoints; pooled, P90 and worst-case displacement are ordered before velocity or physical tie-breakers.
- Verification: `S8_FACTORIAL_R2_MO_PIGNO_PHYSICS_INFORMED_CHECKPOINT_SELECTION_SMOKE_V2` completed with a serialized selection key. The complete campaign is relaunched under distinct V2 identities.

## 2026-08-11 — S10 outer-fold representation BC roundoff

- `S10_OUTER_0_REPRESENTATION` met the displacement and velocity oracle floors, but failed the exact hard-BC basis gate because full-matrix float32 QR reintroduced fixed-row values of order `3.7e-8` to `8.9e-8`.
- No training was started. The remaining 24 representation fits were stopped.
- Representation repair: perform QR only on free observation rows, then embed the orthonormal basis into an exactly zero fixed-row array.
- Failed artifact retained as `failed_attempts/S10_OUTER_0_REPRESENTATION_failed_bc_roundoff_v1.h5`.

## 2026-08-11 — Transient G: path-resolution interruption during S10 R2 inner fold 3

- One read-only poll returned `PathNotFound` for `s10_nested_grouped_oof/campaign_status.json` while the G: filesystem drive itself remained mounted.
- Immediate independent checks showed the S10 parent, wrapper and child Python processes still alive; the child continued accumulating CPU time and had no stderr.
- The project root, package and status file became readable again on the next check. The status JSON was valid, the active run advanced to epoch 0, and no training process was restarted or modified.
- Classification: transient Google Drive/path-resolution interruption, not demonstrated data corruption. Continue monitoring; treat recurrence with process inactivity or invalid/missing artifacts as a separate operational failure.

## 2026-08-11 — Two rejected launches of the post-S10 scientific-decision watcher

- The first duplicate-process query matched its own PowerShell command because the command text contained the watcher filename; no watcher was launched.
- The second `Start-Process` call passed the script path without preserving quotes around the spaces in `G:\Mi unidad`; the Python wrapper exited before executing the watcher and no status artifact was created.
- Neither attempt interacted with the active S10 trainer, its parent runner, predictions, checkpoints or campaign status.
- The launch contract was corrected to quote the complete script path and to require both a live Python process and a valid watcher status JSON. The admitted watcher runs as PIDs 38456/39548 and reports `WAITING_FOR_ADMITTED_S10_OOF`.

## 2026-08-11 — Rejected F01–F06 visual/provenance attempts

- Visual QA rejected the first isometric F03, F04 and F06 exports because true span aspect compressed the axes, superposed labels and made local-frame glyphs or field values unreadable. The data were not rejected.
- A second/third display repair remained inadequate for F04/F06. The accepted repair uses true-aspect orthographic projections where the isometric view did not communicate the quantity reliably.
- The first provenance audit failed because F01–F05 manifests contained the pre-final SHA-256 of script 69 after a later F06 source repair. All six outputs were regenerated from the frozen final generator, and the repeat audit passed every generator and artifact hash.
- Every rejected file is preserved under `s12_final_diagnostics/rejected_visual_qa_v1`, `v2`, `v3` or `v4_stale_generator_hash`. No rejected attempt is counted among accepted contract figures.

## 2026-08-11 — Rejected first F07–F16 visual package

- The first historical-figure batch used six route colors, violating the frozen two-root palette plus neutral/non-color distinction contract, and F10 used black text on dark heatmap cells.
- The artifact was preserved under `s12_final_diagnostics/rejected_visual_qa_historical_v1` and is not accepted evidence.
- Repair: route identity now uses line style and marker; physics/control use blue/orange only where semantically applicable; F10 label color is contrast-aware. The repeat visual/hash audit passed.
# S10 operational interruption — 2026-08-11 14:28 UTC

- `49_run_s10_nested_oof_campaign.py` and its R6-control worker disappeared without a Python failure event after creating only epoch-0 `status.json` and `live_progress.csv`; GPU utilization was zero and no matching process remained.
- No checkpoint, prediction H5 or scientific report existed, so the incomplete directory was not admissible evidence.
- The incomplete directory was preserved at `s10_nested_grouped_oof/operational_failures/S10_OUTER_R6_LHS_04_OUTER_0_OUTER_OOF_CONTROL_SEED_20260813_INTERRUPTED_20260811T142806Z`.
- Root cause remains `UNKNOWN_EXTERNAL_PROCESS_TERMINATION`; Windows Application events did not identify a Python crash in the inspected interval.
- The evidence-aware runner was relaunched once. It skipped every completed report and restarted only the incomplete R6-control run.

## 2026-08-11 — R4 no ejecutaba el núcleo port-Hamiltoniano/OpInf congelado

- `PortHamiltonianOpInf` estaba definido y probado aisladamente, pero los entrenadores S6–S10 de R4 instanciaban solo `PortHamiltonianResidualOperator`.
- El `forward` sustituía `q`, `v` y `a` por un ancla Newmark fija; por ello `train_physics_loss` era constante y no informaba parámetros aprendibles.
- Se preservan 7 reportes R4 de capacity, 4 de micropanel, 4 de S8, 36 de S9, 11 completos de S10 y un outer parcial interrumpido, todos sin borrado.
- Clasificación: pruebas primitivas `CAPACITY_ONLY`; resultados R4 entrenados `INVALIDATED_BY_ARCHITECTURE_CONTRACT_AS_PORT_HAMILTONIAN_OPINF`.
- Se detuvo únicamente PID 25932; S10 cerró en `OPERATIONAL_FAILURE_S10` y S11 permaneció bloqueado.
- Evidencia: `audits/R4_PORT_HAMILTONIAN_ARCHITECTURE_CONTRACT_AUDIT_V1.json`.

## 2026-08-11 - Capacity reparado R4 E10: fallo posterior de serializacion

- La corrida `S6_R4_REPAIRED_PH_OPINF_CAPACITY_E10_V1` completo las 10 epocas y preservo `live_progress.csv`, `status.json` y el checkpoint.
- El proceso fallo despues del entrenamiento al intentar expresar dos fuentes externas como rutas relativas a la raiz del portafolio.
- La evidencia no se promueve: no existe `report.json` completo y los errores de desplazamiento X/Y (26,6 %/14,7 %) no cumplen la puerta congelada posterior de 10 % por eje.
- La correccion solo modifica la serializacion de procedencia. La reparacion de optimizacion autorizada aumenta el presupuesto a 30 epocas sin cambiar arquitectura, datos, split ni semilla.

## 2026-08-11 - Primer preflight S8 R4 reparado detenido antes de entrenamiento

- `S8_FACTORIAL_R4_..._REPAIRED_SEED_20260810_V1` se detuvo en el ajuste pH previo; no ejecuto ninguna epoca ni genero evidencia predictiva.
- El solver estaba finito, con rango gradiente 64 y rango conjunto 96, pero alcanzo 750 iteraciones con paso relativo 6,48e-7 frente a tolerancia 5e-7.
- La prueba dirigida mostro convergencia en 1159 iteraciones manteniendo la tolerancia, con error de derivada 0,143 %. Se amplia solo el presupuesto del solve convexo S8 a 1500; no cambian datos, loss, red, epochs ni gates.

## 2026-08-11 - S8 R4 reparado V2 detenido en epoca 5 por auditor de gradientes

- El entrenamiento activo progresaba, pero `evaluate()` usaba siempre el caso indice 1 como probe de gradientes.
- En S6 ese indice era una trayectoria activa; en S8 corresponde a un caso BASE con incremento forzado a cero. Los logs mostraban falsamente loss y gradientes cero.
- Se detuvo el proceso en epoca 5, se preservo el directorio incompleto y no se promueve ninguna de sus metricas.
- La correccion selecciona el primer caso con `case_active>0`; no modifica entrenamiento, objetivos, datos, arquitectura ni gates.

# 2026-08-11 — False-positive S11 command diagnosis caught before execution

- Inspection of only the top-level campaign runner suggested that the R4 repair flag was omitted; the delegated fold wrapper already supplied it together with the frozen epoch and outer phase.
- A transient redundant argument at the top level would have been rejected by the wrapper parser, but it was removed before S11 started. No scientific run or evidence was affected.
- Corrective action: tests now cover the complete two-level call path, and the independent S11 audit verifies the repaired pH-OpInf diagnostics instead of trusting the run identity alone.

## 2026-08-11 - Interrupcion operacional reanudable de S9 R4 reparada

- El runner `92_run_s9_repaired_r4_multifidelity.py` y su proceso hijo desaparecieron despues de que `S9_LOW_R4_LHS_03_FOLD_0_PHYSICS_REPAIRED_EFFECTIVE_PH_OPINF_SEED_20260812` escribiera un `report.json` completo, pero antes de registrar `trial_finished` y lanzar el siguiente fold.
- El reporte cerrado paso ejecucion, finitud, BC, causalidad, estado base y ajuste pH-OpInf; por tanto, no se clasifica como fallo cientifico ni se repite.
- Se verifico dos veces la ausencia de procesos y se relanzo una sola vez el runner reanudable. Este valido identidades, omitio los cinco reportes completos y comenzo exclusivamente `R4_LHS_03/fold_1`.
- Causa: `UNKNOWN_EXTERNAL_PROCESS_TERMINATION`. No existe evidencia de corrupcion de datos, checkpoint o grafo.
- La interrupcion no autoriza promocion, S10 u OOF y permanece registrada como incidente operacional.
# 2026-08-11 — Interrupción gobernada de R4 S9

La corrida R4 media fold 2 fue detenida por sustitución explícita del desarrollo secuencial por un registro paralelo de seis rutas. No fue un fallo científico ni numérico. Se conservaron los artefactos parciales y no se autorizó OOF. La reanudación deberá ejecutar solo folds faltantes mediante el runner idempotente y volver a auditar el ranking común.

# 2026-08-11 — Control R4 con hash histórico incompatible

La auditoría independiente S9 posterior a completar la R4 reparada falló únicamente `r4_control_current_trainer_hash_pass`. Los cuatro controles R4 preservados tienen capacidad equivalente, pero fueron creados con una versión anterior de `39_run_s9_fold_trial.py`. Se mantiene OOF bloqueada. La corrección autorizada es archivar esos controles como evidencia histórica y repetir solo los cuatro controles con el entrenador actual; no se repite physics ni otra familia.
# 2026-08-11 — S10 repaired R4 smoke rank deficiency

- The first repaired-R4 S10 smoke fit stopped before neural training: the inner-fold training partition supplied 65 direct snapshots for a 64-state Hamiltonian representation, but its Hamiltonian-gradient rank was only 39.
- The failed run is retained under `s10_nested_grouped_oof/operational_failures/` and is not evidence of model performance.
- The single authorized representation repair uses the largest fold-local shared generalized-coordinate subspace whose Hamiltonian gradient has full column rank. It is fitted from inner/outer training trajectories with direct reduced states only; validation and outer-OOF targets remain excluded.
- The repair keeps one shared basis for displacement and velocity, projects M/C/K and generalized forces congruently, and maps the propagated state back to the audited 32-coordinate reduced space before the observation heads.
# 2026-08-11 — Historical S12 figures invalidated by repaired R4 identity

- The existing F07–F16 package and its visual QA were produced before the effective port-Hamiltonian OpInf repair and therefore could include legacy R4/Newmark metrics.
- Fifty-two files were moved recoverably to `s12_final_diagnostics/rejected_historical_visuals_pre_r4_opinf_repair/`; no file was deleted.

# 2026-08-11 - S10 R4 epoch mismatch detected before paired audit

- Repaired R4 outer-fold-0 physics selected and executed epoch 95, but the preserved outer control used epoch 85. Treating them as a pair would violate the common-budget contract and fail the independent audit by construction.
- The campaign was stopped while repaired R4 outer-fold-1 inner-fold-0 was at epoch 2. That partial directory was moved recoverably to `s10_nested_grouped_oof/operational_failures/`; it has no completed report and is not scientific evidence.
- No accepted physics report, checkpoint, prediction, split or FEM reference was deleted or modified.
- Corrective action is limited to an idempotent resume contract: archive the incompatible epoch-85 control, rerun only that control at epoch 95, then continue from the first genuinely missing inner run.
- The active F07–F16 locations are now empty. Regeneration and fresh visual QA are mandatory after the repaired S10/S11 evidence closes.
# 2026-08-12 - S10 orchestration process terminated during R6 outer-1 inner-0

- The interactive execution session ended with Windows exit code `1073807364` while `S10_INNER_R6_LHS_04_OUTER_1_INNER_0_PHYSICS_SEED_20260813` was running.
- Independent process inspection found neither the S10 orchestrator nor its fold worker alive; `campaign_status.json` remained stale at `RUNNING`, the GPU was idle, and the last complete progress row was epoch 64/100.
- The run emitted no stderr and no `report.json`; therefore it is an operationally interrupted partial run, not a scientific FAIL and not admissible evidence.
- The partial directory is preserved under `s10_nested_grouped_oof/interrupted_partial_runs/` before exact-identity recomputation. Every prior completed report remains reusable only through the runner's identity, epoch, BC, causality, finiteness and prediction-artifact gates.
- The first restart exposed a resumability defect: the fold worker correctly rejected the still-present report-less output directory with `FileExistsError`. The orchestrator was patched to archive any report-less run directory recoverably before exact-identity recomputation; `tests/test_s10_runner_resume_contract.py` now covers this behavior and passes 4/4 tests.
- A second hidden-process restart recreated the same R6 identity from epoch 0. The new progress file, active worker chain and GPU allocation confirmed operational recovery; the archived epoch-64 partial remains excluded from scientific evidence.
- Process inspection after recovery also showed that the downstream S10-to-S14 watcher chain had died with the original interactive session. All downstream status artifacts remained in gated `WAITING` states, so no premature S11/S12/S14 work had occurred.
- The eight required watchers (54, 58, 61, 64, 68, 80, 83 and 85) were restarted as hidden independent processes. Watcher 58 was repaired to distinguish a live registered PID from a stale waiting status and now records its PID during active postprocessing. Syntax compilation passed and every downstream gate was re-observed in `WAITING` with training/tuning unauthorized.
