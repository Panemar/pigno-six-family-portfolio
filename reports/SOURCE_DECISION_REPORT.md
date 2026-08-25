# S3 — decisión de fuentes y transferencia

Estado: `PASS_S3_SOURCE_TRANSFER_AUDIT_WITH_BLOCKERS`

## Dictamen

Las seis rutas tienen una base metodológica primaria identificable, pero ninguna fuente externa constituye validación directa del puente ferroviario 3D Timoshenko. La adopción queda limitada a mecanismos concretos y verificables. No se copiarán arquitecturas, pérdidas, pesos ni formulaciones de cargas sin superar las puertas del modelo FEM/COMSOL actual.

La campaña no abre una séptima familia. EGNN, marcos locales completos, MGNO, DeepONet/MIONet y generalized-alpha son mecanismos internos, reparaciones o controles dentro de las seis rutas congeladas.

## Decisiones por ruta

### R1 — Bridge-PINO

El paper de Chen et al. cambia tres decisiones: entradas físicas múltiples, pérdidas/diagnósticos separados y un comparador temporal recurrente. No autoriza usar la respuesta FEM previa como entrada obligatoria, derivar aceleraciones con diferencias centrales ni copiar matrices VBI/Newmark. El código PINO/TFNO archivado se utilizará solo como patrón auditable; se reimplementarán interfaces compatibles con los datos del puente.

### R2 — MO-PIGNO

Se conserva el conocimiento previo de operadores/cabezas especializadas y acoplamiento defect-aware. Esta ruta no recibe una arquitectura externa nueva. La física se admitirá por BC hard, modalidad, consistencia integral o residual compatible; cada término debe demostrar un piso sobre la propia referencia FEM.

### R3 — Graph Neural Galerkin

Se adopta la separación de Gao et al. entre mapa de campo gráfico y ensamblaje variacional por elementos. El residual se derivará nuevamente para barras Beam de seis GDL; los ejemplos escalares o estáticos del repositorio no son transferibles. Yamazaki et al. refuerza la exigencia de espacio discreto compatible, pero no valida dinámica Timoshenko 3D.

### R4 — port-Hamiltonian OpInf

Se adopta el problema restringido de inferir operadores reducidos con estructura antisimétrica/disipativa y puerto de entrada. La ruta queda condicionada a cerrar estado, derivada, energía e input en el mismo espacio reducido. El residual neuronal comienza en cero y solo se abre después de que el OpInf lineal forzado pase estabilidad, energía y rollout.

### R5 — GNO rotacional multiescala

EGNN aporta pruebas de equivariancia, no el contrato polar/axial. Los marcos Beam auditados serán la representación principal; el tratamiento de reflexión debe distinguir traslaciones polares y rotaciones axiales. MGNO aporta una jerarquía fino–grueso–fino como reparación de alcance, pero no se presentará como arquitectura rotacional ni como evidencia estructural.

### R6 — Ritz/Krylov dependiente de cargas

Los vectores Ritz y SOAR cambian la construcción de bases, no la referencia numérica. Toda base será reconstruida dentro de cada inner-train usando únicamente direcciones de carga permitidas. Los operadores reducidos Timoshenko existentes son auditores y propagadores candidatos; no se reetiquetan como matrices transitorias completas de COMSOL.

## Código externo

- `chenrongxiu-be/PINO`, `chenrongxiu-be/TFNO` y `Jianxun-Wang/graphGalerkin`: limpios salvo `desktop.ini`; uso de patrones, no copia mecánica.
- `neuraloperator/physics_informed`: bloqueado; el worktree archivado registra borrado masivo y no es ejecutable como fuente confiable.
- `ImprovedDeepONets`: no se encontró archivo de licencia en el checkout; se permite usar el paper para una implementación independiente, no copiar código.

## Autoridad histórica

C1B_02 es evidencia OOF final válida de un resultado negativo de utilidad: cumplió no inferioridad, pero no obtuvo ganancia material frente a B2. B2 continúa como surrogate autoritativo. Esta distinción corrige la ambigüedad del ledger sin reetiquetar el resultado histórico.

## Bloqueadores antes de entrenar

1. Congelar `PORTFOLIO_DEFINITION.json` y contratos comunes S4.
2. Cerrar el contrato del estado base/incremental y la semántica de inputs de fuerza que cada ruta realmente puede usar.
3. Definir pisos oracle y físicos S5 por fold interno, sin acceso al outer fold.
4. Implementar pruebas unitarias de BC, causalidad, marcos, energía, virtual work y bases fold-local.

Hasta entonces: `training_authorized=false`.
