# Estado actual de la campaña Portafolio-PIGNO

Fecha de auditoría: 2026-08-10  
Estado: `S0_FIRST_DELIVERY_COMPLETE__NO_TRAINING_STARTED`  
Dictamen: `GO_PORTFOLIO_DESIGN`

## Autoridad y alcance

- Referencia numérica única: modelo FEM del puente implementado, ensamblado y resuelto en COMSOL.
- Autoridad de desarrollo: Rama O, 68 trayectorias físicas originales, históricamente expuestas.
- Rama C: incompleta (2/68 informes resueltos localizados) y excluida. Rev7/Rev8 no son datos científicos de esta campaña.
- No existe test ciego. La evidencia final será `nested grouped cross-validated evidence` y predicción OOF por trayectoria.
- Sensores, nuevas simulaciones FEM y modificación de MPH permanecen bloqueados.

## Evidencia preservada

Se conservan sin reetiquetar B2, C1B_02, D0, H1/H2, V3, V4, V5 y V5-R/MO, incluidos fallos de capacidad, código, checkpoints, predicciones, métricas y manifiestos. B2 continúa como control predictivo histórico; no es una PIGNO. V5-R mostró capacidad primaria sub-1 % en un caso, pero falló crecimiento tardío y no abrió OOF. Sus resultados no cierran las familias no evaluadas.

## Portafolio congelable

La campaña nueva contiene exactamente seis rutas no redundantes:

1. `R1_BRIDGE_PINO` — Bridge-PINO/multiple-input temporal operator.
2. `R2_MO_PIGNO` — operadores especializados q, v y a con acoplamiento compatible.
3. `R3_GRAPH_NEURAL_GALERKIN` — operador gráfico variacional con trabajo virtual.
4. `R4_PORT_HAMILTONIAN_OPINF` — dinámica reducida pasiva/energética más residual.
5. `R5_ROTATION_MULTISCALE_GNO` — GNO multiescala consciente de vectores polares/axiales.
6. `R6_LOAD_DEPENDENT_RITZ_KRYLOV` — base Ritz/Krylov/SOAR dependiente de cargas más residual gráfico.

No habrá `R7`. DeepONet, TFNO, generalized-alpha, POD, modal graph, MIONet, integradores y mixture-of-experts son módulos, controles o reparaciones posibles dentro de una ruta, nunca nuevas familias.

## Hechos físicos y de datos confirmados

- Grafo exacto disponible: 22.164 nodos, 24.215 barras Beam no dirigidas, una componente conexa, 512 nodos observados.
- Ejes del proyecto: X transversal, Y vertical/altura, Z longitudinal.
- La utilidad dinámica del grafo aún no está demostrada; debe probarse en varios casos con rama gráfica no nula.
- La forma fuerte global completa sigue bloqueada por falta de Uddot y fuerza externa compatibles en el mismo espacio DOF. Forma débil, tangente, modal, energética y BC hard sí pueden evaluarse cuando el piso FEM equivalente pase.
- El estado completo de tasas tiene una limitación histórica en RYdot; no puede cerrar por sí solo una ruta ni obligar a que todos los canales compartan operador.

## Próximo estado

S1–S3 deberán verificar hashes y contratos de autoridad, grafo, cargas, modalidad y fuentes. Solo después S4 congelará configuraciones concretas. Hasta entonces `training_authorized=false`.
