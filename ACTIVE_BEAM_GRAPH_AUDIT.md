# Auditoría inicial del grafo Beam activo

Estado: `PASS_TO_S1_HASH_REVERIFICATION__DYNAMIC_UTILITY_PENDING`

## Evidencia disponible

- Nodos activos: 22.164.
- Barras Beam: 24.215 no dirigidas / 48.430 dirigidas.
- Reciprocidad dirigida: 1,0.
- Componentes conexas: 1; nodos aislados: 0.
- Nodos de vía: A=1.406, B=1.408, unión=2.814.
- Nodos observados: 512.
- Fuente de topología: `Original_extractions_20260801/graph_original_v1/original_exact_timoshenko_graph.npz`.
- SHA-256 histórico: `97a064ff0ac2226f4e0c8eb6c2363799ee9a7b7a238fa131030f0060cafe2e86`.

## Contrato mecánico pendiente de reverificación S1–S2

Cada elemento debe portar conectividad, longitud, material, sección, A, Iy, Iz, J, kappa_y, kappa_z, E, G, densidad, masa lineal/aumentada, vía, apoyos y triedro local. Se verificarán ortogonalidad, determinante positivo, alineación axial y transformaciones global-local-global.

## Riesgo histórico que cambia el diseño

V4/V5 demostró que la salida cambia al perturbar el grafo, pero no que romperlo degrade sistemáticamente la predicción. Tres pasos de mensaje cubrían solo una fracción local y el token global comprimía fuertemente las cargas. Por ello, todas las rutas gráficas deben persistir activaciones, tener rama gráfica inicialmente no nula y pasar perturbaciones multicaso. Una ruta no recibirá crédito físico solo por cargar `edge_index`.

## Pruebas congeladas

Grafo correcto, dirección invertida, permutación de nodos, Iy/Iz intercambiados, propiedades neutralizadas, conectividad eliminada, marcos alterados y cargas enroutadas a vía incorrecta. Las comparaciones preservarán mismo caso, tiempo, nodo, componente, unidad y campo incremental/total.
