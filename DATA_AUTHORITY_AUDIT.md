# Auditoría de autoridad numérica y de datos

## Decisión

`PASS_BRANCH_O_AS_SINGLE_CAMPAIGN_AUTHORITY`

La autoridad numérica única es la solución del modelo FEM implementado y resuelto en COMSOL. COMSOL no se trata como un modelo separado del FEM.

## Rama O — admitida

- 68 filas físicas y 68 `case_id` únicos.
- 4 regímenes 0T y 64 casos cargados.
- 32 casos cargados a 40 km/h y 32 a 52 km/h.
- Cero rutas obligatorias faltantes y cero fallos reportados de identidad, hash, forma, tiempo, ejes, finitud o BC en el manifiesto V5/V5-R.
- Autoridad: MPH original resuelto, solución `dset3`, lectura solamente, sin recomputación.
- Exposición histórica: sí. Test ciego: no.

Fuentes de autoridad: `v5r_mo_pigno_final_campaign/ORIGINAL_BRANCH_MANIFEST.json` y `dynamic_full_graph_flow_pigno_v5/registry/V5_CASE_QUALITY_CHECKS.csv`.

## Rama C — excluida

Se localizaron 2 informes resueltos de 68 esperados. El estado es `INCOMPLETE_EXCLUDED_FROM_PORTFOLIO_MODEL_DEVELOPMENT`. No se completará con datos de Rama O, no se usará para scalers, bases, targets, HPO ni OOF. Rev7/Rev8 permanecen fuera de la evidencia científica.

## Contrato de no mezcla

Cada run deberá registrar `authority_branch=O`, hash del universo de casos, hash del grafo, split, scalers y base. Cualquier ruta que apunte a artefactos Rev7/Rev8 o `corrected_graph_*` fallará el preflight. No se copiarán etiquetas históricas CAL/DEV/TEST como partición actual.

## Bloqueos

No existe conflicto que impida diseñar el portafolio con Rama O. Sí permanecen bloqueados: sensores, panel FEM nuevo, modificación/re-solución de MPH y cualquier claim de test ciego.
