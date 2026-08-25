# Plan de auditoría de fuentes

Estado: `S3_PLANNED__NO_EXTERNAL_CODE_ADOPTED`

## Prioridad

1. MPH originales y documentación oficial COMSOL.
2. Papers primarios revisados por pares.
3. Repositorios oficiales asociados al paper.
4. Libros y tesis.
5. Preprints pertinentes; reviews solo para localizar primarias.

## Preguntas por fuente

Se extraerán problema, ecuaciones, geometría, discretización, entradas, salidas, cargas, BC, datos, split, baselines, pérdidas, ponderación, optimizador, métricas, código, licencia y limitaciones. Toda adopción debe cambiar una decisión concreta y demostrar compatibilidad con grafo irregular Beam, Timoshenko 3D, marcos locales, 512 observaciones, dt=0,025 s, cargas móviles y hardware RTX 5050/32 GB.

## Registros de S3

`SOURCE_INVENTORY.csv`, `SOURCE_TRANSFER_MATRIX.csv`, `SOURCE_EQUATION_CATALOG.csv`, `SOURCE_CODE_REGISTRY.csv` y `SOURCE_DECISION_REPORT.md`.

## Regla de economía científica

Máximo dos fuentes/soluciones externas por bloqueo y tres intervenciones por ruta. No se implementará una técnica porque sea novedosa; debe resolver representación, estabilidad, fase, energía, localidad, escala, modalidad, conflicto de gradientes, optimización, generalización o costo, con prueba sintética, prueba FEM, ablación y reversión.
