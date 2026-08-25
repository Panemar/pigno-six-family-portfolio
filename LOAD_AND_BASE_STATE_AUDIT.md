# Auditoría inicial de cargas y estado base

Estado: `PASS_CONCEPTUAL_LINEAGE__S1_NUMERICAL_RECHECK_REQUIRED`

## Contrato común

Cada caso conserva su configuración propia: 0T/1T/2T, tipo de tren, 40/52 km/h, vía activa, 22 ejes por tren, fuerza y separación por eje, entrada progresiva, regularización gaussiana, rampa, frenado, viento y sismo/excitación de base cuando corresponda. `case_id` no es feature.

El estado se expresará como:

`Delta U = U_loaded - U_base`, y `U_total_pred = U_base_FEM + Delta U_pred`.

El peso propio y la carga muerta se incorporan mediante estado base, masa, gravedad, rigidez/energía y reconstrucción total; no como una columna constante sin contenido informativo.

## Controles obligatorios

- Resultante y momento de fuerzas por tiempo.
- Trabajo virtual y signo/unidades.
- Activación temporal y vía correcta.
- Nulidad de vía inactiva.
- Correspondencia posición-eje-tiempo y soporte de la gaussiana.
- Excitación de base separada de cargas nodales.
- Etiqueta explícita `incremental` o `total` en pérdidas, métricas y figuras.

## Limitación heredada

La fuerza externa completa y Uddot no están cerradas en un espacio DOF compatible; por ello no se autoriza todavía `M a + C v + K u - f = 0` como pérdida fuerte global. Las rutas usarán forma débil/tangente, energía o propagación reducida únicamente cuando la misma formulación pase primero sobre la referencia FEM/COMSOL.
