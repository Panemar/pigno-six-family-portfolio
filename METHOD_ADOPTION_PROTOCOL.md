# Protocolo de adopción de métodos

Toda intervención se registra como `METHOD_ADOPTION_<BLOCKER_ID>.md` con: evidencia cuantitativa, ruta, fuente primaria, ecuación, derivación, unidades, supuestos, aplicabilidad, incompatibilidades, mecanismo, riesgo, modificación mínima, métrica objetivo, prueba sintética, prueba sobre referencia FEM/COMSOL, ablación y criterio de reversión.

## Puertas

1. El bloqueo debe estar demostrado, no inferido de una sola pérdida de entrenamiento.
2. La formulación debe dar defecto cero o piso medible cuando `pred=FEM`, según corresponda.
3. La consistencia dimensional y los casos límite deben pasar.
4. Física y datos se normalizan en CAL interno; outer OOF nunca reajusta pesos.
5. Máximo dos fuentes externas por bloqueo y tres intervenciones por ruta.
6. Cada ruta dispone de una reparación de representación y una de optimización antes de cierre.
7. Dos intervenciones consecutivas con mejora menor de 2 % y sin resolver una puerta física cierran esa rama, no el portafolio.

## Conflicto de gradientes

Se registrarán norma, coseno y peso efectivo por término. Se comparan primero pesos fijos normalizados; GradNorm o PCGrad se habilitan solo con conflicto medido. NTK/self-adaptive queda reservado para un finalista y no se acumula con todas las técnicas.
