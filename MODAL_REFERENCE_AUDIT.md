# Auditoría inicial de referencia modal

Estado: `PASS_REFERENCE_IDENTITY__S2_CONTRACT_REBUILD_REQUIRED`

La referencia modal es la solución modal del modelo FEM implementado en COMSOL. El ensamblaje Timoshenko independiente sirve para auditar geometría, orientación, propiedades, masa y BC; no reemplaza la referencia FEM.

## Protocolo

- Normalización preferente a masa unitaria.
- Clasificación de modos globales/locales, baja masa efectiva, clústeres cercanos/degenerados y modos complejos.
- Emparejamiento por frecuencia más MAC/subspace MAC; no forzar correspondencia uno-a-uno en clústeres degenerados.
- Reportar frecuencia, error, MAC, COMAC, ángulos principales, masa efectiva, participación, energía modal, PSD, fase y coherencia por banda.

## Transferencia a las seis rutas

- R1: regularización/métrica espectral-modal.
- R2: bases y pérdidas propias por operador, con acoplamiento modal defect-aware.
- R3: test space modal o elemental dentro de trabajo virtual.
- R4: coordenadas energéticas y Hamiltoniano reducido.
- R5: jerarquía multiescala y canales de rotación/traslación.
- R6: base propia combinando modos con Ritz/Krylov dependientes de carga.

La limitación RYdot histórica se trata como problema de representación/tarea, no como invalidez automática del campo primario ni de toda una familia.
