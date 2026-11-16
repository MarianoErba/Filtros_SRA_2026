# Mediciones — protocolo del laboratorio (Grupo 3)

Filtro: LP Chebyshev I de orden 3, RC + seguidor y Sallen–Key de ganancia unitaria, TL072CP, alimentación ±12 V.

**Regla: nunca inventar ni "corregir" datos.** Si una medición sale rara se anota en `notas` y se repite; se entregan todas.

## Archivos

| Archivo | Uso |
|---|---|
| `plantilla_mediciones.csv` | Frecuencias del plan con los valores **esperados** (ganancia, Vout, fase, Δt). Copiarla como `mediciones.csv` y completar. |
| `mediciones.csv` | **Datos medidos.** Columnas obligatorias: `frecuencia_Hz`, `Vin_Vrms`, `Vout_Vrms`, `fase_grados` (o `delta_t_s`). |
| `metadatos_plantilla.json` | Copiarla como `metadatos.json` y completar instrumentos, alimentación, resistencias medidas, etc. |
| `hoja_laboratorio.tex` / `.pdf` | Planilla imprimible para anotar a mano en el banco. |
| `fotos/` | Fotos del montaje y de pantallas del osciloscopio (con nombre que indique la frecuencia). |

**Medición usada en el informe.** El Bode del filtro y su respuesta al escalón se tomaron en la sesión del Nivel III, con puntas ×10 y
calibración de canales: `nivel3/bode_filtro.csv`, `nivel3/bode_calibracion.csv` y `nivel3/escalon_cascada.csv` (CH1 entrada, CH2 salida del
filtro). `descartadas/` guarda una medición anterior que no se usa en el informe ni en los notebooks; se conserva sin modificar.

`notebooks/04_Mediciones.ipynb` procesa `mediciones.csv` y `metadatos.json` y genera figuras, tablas y valores para el informe.

## Antes de energizar

1. Revisar la placa con el esquema: orientación del TL072 (muesca/pin 1), pines 8 = +12 V y 4 = −12 V, capacitores de desacople.
2. Con el multímetro: continuidad de masa; **sin cortocircuito** entre +12 V, −12 V y masa.
3. **Medir y anotar las seis resistencias** (R0a, R0b, R1a, R1b, R2a, R2b) antes de soldarlas o con un extremo levantado.
4. Anotar el código impreso de cada capacitor (por ejemplo `473J` = 47 nF ±5 %; `K` = ±10 %; `M` = ±20 %).
5. Fuente partida: ajustar +12 V y −12 V **antes** de conectar, límite de corriente ~50 mA. Al conectar, el consumo esperado es ~3 mA por riel.
6. Con la entrada a masa: medir la tensión continua de salida (offset, esperado: pocos mV). Anotarla en `metadatos.json`.

## Conexión de instrumentos

- Generador → borne de entrada (nodo 1). **CH1 del osciloscopio en el nodo 1** (no en la salida del generador): así la medición es $V_6/V_1$ y no
  depende de la resistencia interna del generador.
- **CH2 en la salida (nodo 6).** Puntas ×10, con el resorte de masa corto si se puede (en la banda de rechazo la salida es de mV).
- Acoplamiento DC, límite de ancho de banda 20 MHz, **promediado** (16 adquisiciones o más) para las frecuencias con salida chica.
- Disparo por CH1 (señal grande y limpia).

## Barrido en frecuencia

Para cada frecuencia de `plantilla_mediciones.csv`:

1. Ajustar la frecuencia y la amplitud sugerida (1 V eficaz hasta 2 kHz; 3 V eficaces desde 2,5 kHz).
2. Ajustar V/div de cada canal para que la señal ocupe la mayor parte de la pantalla (mejor resolución). La columna `Vout_esperada_Vrms` indica la escala.
3. Medir **Vrms** de CH1 y de CH2 con la función de medición del osciloscopio (o Vpp y convertir, indicándolo).
4. Medir **Δt** entre los cruces por cero ascendentes de CH1 y CH2 (cursores o medición "delay"). Δt > 0 si la salida está atrasada.
   Fase: φ = −360° · f · Δt. La columna `delta_t_esperado_us` sirve para verificar que se mide el cruce correcto.
5. Anotar en `notas` cualquier particularidad (ruido, escala, promediado).

Comprobar al pasar: la salida no debe mostrar recorte ni deformación; si la hay, bajar la entrada y anotarlo.

## Ensayos adicionales (recomendados)

- **Respuesta al escalón:** onda cuadrada de 50 Hz y 1 Vpp (offset 0,5 V para que vaya de 0 a 1 V, o simétrica). Guardar la captura y, si el
  osciloscopio lo permite, exportar CSV (`escalon.csv`). Sobrepico esperado ≈ 9,7 %.
- **Gran señal:** a 936 Hz, subir la entrada hasta ver el recorte de la salida (esperado ≈ ±10,4 V con ±12 V). Anotar la amplitud de entrada.

## Incertidumbre

- Ganancia: con una precisión vertical del 3 % por canal, u(G) ≈ 0,21 dB (k = 1). Completar la precisión real del osciloscopio en `metadatos.json`.
- Fase: depende de la resolución temporal (escala de tiempo / puntos por pantalla) y del ruido; por defecto se toma u(φ) = 1°.
- En la banda de rechazo lejana (≥ 10 kHz) la salida es de décimas de mV: el ruido del osciloscopio y el acople de la masa de las puntas pueden
  dominar. Por encima de ~40 kHz, además, la atenuación real queda limitada por la impedancia de salida del AO (ver informe, sección 8).
