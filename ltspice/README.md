# LTspice — filtro LP Chebyshev I de orden 3 (Grupo 3)

Los netlists de esta carpeta los genera `notebooks/03_Sensibilidad_y_Validacion.ipynb` a partir de `resultados/componentes.json`.
**No editarlos a mano**: si cambia un componente, se vuelve a ejecutar el notebook 02 y después el 03.

Numeración de nodos (igual que el esquema del informe): 1 salida del generador (después de Rs = 50 Ω), 2 entrada no inversora de U1,
3 salida de U1, 4 nodo central del Sallen–Key, 5 entrada no inversora de U2, 6 salida. Los nodos 11, 31 y 41 son las uniones de las
resistencias en serie (R0a–R0b, R1a–R1b, R2a–R2b).

| Archivo | AO | Análisis |
|---|---|---|
| `filtro_ideal_ac.cir` | VCVS de ganancia 10⁶ (ideal) | AC 10 Hz – 100 kHz |
| `filtro_lineal_ac.cir` | AO lineal de un polo, GBW 3 MHz (**modelo pedagógico, no comercial**) | AC |
| `filtro_TL072_ac.cir` | Modelo PSpice de TI del TL072, ±12 V | AC |
| `filtro_TL072_tran.cir` | TL072 | Senoidal 1 V eficaz a 200 Hz, 1 kHz y 4 kHz (`.step`) |
| `filtro_TL072_escalon.cir` | TL072 | Escalón de 1 V |
| `filtro_TL072_gran_senal.cir` | TL072 | Senoidal de 4, 8 y 11 V pico en 936 Hz (máximo nivel interno) |
| `filtro_TL072_mc_E1.cir` … `_E4.cir` | TL072 | **Nivel II**: Monte Carlo AC, 500 corridas por escenario (generados por el notebook 05) |
| `ecualizador_ac.cir` | TL072 (como TL081) | **Nivel III**: ecualizador pasatodo solo, AC (generado por el notebook 06) |
| `filtro_ecualizado_ac.cir`, `filtro_ecualizado_escalon.cir` | TL072 | **Nivel III**: filtro del prototipo + ecualizador, AC y escalón |
| `filtro_mc_sondas.cir`, `filtro_mc_sondas_5000.cir` | — | Diagnóstico del generador `mc()`: mismos elementos y orden que E2, aislados; los valores sorteados se leen del `.raw` |

## Monte Carlo (Nivel II)

Cada elemento tiene como valor `{mc(x,tolR)}` o `{mc(x,tolC)}`: LTspice sortea un valor uniforme en [x(1−t), x(1+t)], distinto para cada
elemento, y lo vuelve a sortear en cada paso de `.step param run 1 500 1`. Escenarios: E1 R ±5 % / C ±10 % (ejemplo de la guía),
E2 R ±1 % / C ±10 % (prototipo), E3 R ±1 % / C ±5 %, E4 R ±1 % / C ±2 %. Las resistencias `Rpr` y `Rpc`, alimentadas con 1 A, son
**sondas** desacopladas del filtro: `V(pr)` y `V(pc)` valen exactamente el número sorteado en cada corrida y se registran con `.meas`
para verificar que las corridas son distintas y que la distribución es la declarada. Cada archivo tarda unos minutos. Copiar a
`resultados/` el `.log` y el `.raw` de cada escenario; el notebook 05 los procesa.

LTspice usa siempre la misma secuencia de números aleatorios (misma semilla): los cuatro escenarios comparten los sorteos, escalados por la
tolerancia. `filtro_mc_sondas.cir` reproduce esa secuencia con los elementos aislados y permite recuperar el valor de cada componente en cada
corrida (el notebook 05 los usa para contrastar LTspice y Python corrida por corrida). Estos dos netlists simulan un único punto de frecuencia:
LTspice guarda entonces un punto por corrida en el `.raw` (eje `run`), que el notebook lee directamente.

## Modelo del TL072 (necesario para los archivos `filtro_TL072_*`)

El repositorio incluye `TL072.301`, el macromodelo oficial de Texas Instruments (SLOJ067), sin modificaciones. Para obtenerlo de nuevo:

1. En ti.com, página del **TL072** → *Design & development* → *Simulation model* → **TL072, TL072A, TL072B PSpice Model (SLOJ067)**.
2. Descomprimir y copiar el archivo del modelo en **esta carpeta** con el nombre `TL072.301`. Si el archivo del zip tiene otro nombre,
   renombrarlo o cambiar la línea `.lib TL072.301` de los netlists.
3. Abrir el archivo y verificar que la línea del subcircuito sea `.SUBCKT TL072 1 2 3 4 5` (entrada no inversora, entrada inversora,
   V+, V−, salida). Si el nombre del subcircuito es otro, cambiar `TL072` en las líneas `XU1` y `XU2`.

## Cómo correr

1. LTspice → *File → Open* → elegir tipo de archivo *Netlists (\*.cir, \*.net, …)* → abrir el `.cir`.
2. *Simulate → Run*. Para ver curvas: clic en el nodo deseado en el visor (por ejemplo `V(6)`).
3. Los resultados de los `.meas` quedan en el *SPICE Error Log* (*View → SPICE Error Log*) y en el archivo `.log`.
4. Copiar a `ltspice/resultados/` los archivos **`.raw` y `.log`** de cada simulación (LTspice los genera junto al `.cir`). El notebook
   03 los lee y arma las comparaciones. No subir al repositorio otros temporales (`.net`, `.op.raw`, `.plt`).

Indicar en el informe la versión de LTspice usada (*Help → About*).
