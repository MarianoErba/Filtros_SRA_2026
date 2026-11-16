# Nivel III — Calibración y mediciones (filtro + ecualizador pasatodo)

Diseño y procesamiento: `notebooks/06_Ecualizador_Nivel_III.ipynb`. Todos los CSV se guardan **sin modificar** en esta carpeta, y las capturas
de pantalla en `fotos/`. Los valores medidos se anotan en `metadatos.json`.

## Circuito del ecualizador (nodos del notebook 06)

| Elemento | Conexión | Valor |
|---|---|---|
| R1e | entrada (6) → nodo A (7) | 4,7 kΩ + 330 Ω |
| C1e | nodo A (7) → pin 2 (−) | 47 nF (medido: 47,33 nF) |
| C2e | nodo A (7) → pin 6 (salida) | 4,7 nF (medido: 4,78 nF) |
| R2e | pin 2 (−) → pin 6 (salida) | 24 kΩ + 1,2 kΩ |
| Ra | entrada (6) → pin 3 (+) | 10 kΩ |
| Rb | pin 3 (+) → 18 kΩ → preset de 10 kΩ (resistencia variable) → masa | punto de ajuste ≈ 4,75 kΩ |
| TL081 | pin 7 = +12 V, pin 4 = −12 V, 100 nF a masa en cada uno; pines 1, 5 y 8 sin conectar | |

## Valores esperados (para controlar en el momento)

| Medición | Esperado |
|---|---|
| Consumo | unos 3 mA por rama el filtro (TL072) y 1,5 mA el ecualizador (TL081) |
| Tensión continua de salida, entrada a masa | pocos mV en el nodo 6 y en el 9 |
| Ganancia del ecualizador solo | −3,16 dB (k = 0,695) en todas las frecuencias |
| Fase del ecualizador solo | 0° en baja frecuencia, **−180° a 940 Hz** (f0) y −360° en alta frecuencia |
| Ganancia de la cascada | −3,2 dB a 100 Hz, unos −3,8 dB a 1 kHz y −39 dB a 4 kHz |

## 0. Preparación

1. **Fuente (OWON ODP3063):** ±12 V con límite de corriente de unos 50 mA por canal. Medir las dos tensiones con el UT61D+ y anotarlas.
2. **Placas sin alimentar:**
   - medir Rb + preset entre el pin 3 y masa, con el ecualizador desconectado del filtro y sin alimentación. Sirve para calcular k después del
     ajuste; este primer valor es antes de ajustar;
   - si las resistencias no se midieron antes de soldarlas, medir R1e en el circuito, entre la entrada y el nodo A. R2e no se puede medir en el
     circuito, porque el AO queda en paralelo.
3. **Interconexión:** la salida del filtro (nodo 6) va a la entrada del ecualizador con un cable corto. Las masas de las dos placas se unen en un
   solo punto, en la fuente.
4. **Primer encendido:** sin señal, controlar el consumo y la tensión continua en los nodos 6 y 9. Mirar la salida del ecualizador en el osciloscopio
   con 1 µs/div, para descartar oscilaciones de alta frecuencia.
5. **Puntas ×10:** compensarlas con la señal de calibración del osciloscopio. Usar el resorte de masa corto en lugar del cable con cocodrilo, y la
   misma atenuación en todos los canales.

## 1. Calibración de canales del Bode → `bode_calibracion.csv`

CH1 y CH2 en **el mismo punto**: la salida del generador, con la entrada del filtro como carga. Correr el Bode con exactamente los mismos ajustes
que se usan después: **10 Hz – 100 kHz, 5 Vpp** y la misma cantidad de puntos por década que en la medición del filtro. El resultado ideal es
0 dB y 0°; lo que aparezca es el error del instrumento entre canales, y el notebook lo descuenta de todas las mediciones de la sesión.

## 2. Filtro solo (re-medición en la misma sesión) → `bode_filtro.csv`

CH1 en la entrada del filtro y CH2 en su salida (nodo 6), con el ecualizador conectado. Así el filtro solo y la cascada se comparan con las
mismas puntas, la misma calibración y la misma carga.

## 3. Ajuste del preset (magnitud plana del ecualizador)

El ecualizador se alimenta con el generador: senoidal de 5 Vpp en el nodo 6, con el filtro desconectado o con su salida desconectada del
ecualizador. CH1 va a la entrada del ecualizador y CH2 a su salida.

1. **Ajuste grueso:** correr un Bode rápido de 100 Hz a 5 kHz. Si la ganancia tiene un **valle** alrededor de 940 Hz, bajar la resistencia del
   preset; si tiene un **pico**, subirla. Repetir hasta que el valle o pico sea menor que unos 0,2 dB.
2. **Ajuste fino:** en el modo normal, configurar la adquisición en *High Res* (o promedio de 16) y medir Vrms de CH1 y CH2 con estadísticas.
   Calcular el cociente CH2/CH1 a **100 Hz** y a **940 Hz**, sin tocar las escalas entre una frecuencia y otra, y ajustar el preset hasta que los
   dos cocientes difieran en menos de 0,5 % (0,05 dB). La sensibilidad es alta: 1 % de error en k da 0,6 dB a 940 Hz.
3. **Control de f0:** con el mismo montaje, buscar la frecuencia en la que la salida está exactamente invertida respecto de la entrada (fase de
   −180°; en modo XY, la elipse se cierra en una recta de pendiente negativa). Debería estar entre 920 Hz y 960 Hz; anotar el valor.
4. Anotar el cociente final a 100 Hz, que es k. Al terminar, con la placa sin alimentar y desconectada, medir Rb + preset entre el pin 3 y masa.

## 4. Ecualizador solo → `bode_ecualizador.csv`

Mismo montaje que en el paso 3, Bode de 10 Hz a 100 kHz, 5 Vpp. Con este archivo el notebook ajusta f0, Q, k y el error de magnitud. Si f0 o Q se
apartan más de 2 % del diseño, indica qué resistencias cambiar; después de cambiarlas, se repiten los pasos 3 y 4.

## 5. Cascada (filtro + ecualizador) → `bode_cascada.csv`

CH1 en la entrada del filtro y CH2 en la salida del ecualizador (nodo 9). Bode de 10 Hz a 100 kHz, 5 Vpp. Es la medición principal del Nivel III.

## 6. Respuesta a onda cuadrada → `escalon_cascada.csv`

Cuadrada de 50 Hz y 2 Vpp en la entrada del filtro, con CH1 en la entrada, CH2 en el nodo 6 (salida del filtro) y CH3 en el nodo 9 (salida del
ecualizador). Configurar la adquisición en *High Res*, base de tiempo de 500 µs/div y disparo por flanco ascendente de CH1. Ajustar las escalas
verticales para que cada señal ocupe casi toda la pantalla, porque así hay menos error de cuantización. Exportar los tres canales en un solo CSV.

## 7. Cierre

- Tensión continua en los nodos 6 y 9 con la entrada del filtro a masa.
- Capturas de pantalla de cada medición, en `fotos/`.
- Completar `metadatos.json`: tensiones de alimentación, Rb + preset antes y después del ajuste, cociente final a 100 Hz, frecuencia de −180°,
  offsets y observaciones.

## Archivos de la sesión

| Archivo | Contenido |
|---|---|
| `bode_calibracion.csv` | CH1 y CH2 en el mismo punto: error entre canales, que el notebook descuenta |
| `bode_filtro.csv` | Filtro solo (también lo usa el notebook 04) |
| `bode_ecualizador.csv` | Ecualizador solo, montaje final |
| `bode_cascada.csv` | Filtro + ecualizador, montaje final |
| `escalon_cascada.csv` | Onda cuadrada: CH1 entrada, CH2 salida del filtro, CH3 salida del ecualizador (captura sin *High Res*) |
| `sintonia/` | Mediciones intermedias: ecualizador en el montaje inicial (`bode_ecualizador_antes_correccion.csv`) y tras la primera corrección (`bode_ecualizador_v2.csv`), que usa la tabla de sintonía del informe; repeticiones previas del filtro, la cascada y la onda cuadrada, y una medición del ecualizador descartada por ruido de conexión (`bode_ecualizador_descartado.csv`) |
| `fotos/` | Capturas de pantalla del osciloscopio |
