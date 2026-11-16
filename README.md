# Filtros analógicos activos — TP Integrador (SRA 2026) — Grupo 3

Diseño, simulación e implementación de filtros analógicos activos.
**Tipos asignados:** pasabajos (LP) y rechazabanda (BS). **Aproximaciones:** Chebyshev I y elíptica (Cauer).
**Filtro a implementar (propuesta, sujeta a aprobación docente):** LP Chebyshev I de orden 3 (plantilla: $f_p = 1$ kHz, $A_p = 1$ dB),
sintetizado con margen de diseño ($A_{p,d} = 0{,}3$ dB, $f_{p,d} = 1060$ Hz): RC + seguidor y Sallen–Key de ganancia unitaria con un TL072CP.

| Componente | Valor |
|---|---|
| R0 | 20 kΩ + 560 Ω (serie) |
| C0 | 10 nF |
| R1 | 7,5 kΩ + 3,3 kΩ (serie) |
| R2 | 6,8 kΩ + 560 Ω (serie) |
| C1 | 47 nF |
| C2 | 4,7 nF |

## Estado

| Etapa | Contenido | Estado |
|---|---|---|
| Semana 1 | Cuatro diseños matemáticos, comparación y selección (`notebooks/01_...`) | ✅ Listo, pendiente de aprobación docente |
| Semana 2 | Síntesis simbólica (SymPy), componentes, elección del AO (`notebooks/02_...`) | ✅ Listo |
| Semana 3 | Sensibilidades, LTspice AC/transitorio (`notebooks/03_...`, `ltspice/`) | ✅ Listo (LTspice 24.1.10, modelo TI del TL072) |
| Semana 4 | Montaje y mediciones (`mediciones/`, `notebooks/04_Mediciones.ipynb`) | ✅ Bode y escalón medidos y procesados, con R medidas, capacitores K (±10 %) y alimentación ±11,97 V; ensayo de recorte no realizable (generador de 20 Vpp) |
| Semana 5 | Informe final (secciones 1 a 12, con conclusiones), defensa | ✅ Informe completo; falta la defensa |
| Nivel II | Monte Carlo en LTspice: 4 escenarios × 500 corridas, diagnóstico del generador (`notebooks/05_...`, `ltspice/filtro_TL072_mc_*.cir`, `filtro_mc_sondas*.cir`) | ✅ Listo |
| Nivel III | Ecualizador pasatodo de 2.º orden (TL081): diseño, SymPy, LTspice, montaje, sintonía en dos correcciones, medición final y onda cuadrada (`notebooks/06_...`, `mediciones/nivel3/`) | ✅ Completo: medido (×16,6 en la variación del retardo) y simulado en LTspice con el montaje final |

## Estructura

```
notebooks/      Notebooks ejecutables (Google Colab o Jupyter local)
figuras/        Figuras generadas por los notebooks (PDF vectorial para LaTeX + PNG)
resultados/     Tablas CSV y fragmentos .tex generados por los notebooks
informe/        Informe LaTeX (main.tex, secciones/, referencias.bib)
ltspice/        Esquemas .asc / netlists ejecutables
mediciones/     Datos de laboratorio (CSV) y fotos del montaje
docs/           Hojas de datos de componentes
tools/          Fuentes .py de los notebooks y utilidades de generación
```

Las figuras, tablas y valores numéricos del informe (`resultados/*.tex`, incluido `valores_nb02.tex` con macros LaTeX) **no se editan a mano**: se regeneran ejecutando los notebooks.

## Reproducción

### Opción A — Google Colab
1. Abrir el notebook desde GitHub en Colab (`Archivo → Abrir cuaderno → GitHub`).
2. Ejecutar todas las celdas. Las figuras y tablas se escriben en `figuras/` y `resultados/` (relativas al directorio de trabajo).

### Opción B — local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_Aproximaciones_y_Especificaciones.ipynb
```
Orden de ejecución: 01 → 02 → 03 → 04 → 05 → 06 (el 05 usa `resultados/componentes.json`, los resultados de LTspice en `ltspice/resultados/` y, para ubicar el prototipo, `valores_nb04.tex`).

El notebook 02 dibuja el esquema con `schemdraw` (se instala solo en Colab si falta). Los notebooks se generan a partir de `tools/src_nbXX.py` (celdas `# %%`); para regenerarlos:
```bash
python tools/py2nb.py tools/src_nb01.py notebooks/01_Aproximaciones_y_Especificaciones.ipynb
```

### Informe
```bash
cd informe
latexmk -pdf main.tex
```
En Overleaf: subir el repositorio completo (o la carpeta `informe/` junto con `figuras/` y `resultados/`) y fijar `informe/main.tex`
como documento principal. La bibliografía usa BibTeX con el estilo `babunsrt` (paquete `babelbib`).

## Versiones utilizadas

| Paquete | Versión |
|---|---|
| Python | 3.13 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| SymPy | 1.14.0 |
| Matplotlib | 3.11.2 |
| pandas | 3.0.5 |
| LTspice | (completar con la versión instalada) |

Cada notebook imprime en su primera celda las versiones con que se ejecutó.

## Convenciones

- Frecuencias especificadas en Hz; todo el cálculo en rad/s ($\omega = 2\pi f$).
- Atenuación $A(\omega) = -20\log_{10}|H(j\omega)|$ referida a $\max|H| = 1$ (normalización de SciPy).
- Retardo de grupo $\tau_g = -d\varphi/d\omega$ calculado analíticamente desde polos y ceros, en rad/s.
- Nunca se inventan datos experimentales: `mediciones/` solo contiene datos medidos.
