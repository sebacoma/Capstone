# Extraccion Escalable de Narrativas Coherentes mediante Modelos de Lenguaje Locales

Proyecto Capstone — Sebastian Concha Macias  
Profesor guia: Brian Keith Norambuena, UCN  
Repositorio original de Narrative Maps: [github.com/briankeithn/narrative-maps](https://github.com/briankeithn/narrative-maps)

---

## Descripcion del Proyecto

Este proyecto compara **LLMs locales** (Ollama) contra **metodos algoritmicos** y **modelos propietarios** (OpenAI) para la extraccion de narrativas coherentes basadas en eventos desde colecciones de documentos noticiosos chilenos.

Se implementan dos algoritmos de extraccion:
- **Narrative Maps** (Keith et al., 2021) — Programacion Lineal Entera (ILP) que selecciona un subgrafo optimo
- **Narrative Trails** (German et al., 2025) — Camino MaxiMin que maximiza la coherencia minima

Y tres tipos de scorers de coherencia:
- **Algoritmico**: Sentence-Transformers (all-MiniLM-L6-v2, similitud coseno)
- **LLM local**: Ollama (llama3.2, mistral)
- **Propietario**: OpenAI (gpt-4o-mini)

## Metodos

| Metodo | Scorer de Coherencia | Algoritmo de Extraccion |
|--------|---------------------|------------------------|
| `maps_algorithmic` | Sentence-Transformers | Narrative Maps — ILP |
| `maps_llm` | LLM via Ollama | Narrative Maps — ILP |
| `maps_proprietary` | OpenAI API | Narrative Maps — ILP |
| `trails_algorithmic` | Sentence-Transformers | Narrative Trails — MaxiMin |
| `trails_llm` | LLM via Ollama | Narrative Trails — MaxiMin |
| `trails_proprietary` | OpenAI API | Narrative Trails — MaxiMin |

## Estructura del Proyecto

```
narrative-extraction/
├── pipeline.py                  # CLI principal — ejecuta un experimento
├── data/
│   ├── convert.py               # Convierte FINAL_DATA.json → corpus.jsonl
│   ├── sample.jsonl             # Dataset curado: 15 docs (reforma previsional 2024)
│   └── corpus.jsonl             # Dataset completo: 931 docs (estallido social, generado)
├── baselines/
│   ├── result.py                # NarrativeResult dataclass (documents + edges)
│   ├── maps.py                  # Narrative Maps (ILP, Keith et al. 2021)
│   └── trails.py                # Narrative Trails (MaxiMin, German et al. 2025)
├── scoring/
│   ├── algorithmic.py           # Scorer Sentence-Transformers
│   └── precompute.py            # Cache de matriz de coherencia
├── llm/
│   ├── scorer.py                # Scorer LLM local (Ollama)
│   └── openai_scorer.py         # Scorer propietario (OpenAI)
├── evaluation/
│   └── judge.py                 # LLM-as-a-Judge (pointwise + pairwise)
├── metrics/
│   └── recorder.py              # Wall time, peak memory, edge coherence
├── experiments/
│   └── run_batch.py             # Batch runner para multiples experimentos
├── analysis/
│   ├── statistics.py            # t-tests, Cohen's d, tablas descriptivas
│   ├── plots.py                 # Graficos estadisticos + visualizacion de mapas narrativos
│   └── plots/                   # PNGs generados
├── results/                     # JSONs de resultados (auto-creado)
├── .env                         # OPENAI_API_KEY (no incluido en git)
├── INFORME_RESULTADOS.md        # Informe de resultados de los primeros experimentos
└── requirements.txt
```

## Setup

### 1. Python y dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

Dependencias principales: `sentence-transformers`, `networkx`, `matplotlib`, `pulp`, `ollama`, `openai`, `scipy`, `pandas`, `numpy`

### 2. Ollama (para metodos LLM locales)

Descargar desde https://ollama.com e instalar. Luego:

```bash
ollama pull llama3.2    # 2 GB — rapido, buen rendimiento
ollama pull mistral     # 4.4 GB — mejor coherencia, mas lento
```

**Nota sobre hardware:** Se descartaron phi3 y gemma2 porque sobrecalentaban el Mac. Se recomienda **no correr LLMs en paralelo**.

### 3. OpenAI API (para metodos propietarios)

Crear archivo `.env` en la raiz del proyecto:
```
OPENAI_API_KEY=sk-...
```

### 4. Preparar dataset

```bash
# Convertir FINAL_DATA.json (931 articulos) a formato JSONL
python data/convert.py
# Genera: data/corpus.jsonl
```

**Nota sobre corpus.jsonl:** 494 de 931 titulos estan truncados con "..." en el archivo fuente original (FINAL_DATA.json). Esto proviene del scraping original y no afecta el contenido completo de los articulos (campo `text`).

---

## Uso

### Ejecucion individual

```bash
# Algoritmico (no requiere Ollama ni OpenAI)
python pipeline.py --method maps_algorithmic --input data/sample.jsonl --n 6 --window 30 --skip-judge

# LLM local (requiere Ollama corriendo)
python pipeline.py --method maps_llm --model llama3.2 --input data/sample.jsonl --n 6 --window 30

# Propietario (requiere OPENAI_API_KEY en .env)
python pipeline.py --method maps_proprietary --openai-model gpt-4o-mini --input data/sample.jsonl --n 6 --window 30
```

### Opciones del pipeline

```
--input          Ruta al dataset JSONL (default: data/sample.jsonl)
--method         maps_algorithmic | maps_llm | maps_proprietary |
                 trails_algorithmic | trails_llm | trails_proprietary
--n              Largo de la narrativa en documentos (default: 6)
--model          Modelo Ollama para scorer LLM (default: llama3.2)
--openai-model   Modelo OpenAI para scorer propietario (default: gpt-4o-mini)
--judge-model    Modelo Ollama para evaluacion judge (default: mismo que --model)
--window         Ventana temporal en dias para filtrado de aristas (default: None)
--start          Indice del nodo de inicio en los docs ordenados por fecha (default: 0 = mas antiguo)
--end            Indice del nodo final en los docs ordenados por fecha (default: None = libre)
--skip-judge     Omitir evaluacion LLM-as-a-Judge
--output         Ruta del JSON de salida (default: results/run_{timestamp}.json)
```

**Nota sobre --start y --end:** Los indices se refieren a la posicion del documento en la lista ordenada por fecha. Por ejemplo, con 120 documentos, `--start 0` usa el mas antiguo y `--start 60` comienza desde la mitad del corpus. Util para probar distintos puntos de inicio y validar estadisticamente que los resultados no dependen de un inicio especifico.

### Batch de experimentos

```bash
# Vista previa (sin ejecutar)
python experiments/run_batch.py --dry-run

# Ejecutar todos los experimentos de maps con sample.jsonl
python experiments/run_batch.py \
    --input data/sample.jsonl \
    --methods maps_algorithmic maps_llm maps_proprietary \
    --models llama3.2 mistral \
    --sizes 6 12 \
    --reps-algorithmic 5 --reps-llm 5 --reps-proprietary 5 \
    --window 30 --judge-model llama3.2

# Ejecutar solo trails
python experiments/run_batch.py \
    --input data/sample.jsonl \
    --methods trails_algorithmic trails_llm trails_proprietary \
    --models llama3.2 mistral \
    --sizes 6 12 \
    --window 30

# El batch runner detecta resultados existentes y solo ejecuta los pendientes
```

### Analisis y graficos

```bash
# Analisis estadistico (tablas descriptivas, t-tests, Cohen's d)
python analysis/statistics.py --results-dir results

# Generar todos los graficos (incluye visualizacion de narrative maps)
python analysis/plots.py --results-dir results --output-dir analysis/plots
```

Graficos generados:
- `coherence_boxplot.png` — Box plot de coherencia por metodo
- `judge_boxplot.png` — Box plot de scores del judge
- `coherence_by_n.png` — Barras agrupadas: coherencia vs metodo vs n
- `quality_vs_time.png` — Scatter: calidad vs tiempo (Pareto)
- `effect_size_heatmap.png` — Heatmap de Cohen's d
- `narrative_map_*.png` — **Visualizacion de cada Narrative Map como grafo dirigido** (solo para resultados maps_*)

---

## Formato de Resultados

Cada experimento genera un JSON con esta estructura:

```json
{
  "method": "maps_llm",
  "model": "llama3.2",
  "n": 6,
  "window_days": 30,
  "narrative": [
    {"id": "doc_01", "date": "2024-01-15", "title": "..."}
  ],
  "metrics": {
    "wall_time_seconds": 57.17,
    "peak_memory_mb": 2.35,
    "avg_edge_coherence": 0.80
  },
  "judge": {
    "coherence_score": 8,
    "justification": "...",
    "evaluation_type": "pointwise"
  },
  "graph": {
    "nodes": [{"id": "doc_01", "date": "2024-01-15", "title": "..."}],
    "edges": [{"source": "doc_01", "target": "doc_05", "coherence_score": 0.85}]
  }
}
```

El campo `"graph"` solo esta presente en metodos `maps_*`. Contiene:
- **nodes**: Los n documentos seleccionados por el ILP
- **edges**: TODAS las aristas entre nodos seleccionados con su score de coherencia (no solo el path, sino las cross-edges que forman el mapa)

Los metodos `trails_*` no generan campo `"graph"` ya que producen un camino lineal, no un mapa.

---

## Evaluacion

El pipeline usa el protocolo **LLM-as-a-Judge** (Keith, 2025):
- **Pointwise**: Puntua cada narrativa en escala 0-10 de coherencia
- **Pairwise**: Comparacion directa A/B entre dos narrativas

---

## Lo Que Se Ha Hecho

### Experimentos completados (sample.jsonl, 15 docs)

Se ejecutaron **64 experimentos** iniciales (maps + trails) y luego se re-ejecutaron **40 experimentos maps** con la nueva estructura de grafo:

| Configuracion | Detalle |
|---|---|
| **Dataset** | sample.jsonl (15 docs, reforma previsional Chile 2024) |
| **Modelos locales** | llama3.2 (2GB), mistral (4.4GB) |
| **Modelo propietario** | gpt-4o-mini |
| **Algoritmico** | sentence-transformers (all-MiniLM-L6-v2) |
| **Tamanos** | n=6, n=12 (n=18 descartado: sample solo tiene 15 docs) |
| **Repeticiones** | 5 por combinacion |
| **Ventana temporal** | 30 dias |

### Resultados clave

| Hallazgo | Detalle |
|---|---|
| LLMs > Algoritmicos | p < 0.001, diferencia significativa |
| Mejor scorer | gpt-4o-mini (coherencia promedio: 0.874) |
| Mejor LLM local | mistral (0.851) — 97% del rendimiento de gpt-4o-mini |
| Maps > Trails | Narrative Maps supera a Narrative Trails consistentemente |
| Judge scores | 8-9/10 para todos, sin diferencia estadistica significativa |

### Modificaciones recientes al codigo

1. **Narrative Maps ahora retorna estructura de grafo** — `NarrativeResult` dataclass en `baselines/result.py` con `documents` (path) y `edges` (todas las aristas entre nodos seleccionados)
2. **JSON de salida incluye campo `"graph"`** para metodos maps_* con nodos y aristas con scores de coherencia
3. **Visualizacion de mapas narrativos** — Nueva funcion `plot_narrative_map()` en `analysis/plots.py` que dibuja el grafo dirigido con path principal (azul) y cross-edges coloreadas por coherencia

---

## Lo Que Falta Por Hacer

### 1. Experimentos con corpus grande (~120 eventos) y distintos puntos de inicio

Los experimentos actuales usan solo 15 documentos curados (sample.jsonl). Para validez estadistica es necesario:

- Usar el corpus completo (corpus.jsonl, 931 docs) con **distintos puntos de inicio**
- El parametro `--start-indices` permite definir desde que documento (ordenado por fecha) comienza la narrativa
- Esto demuestra que los resultados no dependen de un punto de inicio especifico
- Tamanos sugeridos: n=6, n=12, posiblemente n=18

```bash
# Experimentos con distintos puntos de inicio sobre el corpus completo
python experiments/run_batch.py \
    --input data/corpus.jsonl \
    --methods maps_algorithmic maps_llm maps_proprietary \
    --models llama3.2 mistral \
    --sizes 6 12 18 \
    --start-indices 0 100 200 300 400 \
    --reps-llm 3 --reps-proprietary 3 \
    --window 30

# Vista previa sin ejecutar
python experiments/run_batch.py \
    --input data/corpus.jsonl \
    --methods maps_llm \
    --models llama3.2 \
    --sizes 6 12 \
    --start-indices 0 100 200 300 400 \
    --reps-llm 3 \
    --dry-run

# Los resultados se guardan en:
#   results/{method}/{model}/n{size}/start{idx}/rep{r}.json
# Para start=0 (default) la ruta es sin subcarpeta start (backward compatible).

# Tambien se puede ejecutar un solo experimento con inicio especifico:
python pipeline.py --method maps_llm --model llama3.2 --input data/corpus.jsonl \
    --n 12 --start 200 --window 30 --skip-judge
```

**Importante:** La matriz de coherencia crece cuadraticamente con el numero de documentos (931^2 = 866,761 pares). Se necesita una maquina con GPU o recursos suficientes. No es viable en un Mac sin GPU dedicada, especialmente con mistral.

### 2. Re-generar analisis estadistico

Despues de los nuevos experimentos:

```bash
python analysis/statistics.py --results-dir results
python analysis/plots.py --results-dir results --output-dir analysis/plots
```

Esto actualiza:
- `analysis/summary_table.csv` — Estadisticas descriptivas
- `analysis/significance_coherence.csv` — Tests de significancia
- `analysis/plots/` — Todos los graficos + visualizaciones de narrative maps

### 3. Redactar informe final de Capstone

- **Deadline: 16 de junio de 2026**
- Base: `INFORME_RESULTADOS.md` (ya generado con resultados preliminares)
- Incorporar resultados del corpus grande y las visualizaciones de mapas narrativos

---

## Tiempos de Ejecucion de Referencia (Mac M-series, sample.jsonl, 15 docs)

| Metodo | Modelo | n=6 | n=12 |
|---|---|---|---|
| maps_algorithmic | sentence-transformers | ~9s | ~5s |
| maps_llm | llama3.2 | ~56s | ~56s |
| maps_llm | mistral | ~131s | ~213s |
| maps_proprietary | gpt-4o-mini | ~115s | ~116s |

Con 120 documentos los tiempos seran **significativamente mayores** debido al crecimiento cuadratico de la matriz de coherencia.

---

## Referencias

- Keith, B. F., & Mitra, T. (2020). *Narrative Maps: An Algorithmic Approach to Represent and Extract Information Narratives*. ACM CSCW. [GitHub](https://github.com/briankeithn/narrative-maps)
- Keith, B. (2025). *Narrative Maps Visualization Tool (NMVT)*. SoftwareX, 32, 102377.
- German, F., Keith, B., & North, C. (2025). *Narrative Trails*. CEUR-WS Vol-3964. [GitHub](https://github.com/faustogerman/narrative-trails)
- Keith, B. (2025). *LLM-as-a-judge approaches as proxies for mathematical coherence in narrative extraction*. Electronics, 14(13), 2735.
- Zheng, L. et al. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.
