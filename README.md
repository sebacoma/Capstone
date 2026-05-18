# Comparacion de Metodos de Extraccion de Narrativas desde Corpus de Noticias

**Autor:** Sebastian Concha Macias
**Profesor guia:** Brian Keith Norambuena
**Universidad:** Universidad Catolica del Norte
**Fecha:** Mayo 2026

---

## Descripcion

Proyecto Capstone que compara metodos algoritmicos y basados en LLMs para la extraccion automatica de narrativas coherentes desde un corpus de 931 noticias chilenas.

Se evaluan seis metodos:

| Metodo | Tipo | Modelo |
|---|---|---|
| Narrative Maps | Algoritmico (LP Relaxation) | Sentence-Transformers |
| Narrative Trails | Algoritmico (MaxiMin Dijkstra) | Sentence-Transformers |
| LLM Extractor (LLaMA 3.2) | LLM local | llama3.2 (3B) via Ollama |
| LLM Extractor (Mistral) | LLM local | mistral (7B) via Ollama |
| LLM Extractor (Gemma 3) | LLM local | gemma3:4b (4B) via Ollama |
| LLM Extractor (GPT-4o-mini) | LLM propietario | gpt-4o-mini via OpenAI API |

## Estructura del Proyecto

```
CAPSTONE/
├── README.md                        # Este archivo
├── INFORME_FINAL.md                 # Informe final del Capstone
├── FINAL_DATA.json                  # Corpus original (931 articulos)
├── run_experiments_local.py         # Script principal de experimentos (v2)
├── analyze_results.py               # Analisis estadistico y generacion de tablas/plots
├── .env                             # OPENAI_API_KEY (no incluido en git)
│
├── narrative-extraction/            # Modulos del pipeline
│   ├── pipeline.py                  # CLI para ejecucion individual
│   ├── data/
│   │   ├── convert.py               # Convierte FINAL_DATA.json → corpus.jsonl
│   │   ├── corpus.jsonl             # Dataset completo (931 docs)
│   │   └── sample.jsonl             # Dataset curado (15 docs, pruebas rapidas)
│   ├── baselines/
│   │   ├── maps.py                  # Narrative Maps (ILP, Keith et al. 2021)
│   │   └── trails.py                # Narrative Trails (MaxiMin, German et al. 2025)
│   ├── scoring/
│   │   ├── algorithmic.py           # Scorer Sentence-Transformers
│   │   └── precompute.py            # Cache de matriz de coherencia
│   ├── llm/
│   │   ├── scorer.py                # Scorer LLM local (Ollama)
│   │   └── openai_scorer.py         # Scorer propietario (OpenAI)
│   ├── evaluation/
│   │   └── judge.py                 # LLM-as-a-Judge (pointwise + pairwise)
│   ├── metrics/
│   │   └── recorder.py              # Wall time, peak memory, edge coherence
│   ├── experiments/
│   │   └── run_batch.py             # Batch runner (v1, para maps/trails con scorers)
│   ├── analysis/
│   │   ├── statistics.py            # t-tests, Cohen's d
│   │   ├── plots.py                 # Graficos
│   │   └── plots/                   # PNGs generados (v1)
│   └── results/                     # Resultados v1 (maps/trails con scorers)
│
└── results_local/                   # Resultados v2 (definitivos)
    ├── results_v2/                  # JSONs de cada experimento
    │   ├── maps_algorithmic/
    │   ├── trails_algorithmic/
    │   ├── llm_extractor_llama32/
    │   ├── llm_extractor_mistral/
    │   ├── llm_extractor_gemma3/
    │   └── llm_extractor_gpt4omini/
    ├── llm_extractor_cache/         # Cache de respuestas LLM
    └── analysis/
        ├── plots/                   # Graficos finales (PNG)
        └── tables/                  # Tablas estadisticas (CSV)
```

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com) (para LLMs locales)
- Clave de API de OpenAI (para GPT-4o-mini)

## Instalacion

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd CAPSTONE

# Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r narrative-extraction/requirements.txt
```

### Modelos Ollama

```bash
ollama pull llama3.2      # 3B params, ~2 GB
ollama pull mistral        # 7B params, ~4.4 GB
ollama pull gemma3:4b      # 4B params, ~3 GB
```

### Configurar API de OpenAI

Crear archivo `.env` en la raiz:
```
OPENAI_API_KEY=sk-...
```

### Preparar corpus

Si `narrative-extraction/data/corpus.jsonl` no existe:
```bash
cd narrative-extraction
python data/convert.py
cd ..
```

## Uso

### Ejecutar experimentos (v2 - definitivo)

El script `run_experiments_local.py` ejecuta todos los experimentos del benchmark:

```bash
# Ejecutar todos los metodos pendientes
python run_experiments_local.py

# Solo metodos LLM
python run_experiments_local.py --only-llm

# Solo metodos algoritmicos
python run_experiments_local.py --only-algorithmic
```

Configuracion experimental (definida en el script):
- **Tamanos de narrativa:** n = 6, 12, 18
- **Puntos de inicio:** start_index = 0, 50, 100, 150
- **Repeticiones:** 5 por configuracion (con semillas fijas para reproducibilidad)
- **Muestra:** 200 documentos del corpus (muestreo sistematico)

Los resultados se guardan en `results_local/results_v2/{metodo}/n{size}/[start{idx}/]rep{r}.json`.

El script detecta resultados existentes y solo ejecuta los experimentos pendientes.

### Analisis de resultados

```bash
# Genera tablas y plots en results_local/analysis/
python analyze_results.py

# Directorio de resultados custom
python analyze_results.py --results-dir results_local/results_v2
```

Genera:
- `results_local/analysis/tables/descriptive_stats.csv` — Estadisticas descriptivas
- `results_local/analysis/tables/pairwise_welch_t.csv` — Tests de Welch
- `results_local/analysis/plots/*.png` — Graficos del benchmark

### Ejecucion individual (pipeline v1)

Para ejecutar un solo experimento con el pipeline original:

```bash
cd narrative-extraction

# Algoritmico
python pipeline.py --method maps_algorithmic --input data/corpus.jsonl --n 6 --window 30

# LLM local
python pipeline.py --method maps_llm --model llama3.2 --input data/corpus.jsonl --n 6 --window 30

# Propietario
python pipeline.py --method maps_proprietary --openai-model gpt-4o-mini --input data/corpus.jsonl --n 6 --window 30
```

## Formato de Resultados

Cada experimento genera un JSON:

```json
{
  "method": "llm_extractor_gpt4omini",
  "n": 12,
  "start_index": 50,
  "seed": 2,
  "narrative": [
    {"id": 42, "date": "2019-11-15", "title": "..."}
  ],
  "metrics": {
    "wall_time_seconds": 15.3,
    "peak_memory_mb": 1.2,
    "avg_edge_coherence": 0.82
  }
}
```

## Referencias

- Keith, B. F., & Mitra, T. (2020). *Narrative Maps: An Algorithmic Approach to Represent and Extract Information Narratives*. ACM CSCW. [GitHub](https://github.com/briankeithn/narrative-maps)
- German, F., Keith, B., & North, C. (2025). *Narrative Trails*. CEUR-WS Vol-3964. [GitHub](https://github.com/faustogerman/narrative-trails)
- Keith, B. (2025). *LLM-as-a-judge approaches as proxies for mathematical coherence in narrative extraction*. Electronics, 14(13), 2735.
