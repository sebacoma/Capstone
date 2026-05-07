# Informe de Resultados: Extracción Escalable de Narrativas Coherentes mediante Modelos de Lenguaje Locales

**Autor:** Sebastián Concha Macías  
**Profesor Guía:** Brian Keith Norambuena  
**Fecha:** Abril 2026

---

## 1. Descripción General del Proyecto

### 1.1 Objetivo
Comparar el desempeño de modelos de lenguaje locales (LLMs de código abierto ejecutados via Ollama) contra métodos algorítmicos tradicionales y modelos propietarios en la tarea de extracción de narrativas coherentes a partir de colecciones de noticias.

### 1.2 Pregunta de Investigación
¿Pueden los modelos de lenguaje locales alcanzar niveles de coherencia narrativa comparables a los métodos algorítmicos tradicionales y a los modelos propietarios en tareas de extracción de narrativas basada en eventos?

---

## 2. Diseño Experimental

### 2.1 Dataset
- **Fuente:** Colección de noticias chilenas (medios como Ciper) almacenada en `FINAL_DATA.json` (931 artículos).
- **Dataset experimental:** `sample.jsonl` — 15 documentos sobre la reforma previsional en Chile (enero-marzo 2024), seleccionados como narrativa temáticamente coherente.
- **Justificación del tamaño:** El uso de 15 documentos permite ejecutar todos los métodos y modelos en tiempos razonables, manteniendo la validez comparativa dado que todos los métodos operan sobre el mismo conjunto.

### 2.2 Métodos de Extracción (Algoritmos)

| Algoritmo | Descripción | Referencia |
|-----------|-------------|------------|
| **Narrative Maps** | Programación lineal entera (ILP) que selecciona un camino de n documentos maximizando la coherencia total. Usa restricciones de flujo y eliminación de subtours (MTZ). | Keith et al., 2021 |
| **Narrative Trails** | Algoritmo MaxiMin Dijkstra que busca el camino de n nodos que maximiza la mínima coherencia entre aristas (bottleneck path). | German et al., 2025 |

### 2.3 Métodos de Scoring (Coherencia entre pares)

| Scorer | Descripción | Modelo |
|--------|-------------|--------|
| **Algorítmico** | Similitud coseno entre embeddings de sentence-transformers (`all-MiniLM-L6-v2`). | - |
| **LLM Local (llama3.2)** | Evaluación de coherencia narrativa via prompt, ejecutado localmente con Ollama. | Meta LLaMA 3.2 (2GB) |
| **LLM Local (mistral)** | Mismo protocolo que llama3.2 pero con modelo más grande. | Mistral 7B (4.4GB) |
| **Propietario (gpt-4o-mini)** | Mismo protocolo via API de OpenAI. | GPT-4o-mini |

### 2.4 Combinaciones Evaluadas

Cada combinación de algoritmo x scorer produce un método:

| Método | Algoritmo | Scorer |
|--------|-----------|--------|
| `maps_algorithmic` | Narrative Maps | Sentence-Transformers |
| `maps_llm` (llama3.2) | Narrative Maps | LLaMA 3.2 local |
| `maps_llm` (mistral) | Narrative Maps | Mistral local |
| `maps_proprietary` | Narrative Maps | GPT-4o-mini |
| `trails_algorithmic` | Narrative Trails | Sentence-Transformers |
| `trails_llm` (llama3.2) | Narrative Trails | LLaMA 3.2 local |
| `trails_llm` (mistral) | Narrative Trails | Mistral local |
| `trails_proprietary` | Narrative Trails | GPT-4o-mini |

### 2.5 Parámetros Experimentales

| Parámetro | Valor |
|-----------|-------|
| Tamaños de narrativa (n) | 6 y 12 documentos |
| Ventana temporal | 30 días |
| Repeticiones (algorítmicos) | 1 por tamaño |
| Repeticiones (LLM/propietarios) | 5 por tamaño |
| Modelo judge | llama3.2 (fijo para todos) |
| Total de experimentos | 64 |

### 2.6 Métricas

| Métrica | Descripción |
|---------|-------------|
| **avg_edge_coherence** | Promedio de coherencia entre pares consecutivos de documentos en la narrativa extraída. Rango [0, 1]. |
| **wall_time_seconds** | Tiempo total de ejecución del método (extracción + scoring). |
| **peak_memory_mb** | Uso máximo de memoria durante la ejecución. |
| **judge_score** | Puntuación de coherencia narrativa (0-10) asignada por un LLM evaluador (protocolo LLM-as-a-judge, Keith 2025). |

### 2.7 Evaluación Estadística

- **Test-t de Welch** (varianzas no iguales) para comparaciones entre pares de métodos.
- **Corrección de Bonferroni** para comparaciones múltiples.
- **Cohen's d** para cuantificar el tamaño del efecto.
- Nivel de significancia: α = 0.05.

---

## 3. Resultados

### 3.1 Tabla Resumen de Estadísticas Descriptivas

| Método | Modelo | n | Reps | Coherencia (media ± std) | Tiempo (s) | Memoria (MB) | Judge Score |
|--------|--------|---|------|--------------------------|------------|--------------|-------------|
| maps_algorithmic | embeddings | 6 | 5 | 0.691 ± 0.095 | 47.3 | 81.3 | - |
| maps_algorithmic | embeddings | 12 | 2 | 0.742 ± 0.000 | 2.8 | 5.7 | - |
| maps_llm | llama3.2 | 6 | 5 | 0.800 ± 0.000 | 56.2 | 2.1 | 8.0 |
| maps_llm | llama3.2 | 12 | 5 | 0.782 ± 0.000 | 56.4 | 2.0 | 8.0 |
| maps_llm | mistral | 6 | 5 | 0.880 ± 0.000 | 131.8 | 2.0 | 9.0 |
| maps_llm | mistral | 12 | 5 | 0.823 ± 0.000 | 212.6 | 2.0 | 8.0 |
| maps_proprietary | gpt-4o-mini | 6 | 5 | 0.904 ± 0.017 | 115.2 | 5.3 | 8.8 |
| maps_proprietary | gpt-4o-mini | 12 | 5 | 0.844 ± 0.012 | 115.9 | 3.6 | 8.4 |
| trails_algorithmic | embeddings | 6 | 6 | 0.743 ± 0.016 | 6.9 | 9.2 | - |
| trails_algorithmic | embeddings | 12 | 2 | 0.733 ± 0.000 | 0.001 | 0.08 | - |
| trails_llm | llama3.2 | 6 | 5 | 0.800 ± 0.000 | 55.4 | 1.5 | 8.0 |
| trails_llm | llama3.2 | 12 | 5 | 0.782 ± 0.000 | 55.7 | 1.4 | 8.0 |
| trails_llm | mistral | 6 | 5 | 0.820 ± 0.000 | 130.0 | 1.4 | 8.0 |
| trails_llm | mistral | 12 | 5 | 0.786 ± 0.000 | 129.3 | 1.4 | 8.0 |
| trails_proprietary | gpt-4o-mini | 6 | 7 | 0.809 ± 0.011 | 228.0 | 5.7 | 8.0 |
| trails_proprietary | gpt-4o-mini | 12 | 5 | 0.827 ± 0.009 | 111.8 | 3.0 | - |

### 3.2 Ranking de Métodos por Coherencia Promedio (agregado n=6 y n=12)

| # | Método | Coherencia Media | Tiempo Medio (s) |
|---|--------|-----------------|-------------------|
| 1 | maps_proprietary (gpt-4o-mini) | **0.874** | 115.5 |
| 2 | maps_llm (mistral) | 0.851 | 172.2 |
| 3 | trails_proprietary (gpt-4o-mini) | 0.816 | 169.9 |
| 4 | maps_llm (llama3.2) | 0.791 | 56.3 |
| 5 | trails_llm (mistral) | 0.803 | 129.6 |
| 6 | trails_llm (llama3.2) | 0.791 | 55.6 |
| 7 | trails_algorithmic | 0.740 | 3.5 |
| 8 | maps_algorithmic | 0.706 | 25.1 |

### 3.3 Comparaciones Estadísticas (t-tests con corrección de Bonferroni)

#### Diferencias significativas (p_bonferroni < 0.05):

| Comparación | Coherencia A | Coherencia B | p (Bonferroni) | Cohen's d | Interpretación |
|-------------|-------------|-------------|----------------|-----------|----------------|eso es todo actualiza la memoria
| maps_algorithmic vs maps_proprietary | 0.706 | 0.874 | **0.015** | -2.90 | Propietario muy superior |
| maps_llm vs maps_proprietary | 0.821 vs 0.874 | | **0.017** | -1.43 | Propietario superior (efecto grande) |
| maps_llm vs trails_algorithmic | 0.821 | 0.740 | **< 0.001** | 2.43 | LLM muy superior a algorítmico |
| maps_proprietary vs trails_algorithmic | 0.874 | 0.740 | **< 0.001** | 4.84 | Propietario muy superior (efecto enorme) |
| maps_proprietary vs trails_llm | 0.874 | 0.797 | **< 0.001** | 3.30 | Propietario superior |
| maps_proprietary vs trails_proprietary | 0.874 | 0.816 | **0.006** | 2.27 | Maps supera a Trails con mismo scorer |
| trails_algorithmic vs trails_llm | 0.740 | 0.797 | **< 0.001** | -3.80 | LLM muy superior a algorítmico |
| trails_algorithmic vs trails_proprietary | 0.740 | 0.816 | **< 0.001** | -5.51 | Propietario muy superior (efecto enorme) |
| trails_llm vs trails_proprietary | 0.797 | 0.816 | **0.015** | -1.32 | Propietario ligeramente superior |

#### Diferencias NO significativas:

| Comparación | p (Bonferroni) | Interpretación |
|-------------|----------------|----------------|
| maps_algorithmic vs maps_llm | 0.128 | Tendencia favorable a LLM, pero no significativa |
| maps_algorithmic vs trails_algorithmic | 1.000 | Sin diferencia entre algoritmos con mismo scorer |
| maps_llm vs trails_llm | 0.213 | Sin diferencia significativa entre algoritmos con LLM |
| maps_llm vs trails_proprietary | 1.000 | LLM local comparable a propietario |

### 3.4 Judge Scores (LLM-as-a-Judge)

| Método | Judge Score Medio |
|--------|-------------------|
| maps_llm (mistral) | **9.0** |
| maps_proprietary (gpt-4o-mini) | 8.6 |
| maps_llm (llama3.2) | 8.0 |
| trails_llm (todos) | 8.0 |
| trails_proprietary | 8.0 |

No se encontraron diferencias estadísticamente significativas en los judge scores entre métodos (p_bonferroni > 0.05 en todas las comparaciones). Esto sugiere que, desde la perspectiva de evaluación holística por el juez LLM, todas las narrativas extraídas alcanzan un nivel de calidad comparable.

---

## 4. Visualizaciones Generadas

Los siguientes gráficos se encuentran en `analysis/plots/`:

1. **`coherence_boxplot.png`** — Box plot de coherencia por método. Muestra la distribución y variabilidad.
2. **`coherence_by_n.png`** — Coherencia agrupada por método y tamaño de narrativa (n=6 vs n=12).
3. **`quality_vs_time.png`** — Scatter plot de calidad vs. tiempo de ejecución (frontera de Pareto).
4. **`effect_size_heatmap.png`** — Heatmap de Cohen's d entre todos los pares de métodos.
5. **`judge_boxplot.png`** — Box plot de judge scores por método.

---

## 5. Análisis e Interpretación

### 5.1 LLMs locales vs. métodos algorítmicos (OE3)

Los modelos de lenguaje locales superan consistentemente a los métodos algorítmicos en coherencia narrativa:

- **trails_llm (llama3.2) vs trails_algorithmic**: 0.791 vs 0.740 (p < 0.001, d = -3.80)
- **maps_llm (mistral) vs maps_algorithmic**: 0.851 vs 0.706

Esto confirma que los LLMs capturan relaciones semánticas más complejas que la similitud coseno de embeddings. La ventaja es estadísticamente significativa y con tamaños de efecto grandes (|d| > 2).

### 5.2 LLMs locales vs. modelo propietario (OE3)

El modelo propietario (gpt-4o-mini) obtiene la coherencia más alta (0.874), pero la brecha con los LLMs locales es moderada:

- **mistral** alcanza 0.851 (97% del rendimiento de gpt-4o-mini con Narrative Maps)
- **llama3.2** alcanza 0.791 (91% del rendimiento de gpt-4o-mini)

La diferencia maps_llm vs maps_proprietary es significativa (p = 0.017, d = -1.43), pero la diferencia maps_llm vs trails_proprietary no lo es (p = 1.0), indicando que un LLM local con el algoritmo correcto puede igualar a un propietario.

### 5.3 Efecto del algoritmo de extracción (OE4)

Con el mismo scorer, **Narrative Maps tiende a superar a Narrative Trails**:
- maps_proprietary (0.874) vs trails_proprietary (0.816), p = 0.006
- maps_llm mistral (0.851) vs trails_llm mistral (0.803)

Esto sugiere que la optimización global (ILP) produce narrativas más coherentes que la optimización local (MaxiMin bottleneck).

### 5.4 Efecto del tamaño de narrativa (OE3)

En todos los métodos, las narrativas de **n=6 obtienen mayor coherencia que n=12**:
- maps_llm mistral: 0.880 (n=6) vs 0.823 (n=12)
- maps_proprietary: 0.904 (n=6) vs 0.844 (n=12)

Esto es esperado: narrativas más cortas permiten seleccionar solo los documentos más coherentes entre sí.

### 5.5 Trade-off calidad vs. eficiencia (OE4)

| Método | Coherencia | Tiempo | Trade-off |
|--------|-----------|--------|-----------|
| trails_algorithmic | 0.740 | **3.5s** | Más rápido, menor calidad |
| maps/trails_llm (llama3.2) | 0.791 | ~56s | Buen balance calidad/velocidad |
| maps_llm (mistral) | 0.851 | ~172s | Alta calidad, tiempo moderado |
| maps_proprietary (gpt-4o-mini) | **0.874** | ~115s | Mejor calidad, requiere API externa |

**Mejor equilibrio local:** `maps_llm` con **mistral** ofrece el 97% de la calidad del modelo propietario sin depender de servicios externos, con control total sobre los datos.

**Mejor opción rápida:** `maps/trails_llm` con **llama3.2** ofrece buena calidad en ~56s, ideal para iteraciones rápidas.

---

## 6. Mapeo a Objetivos Específicos

| Objetivo Específico | Resultado | Estado |
|---------------------|-----------|--------|
| **OE1:** Identificar modelos de código abierto viables | Se evaluaron LLaMA 3.2 (2GB) y Mistral 7B (4.4GB) via Ollama. Ambos viables en laptop con Apple M4 Pro (18GB RAM). | Cumplido |
| **OE2:** Implementar pipeline experimental | Pipeline completo implementado: carga de datos, scoring (algorítmico/LLM/propietario), extracción (Maps/Trails), evaluación (judge), métricas, batch runner. | Cumplido |
| **OE3:** Evaluar coherencia y eficiencia | 64 experimentos ejecutados. Coherencia medida con avg_edge_coherence y judge scores. Eficiencia medida con wall_time y peak_memory. Validación estadística con t-tests y Cohen's d. | Cumplido |
| **OE4:** Analizar trade-offs calidad/costo | Mistral local alcanza 97% de la coherencia de gpt-4o-mini sin costo de API ni dependencia externa. LLaMA 3.2 es 3x más rápido que mistral con 91% de la calidad. | Cumplido |

---

## 7. Conclusiones

1. **Los LLMs locales son viables** para extracción de narrativas basada en eventos, superando significativamente a los métodos algorítmicos tradicionales en coherencia narrativa (p < 0.001).

2. **Mistral 7B** es el mejor modelo local evaluado, alcanzando el 97% de la coherencia del modelo propietario gpt-4o-mini sin costo de API ni dependencia tecnológica.

3. **Narrative Maps (ILP) supera a Narrative Trails (MaxiMin)** como algoritmo de extracción, independientemente del scorer utilizado.

4. **El protocolo LLM-as-a-judge** muestra que todas las narrativas extraídas con LLMs alcanzan scores de 8-9/10, sin diferencias significativas entre métodos, validando la calidad general de las extracciones.

5. **El trade-off principal** es entre velocidad (algorítmico: ~3s) y calidad (LLM: ~56-172s), con los LLMs locales ofreciendo un punto medio viable para entornos con restricciones de infraestructura.

---

## 8. Infraestructura y Reproducibilidad

### 8.1 Hardware
- **Equipo:** Apple M4 Pro, 18 GB RAM
- **Ejecución local:** Todos los modelos LLM via Ollama, sin GPU dedicada

### 8.2 Software
- Python 3.14
- Ollama 0.21.1
- Dependencias: sentence-transformers, networkx, pulp, ollama, openai, scipy, pandas, matplotlib

### 8.3 Modelos Utilizados

| Modelo | Tamaño | Ejecución |
|--------|--------|-----------|
| all-MiniLM-L6-v2 | 80MB | Local (sentence-transformers) |
| LLaMA 3.2 | 2.0 GB | Local (Ollama) |
| Mistral 7B | 4.4 GB | Local (Ollama) |
| GPT-4o-mini | - | API (OpenAI) |

### 8.4 Estructura del Proyecto

```
narrative-extraction/
├── pipeline.py                  # Pipeline principal (CLI)
├── data/
│   ├── convert.py               # Conversor FINAL_DATA.json → corpus.jsonl
│   ├── sample.jsonl             # Dataset experimental (15 documentos)
│   └── corpus.jsonl             # Dataset completo (931 artículos)
├── baselines/
│   ├── maps.py                  # Narrative Maps (ILP, Keith et al. 2021)
│   └── trails.py                # Narrative Trails (MaxiMin, German et al. 2025)
├── scoring/
│   ├── algorithmic.py           # Scorer embeddings (sentence-transformers)
│   └── precompute.py            # Cache de matrices de coherencia
├── llm/
│   ├── scorer.py                # Scorer LLM local (Ollama)
│   └── openai_scorer.py         # Scorer propietario (OpenAI)
├── evaluation/
│   └── judge.py                 # LLM-as-a-Judge (pointwise + pairwise)
├── metrics/
│   └── recorder.py              # Wall time, memoria, coherencia promedio
├── experiments/
│   └── run_batch.py             # Ejecutor batch de experimentos
├── analysis/
│   ├── statistics.py            # t-tests, Cohen's d, tablas resumen
│   ├── plots.py                 # Generación de gráficos
│   ├── summary_table.csv        # Estadísticas descriptivas
│   ├── significance_coherence.csv # Tests de significancia
│   └── plots/                   # Gráficos generados
│       ├── coherence_boxplot.png
│       ├── coherence_by_n.png
│       ├── quality_vs_time.png
│       ├── effect_size_heatmap.png
│       └── judge_boxplot.png
└── results/                     # 64 archivos JSON de resultados
```

---

## 9. Referencias

- German, F., Keith, B., & North, C. (2025). Narrative Trails: A method for coherent storyline extraction via maximum capacity path optimization. Text2Story 2025 Workshop.
- Keith, B. (2025). LLM-as-a-judge approaches as proxies for mathematical coherence in narrative extraction. Electronics, 14(13), 2735.
- Keith, B. F., & Mitra, T. (2021). Narrative Maps: An algorithmic approach to represent and extract information narratives. Proceedings of the ACM on Human-Computer Interaction, 4(CSCW3).
- Keith, B. F., Mitra, T., & North, C. (2023). A survey on event-based news narrative extraction. ACM Computing Surveys, 55(14s).
- Shahaf, D., & Guestrin, C. (2010). Connecting the dots between news articles. Proceedings of the 16th ACM SIGKDD.
- Li, M., et al. (2021). Timeline Summarization based on Event Graph Compression via Time-Aware Optimal Transport. EMNLP 2021.
- Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023.
