# Informe Final: Comparación de Métodos de Extracción de Narrativas desde Corpus de Noticias

**Autor:** Sebastián Concha  
**Profesor guía:** Brian Keith  
**Universidad:** Universidad Católica del Norte  
**Fecha:** Mayo 2026

---

## 1. Introducción

Este trabajo compara métodos algorítmicos y basados en LLMs para la extracción automática de narrativas coherentes desde un corpus de noticias chilenas. Se evalúan dos enfoques algorítmicos clásicos (Narrative Maps y Narrative Trails) contra cuatro extractores basados en modelos de lenguaje: tres locales (LLaMA 3.2, Mistral, Gemma 3) y uno propietario (GPT-4o-mini).

El objetivo es determinar si los LLMs pueden igualar o superar a los métodos algorítmicos en calidad narrativa, y bajo qué condiciones de costo y eficiencia.

## 2. Metodología

### 2.1 Corpus

- **Fuente:** 931 artículos de noticias chilenas en formato JSONL
- **Muestra:** 200 documentos seleccionados mediante muestreo sistemático (equiespaciado temporal)
- **Período:** Cobertura temporal completa del corpus

### 2.2 Métodos Evaluados

| Método | Tipo | Modelo | Infraestructura |
|---|---|---|---|
| Narrative Maps | Algorítmico (LP Relaxation) | — | Local (CPU) |
| Narrative Trails | Algorítmico (MaxiMin Dijkstra) | — | Local (CPU) |
| LLM Extractor (LLaMA 3.2) | LLM local | llama3.2 (3B) | Ollama |
| LLM Extractor (Mistral) | LLM local | mistral (7B) | Ollama |
| LLM Extractor (Gemma 3) | LLM local | gemma3:4b (4B) | Ollama |
| LLM Extractor (GPT-4o-mini) | LLM propietario | gpt-4o-mini | OpenAI API |

**Nota:** Phi-3 fue excluido del benchmark porque no logró generar respuestas parseables en el formato requerido (JSON estructurado) tras múltiples intentos con distintas temperaturas (0.0, 0.3, 0.5), lo que evidencia una limitación del modelo para tareas de extracción narrativa estructurada en español.

### 2.3 Diseño Experimental

- **Tamaños de narrativa:** n = 6, 12, 18 documentos
- **Puntos de inicio:** start_index = 0, 50, 100, 150
- **Repeticiones:** 1 (algorítmicos), 5 (LLM extractors, con seeds 1-5)
- **Ventana temporal:** 30 días entre documentos consecutivos
- **Total planificado:** 264 experimentos (6 métodos × configuraciones)
- **Total ejecutado:** 194 experimentos exitosos

### 2.4 Métricas

1. **Judge Score (0-10):** Evaluación de coherencia narrativa mediante LLM-as-a-Judge (LLaMA 3.2), siguiendo un esquema pointwise que evalúa la progresión lógica y temporal de la narrativa.
2. **Avg Edge Coherence:** Similitud coseno promedio entre documentos consecutivos, calculada con sentence-transformers (all-MiniLM-L6-v2).
3. **Wall Time (segundos):** Tiempo de ejecución por experimento.
4. **Peak Memory (MB):** Consumo máximo de memoria.
5. **API Cost (USD):** Costo monetario (solo aplica a GPT-4o-mini).

### 2.5 Análisis Estadístico

- Test-t de Welch para comparaciones pairwise (no asume varianzas iguales)
- Cohen's d para tamaño del efecto
- Nivel de significancia: α = 0.05

## 3. Resultados

### 3.1 Resumen por Método

| Método | Judge Score | σ | Edge Coherence | Tiempo (s) | Costo (USD) | N experimentos |
|---|---|---|---|---|---|---|
| Trails Algorithmic | **8.00** | 0.00 | 0.694 | **0.02** | 0.00 | 12 |
| GPT-4o-mini | 7.98 | 0.44 | 0.577 | 7.94 | 0.11 | 58 |
| LLaMA 3.2 | 7.95 | 0.30 | **0.721** | 8.91 | 0.00 | 44 |
| Gemma 3 | 7.88 | 0.71 | 0.708 | 23.53 | 0.00 | 41 |
| Maps Algorithmic | 7.83 | 0.58 | 0.581 | 0.34 | 0.00 | 12 |
| Mistral | 7.77 | 0.86 | 0.625 | 22.94 | 0.00 | 26 |

### 3.2 Resultados por Tamaño de Narrativa

#### n = 6

| Método | Judge Score | Edge Coherence | Tiempo (s) |
|---|---|---|---|
| Gemma 3 | **8.06** | 0.577 | 15.85 |
| GPT-4o-mini | 8.05 | 0.586 | 4.58 |
| LLaMA 3.2 | 8.00 | 0.575 | 6.73 |
| Mistral | 8.00 | 0.572 | 18.43 |
| Trails | 8.00 | **0.714** | **0.01** |
| Maps | 7.50 | 0.576 | 0.62 |

#### n = 12

| Método | Judge Score | Edge Coherence | Tiempo (s) |
|---|---|---|---|
| Maps | **8.00** | 0.570 | 0.21 |
| Trails | **8.00** | 0.688 | **0.02** |
| LLaMA 3.2 | 7.83 | **0.794** | 11.37 |
| GPT-4o-mini | 7.79 | 0.569 | 7.75 |
| Gemma 3 | 7.57 | 0.785 | 28.54 |
| Mistral | 7.00 | 0.800 | 37.96 |

#### n = 18

| Método | Judge Score | Edge Coherence | Tiempo (s) |
|---|---|---|---|
| GPT-4o-mini | **8.11** | 0.577 | 11.65 |
| Maps | 8.00 | 0.597 | 0.19 |
| Trails | 8.00 | 0.681 | **0.03** |
| LLaMA 3.2 | 8.00 | **0.878** | 10.00 |
| Gemma 3 | 8.00 | 0.850 | 31.09 |
| Mistral | — | — | — |

*Mistral no completó experimentos para n=18.*

### 3.3 Comparaciones Estadísticas (Test-t de Welch)

De las 15 comparaciones pairwise realizadas sobre el Judge Score:

- **0 diferencias estadísticamente significativas** (p < 0.05)
- Todos los tamaños de efecto fueron **negligible** (|d| < 0.2) o **small** (0.2 ≤ |d| < 0.5)

| Comparación | Δ Score | p-value | Cohen's d | Efecto |
|---|---|---|---|---|
| Trails vs Maps | +0.17 | 0.339 | 0.41 | Small |
| GPT-4o-mini vs Mistral | +0.21 | 0.241 | 0.36 | Small |
| LLaMA 3.2 vs Mistral | +0.19 | 0.299 | 0.32 | Small |
| LLaMA 3.2 vs Maps | +0.12 | 0.496 | 0.32 | Small |
| GPT-4o-mini vs Maps | +0.15 | 0.411 | 0.32 | Small |
| Trails vs Mistral | +0.23 | 0.185 | 0.32 | Small |
| Gemma 3 vs Trails | -0.12 | 0.281 | 0.19 | Negligible |
| GPT-4o-mini vs Gemma 3 | +0.10 | 0.407 | 0.18 | Negligible |
| Gemma 3 vs LLaMA 3.2 | -0.08 | 0.528 | 0.14 | Negligible |
| Gemma 3 vs Mistral | +0.11 | 0.594 | 0.14 | Negligible |
| LLaMA 3.2 vs Trails | -0.05 | 0.323 | 0.17 | Negligible |
| GPT-4o-mini vs LLaMA 3.2 | +0.03 | 0.702 | 0.07 | Negligible |
| Gemma 3 vs Maps | +0.04 | 0.826 | 0.07 | Negligible |
| GPT-4o-mini vs Trails | -0.02 | 0.766 | 0.04 | Negligible |
| Mistral vs Maps | -0.06 | 0.789 | 0.08 | Negligible |

### 3.4 Coherencia por Embeddings vs. Judge Score

Se observa una **disociación parcial** entre las dos métricas de coherencia:

- **LLaMA 3.2** obtiene la mayor coherencia por embeddings (0.721) pero no el mayor Judge Score
- **Trails** tiene alta coherencia por embeddings (0.694) y el mejor Judge Score (8.00)
- **GPT-4o-mini** tiene la menor coherencia por embeddings (0.577) pero el segundo mejor Judge Score (7.98)

Esto sugiere que la similitud semántica entre documentos consecutivos (edge coherence) no captura completamente la coherencia narrativa global que evalúa el juez LLM.

### 3.5 Eficiencia y Costos

| Método | Tiempo total | Speedup vs más lento | Costo total |
|---|---|---|---|
| Trails Algorithmic | 0.23 s | **4,244×** | $0.00 |
| Maps Algorithmic | 4.13 s | 234× | $0.00 |
| LLaMA 3.2 | 401 s | 2.4× | $0.00 |
| GPT-4o-mini | 460 s | 2.1× | $0.11 |
| Mistral | 596 s | 1.6× | $0.00 |
| Gemma 3 | 965 s | 1× (ref.) | $0.00 |

## 4. Discusión

### 4.1 Hallazgo Principal: Equivalencia Estadística

El resultado más notable es la **ausencia de diferencias significativas** entre cualquier par de métodos. Todos los enfoques evaluados — algorítmicos, LLMs locales y propietarios — producen narrativas de calidad comparable según el juez LLM (rango: 7.77–8.00/10).

Esto tiene implicaciones importantes:

1. **Los métodos algorítmicos clásicos siguen siendo competitivos.** Narrative Trails iguala o supera a todos los LLMs en Judge Score, con un costo computacional varios órdenes de magnitud menor.

2. **Los LLMs locales pueden reemplazar a los propietarios.** LLaMA 3.2 (3B parámetros) obtiene prácticamente el mismo score que GPT-4o-mini (7.95 vs 7.98), sin costo monetario ni dependencia de APIs externas.

3. **El tamaño del modelo no es determinante.** Mistral (7B) obtiene peor score que LLaMA 3.2 (3B), sugiriendo que la capacidad de seguir instrucciones y generar JSON válido es más relevante que el tamaño del modelo.

### 4.2 Trade-off Calidad-Eficiencia

Si bien la calidad es equivalente, la eficiencia varía dramáticamente:

- **Trails Algorithmic** es el método dominante en el sentido de Pareto: mejor calidad y 4,000× más rápido que los LLMs.
- **GPT-4o-mini** es el único método con costo monetario ($0.11 para 58 experimentos), sin ventaja en calidad.
- **Gemma 3** es el más lento de los LLMs (23.5s por experimento), pero entrega la segunda mejor coherencia por embeddings.

### 4.3 Robustez de los LLMs

Los LLMs muestran diferencias en robustez según el tamaño de la narrativa:

- **n=6:** Todos los métodos son robustos. Los LLMs completan prácticamente todos los experimentos.
- **n=12:** Mistral muestra alta varianza (σ=1.67) y baja tasa de éxito (6/20).
- **n=18:** Mistral no completa ningún experimento. Gemma 3 completa solo 9/20.

Esto evidencia que **narrativas más largas son más difíciles para los LLMs** locales, que deben seleccionar más IDs y son más propensos a errores de formato.

### 4.4 Limitaciones

1. **Juez único:** Se utilizó un solo modelo (LLaMA 3.2) como juez. Un panel de jueces o evaluación humana podría revelar diferencias no capturadas.
2. **Corpus específico:** Los resultados aplican a un corpus de noticias chilenas; la generalización a otros dominios requiere validación.
3. **Baja varianza en Judge Score:** La mediana de 8/10 para casi todos los métodos sugiere posible efecto techo en la escala, lo que reduce la sensibilidad para detectar diferencias.
4. **Phi-3 excluido:** No fue posible evaluar este modelo por incapacidad de generar respuestas estructuradas.
5. **Repeticiones algorítmicas:** Los métodos algorítmicos (1 repetición por configuración, deterministas) tienen menor N que los LLMs, lo que reduce la potencia estadística para comparaciones que los involucran.

## 5. Conclusiones

1. **No existen diferencias estadísticamente significativas** en la calidad narrativa entre métodos algorítmicos y basados en LLMs para este corpus y diseño experimental.

2. **Narrative Trails es el método óptimo** considerando el trade-off calidad-eficiencia: obtiene el mejor Judge Score (8.00/10) con tiempos de ejecución de milisegundos.

3. **Los LLMs locales son alternativas viables** a los propietarios. LLaMA 3.2 alcanza calidad equivalente a GPT-4o-mini sin costo monetario, aunque con mayor tiempo de cómputo.

4. **La escalabilidad es un desafío para los LLMs:** su tasa de éxito disminuye significativamente para narrativas largas (n≥12), particularmente en modelos más pequeños o menos orientados a instrucciones.

5. **La coherencia por embeddings no es un proxy perfecto** de la calidad narrativa global. Las dos métricas capturan aspectos distintos de la coherencia.

## 6. Trabajo Futuro

- Evaluación con jueces humanos para validar las puntuaciones del LLM-as-a-Judge
- Extensión a otros corpus (noticias internacionales, textos científicos)
- Exploración de métodos híbridos: selección algorítmica + refinamiento por LLM
- Evaluación de modelos más recientes con mayor capacidad de contexto

---

## Anexos

### A. Configuración del Entorno

- **Hardware:** Apple Silicon (Mac)
- **Modelos locales:** Ollama (LLaMA 3.2 3B, Mistral 7B, Gemma 3 4B)
- **Embeddings:** all-MiniLM-L6-v2 (sentence-transformers)
- **Solver LP:** PuLP + CBC (para Narrative Maps)

### B. Archivos de Resultados

- Resultados JSON: `results_local/results_v2/`
- Tablas estadísticas: `results_local/analysis/tables/`
- Visualizaciones: `results_local/analysis/plots/`
- Script de experimentos: `run_experiments_local.py`
- Script de análisis: `analyze_results.py`
