"""
Genera la matriz de coherencia completa (pairwise cosine similarity)
para los 200 documentos muestreados sistemáticamente del corpus.

Output:
  coherence_matrix/coherence_matrix_200docs.csv   — matriz con IDs como índices
  coherence_matrix/coherence_matrix_200docs.npy   — array numpy (200x200)
  coherence_matrix/doc_ids.txt                    — lista de IDs en orden
"""

import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "narrative-extraction/data/corpus.jsonl"
MAX_DOCS = 200
OUTPUT_DIR = "coherence_matrix"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Cargar corpus ──────────────────────────────────────────────────────────
print("Cargando corpus...")
with open(CORPUS_PATH) as f:
    all_docs = [json.loads(line) for line in f]

# Mismo sampling que run_experiments_local.py
all_docs.sort(key=lambda d: d["date"])
if len(all_docs) > MAX_DOCS:
    step = len(all_docs) / MAX_DOCS
    indices = [int(i * step) for i in range(MAX_DOCS)]
    docs = [all_docs[i] for i in indices]
else:
    docs = all_docs

print(f"Docs seleccionados: {len(docs)} (de {len(all_docs)} totales)")
print(f"Rango fechas: {docs[0]['date']} → {docs[-1]['date']}")

# ── 2. Embeddings ─────────────────────────────────────────────────────────────
print("\nCargando modelo sentence-transformers (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc["text"] for doc in docs]
doc_ids = [doc["id"] for doc in docs]

print(f"Computando embeddings para {len(texts)} documentos...")
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32)
print(f"Embeddings shape: {embeddings.shape}")

# ── 3. Matriz de coherencia (coseno) ──────────────────────────────────────────
print("\nCalculando matriz de similitud coseno...")
# Como los embeddings están normalizados, coseno = producto punto
coherence_matrix = embeddings @ embeddings.T
np.fill_diagonal(coherence_matrix, 1.0)  # por precisión numérica
print(f"Matriz shape: {coherence_matrix.shape}")
print(f"  min={coherence_matrix.min():.4f}  max={coherence_matrix.max():.4f}  mean={coherence_matrix.mean():.4f}")

# ── 4. Guardar ────────────────────────────────────────────────────────────────
# .npy
npy_path = os.path.join(OUTPUT_DIR, "coherence_matrix_200docs.npy")
np.save(npy_path, coherence_matrix)
print(f"\nGuardado: {npy_path}")

# .csv con IDs
csv_path = os.path.join(OUTPUT_DIR, "coherence_matrix_200docs.csv")
df = pd.DataFrame(coherence_matrix, index=doc_ids, columns=doc_ids)
df.to_csv(csv_path, float_format="%.6f")
print(f"Guardado: {csv_path}")

# lista de IDs
ids_path = os.path.join(OUTPUT_DIR, "doc_ids.txt")
with open(ids_path, "w") as f:
    for i, doc_id in enumerate(doc_ids):
        f.write(f"{i}\t{doc_id}\t{docs[i]['date']}\t{docs[i]['title'][:80]}\n")
print(f"Guardado: {ids_path}")

print("\nListo. Archivos en coherence_matrix/")
