# Guía de estudio: RAG sobre Azure AI Search

Guía pensada para **repasar y entender** el proceso completo, no solo para ejecutarlo. Cada sección tiene: concepto clave, preguntas para autoevaluarte, y dónde ver el código real en este repo.

---

## 1. ¿Qué es RAG y por qué existe?

**Concepto:** RAG (Retrieval-Augmented Generation) combina dos cosas que un LLM no hace bien solo:
- Un LLM no "sabe" tus documentos privados (no estaban en su entrenamiento).
- Un LLM puede alucinar si le pedís datos específicos sin contexto.

La solución: **buscar** primero la información relevante en tus propios documentos, y después **pasársela al LLM** como contexto para que genere la respuesta.

```
Pregunta del usuario
      │
      ▼
 [1] Vectorizar la pregunta  ──────┐
      │                            │
      ▼                            │
 [2] Buscar en el índice           │  (esto es "Retrieval")
      │                            │
      ▼                            │
 [3] Chunks relevantes ────────────┘
      │
      ▼
 [4] LLM + chunks como contexto     (esto es "Generation", aumentada)
      │
      ▼
 Respuesta final
```

**Autoevaluación:**
- ¿Por qué no alcanza con pasarle todo el PDF al LLM directamente? *(pista: límite de contexto, costo por token, precisión)*
- ¿Qué pasaría si el paso de búsqueda (Retrieval) devuelve basura? *(pista: "garbage in, garbage out" — el LLM solo puede responder tan bien como el contexto que recibe)*

---

## 2. Chunking: por qué partir los documentos

**Concepto:** no se vectoriza el PDF entero de una — se parte en fragmentos (`chunks`) de tamaño manejable, con **overlap** (solapamiento) entre fragmentos consecutivos.

```
Texto completo: [AAAAAAAAAABBBBBBBBBBCCCCCCCCCC...]

Con chunk_size=10, overlap=3:
chunk 1: AAAAAAAAAA
chunk 2:        ABBBBBBBBB   <- repite las últimas 3 letras del chunk anterior
chunk 3:               BBCCCCCCCCC
```

**Por qué el overlap:** sin él, una idea que cruza el límite de un chunk ("el plan Plus incluye | cobertura dental") se parte en dos fragmentos que, por separado, no significan nada completo.

**Trade-off de tamaño:**
| Chunk chico | Chunk grande |
|---|---|
| Más preciso (el embedding representa una idea específica) | Más contexto por fragmento |
| Más fragmentos → más almacenamiento/cómputo | Menos fragmentos, pero cada uno "mezcla" varios temas |

**Código real:** ver `chunk_text()` en `rag_pipeline/rag_pipeline.py` y las 3 configuraciones distintas en `bulk_indexing/bulk_index_configs.py`.

**Autoevaluación:**
- Si tu documento tiene párrafos de 200 caracteres cada uno, ¿tiene sentido usar chunk_size=1500? ¿Por qué sí o no?
- ¿Qué overlap usarías si tus documentos son listas de ítems cortos sin relación entre sí?

---

## 3. Embeddings: convertir texto en números

**Concepto:** un modelo de embeddings transforma texto en un vector (lista de números) que representa su significado. Dos textos con significado parecido producen vectores "cercanos" en ese espacio matemático.

```
"El plan cubre dental"     → [0.12, -0.45, 0.78, ...]  (384 números)
"La cobertura incluye odontología" → [0.14, -0.41, 0.80, ...]  (vector MUY parecido, aunque las palabras son distintas)
"El clima en Madrid"       → [-0.90, 0.33, -0.12, ...]  (vector muy distinto)
```

**Decisión clave de este proyecto:** usamos un modelo **local** (`sentence-transformers`, `all-MiniLM-L6-v2`) en vez de un servicio de pago (Azure OpenAI embeddings). Corre en tu CPU, es gratis, y para prototipos es suficiente.

**Código real:** ver la celda/función que carga `SentenceTransformer` en `rag_pipeline/rag_pipeline.py`.

**Autoevaluación:**
- ¿Por qué dos preguntas con las mismas palabras clave pero distinto orden ("plan Plus vs Standard" vs "Standard vs Plus plan") generan vectores casi idénticos?
- ¿Qué pasaría si vectorizaras los documentos con un modelo y la pregunta con otro distinto? *(pista: viven en "espacios" distintos, no son comparables)*

---

## 4. Azure AI Search: el motor de búsqueda vectorial

**Concepto:** es una base de datos especializada en guardar vectores y encontrar rápidamente los "más parecidos" a un vector de consulta (búsqueda por similitud, no por igualdad exacta).

**Decisión clave:** tier **Free (F0)** — $0/mes, hasta 3 índices, 50MB. Suficiente para prototipos; no apto para producción con mucho volumen.

**Dos algoritmos de búsqueda vectorial que comparamos:**
| HNSW | ExhaustiveKNN |
|---|---|
| Aproximado (no garantiza el resultado 100% exacto) | Exacto (compara contra absolutamente todos los vectores) |
| Mucho más rápido en datasets grandes | Más lento, pero perfecto si el dataset es chico |
| Lo normal en producción | Útil para medir qué tan "exacto" pierde precisión HNSW |

**Código real:** `create_index()` en `rag_pipeline/rag_pipeline.py` y `bulk_indexing/bulk_index_configs.py` (ahí se usan ambos algoritmos en índices distintos, a propósito, para comparar).

**Autoevaluación:**
- ¿Por qué Azure AI Search no te deja cambiar el algoritmo de un índice ya creado sin borrarlo? *(pista: la estructura interna de datos es distinta para cada algoritmo)*
- Si tenés 50 documentos, ¿importa mucho la diferencia de velocidad entre HNSW y ExhaustiveKNN? ¿Y si tenés 50 millones?

---

## 5. Las tres estrategias de búsqueda (y cuándo gana cada una)

Repasá los resultados reales que obtuvimos en `search_comparison/compare_search.py`:

| Estrategia | Cómo funciona | Gana cuando... |
|---|---|---|
| **Texto (BM25)** | Cuenta coincidencias de palabras clave (con pesos por frecuencia/rareza) | La pregunta usa el **vocabulario literal** del documento correcto (ej. "PerksPlus") |
| **Vectorial** | Compara significado semántico | La pregunta es **conceptual**, sin las palabras exactas del documento (ej. "qué tiene el plan superior que no tiene el básico") |
| **Híbrida** | Combina ambas (Reciprocal Rank Fusion) | En teoría, lo mejor de ambos mundos — pero en la práctica, si una señal domina mucho, la híbrida se parece más a esa |

**Resultado real que obtuvimos:**
- Pregunta conceptual → texto y vector casi no coincidieron (1 de 5 resultados en común).
- Pregunta con término exacto ("PerksPlus") → las tres estrategias coincidieron al 100%.

**Autoevaluación:**
- ¿Por qué un buscador de e-commerce (buscar "iPhone 15") se beneficiaría más de texto puro que de vectorial?
- ¿Por qué un chatbot de soporte técnico (preguntas largas y variadas) se beneficia más de vectorial?
- ¿Se te ocurre un caso donde la búsqueda híbrida claramente le gane a ambas por separado? *(pista: pregunta con un código de producto exacto DENTRO de una oración conceptual larga)*

---

## 6. RAG completo: de la búsqueda a la respuesta

**Concepto:** después de recuperar los chunks relevantes, se los pasa a un LLM como **contexto** dentro del prompt, junto con una instrucción de "responder solo con lo que está en el contexto" (para reducir alucinaciones).

```python
SYSTEM_MESSAGE = f"""
Sos un asistente. Respondé SOLO con los hechos del contexto.
Contexto:
{context}
"""
```

**Decisión clave:** usamos una API key de OpenAI directo (no Azure OpenAI) para evitar crear un recurso de pago nuevo en Azure, y elegimos el modelo más económico disponible.

**Código real:** celda final de `clase7.ipynb` y función `ask()` en `rag_pipeline/rag_pipeline.py`.

**Autoevaluación:**
- ¿Por qué la instrucción "respondé solo con los hechos del contexto" reduce (pero no elimina) las alucinaciones?
- Si el buscador devuelve chunks irrelevantes, ¿qué va a pasar con la respuesta del LLM aunque el modelo sea excelente?

---

## 7. Gestión de costos en la nube (la lección "no técnica" más importante)

Reglas aplicadas durante **todo** el proyecto:
1. Pedir autorización y mostrar el costo **antes** de crear un recurso de pago.
2. Usar siempre, por defecto, la opción más económica (tier Free, modelo más chico).
3. Patrón **crear → probar → destruir** para experimentos puntuales (ver el caso de Azure OpenAI/AI Foundry: se creó, se probó con un `curl`, y se eliminó y purgó en la misma sesión).
4. Antes de borrar algo a pedido del usuario, **separar categorías** (qué es de hoy, qué genera costo real vs. insignificante, qué es ajeno al proyecto) y confirmar cada una por separado.

**Autoevaluación:**
- ¿Por qué un recurso "eliminado" (soft-delete) puede seguir reservando cuota o nombre hasta que se "purga"?
- ¿Por qué verificar la lista de modelos disponibles ANTES de desplegar, en vez de asumir que un nombre de modelo conocido sigue vigente? *(pista real de este proyecto: `gpt-4o-mini` se descubrió deprecado al intentar desplegarlo)*

---

## Mapa de estudio sugerido (orden de lectura del repo)

1. `rag_pipeline/rag_pipeline.py` — el pipeline completo de punta a punta, para entender el flujo general.
2. `search_comparison/compare_search.py` — correrlo con distintas preguntas propias y observar cuándo cambia el ganador.
3. `bulk_indexing/bulk_index_configs.py` — comparar los 3 índices generados y pensar qué configuración usarías para tu propio caso de uso.
4. `rag-azure-skill.md` — si vas a replicar esto en otro proyecto, es la referencia paso a paso con el *por qué* de cada decisión.
5. `memoria.md` y `guia_paso_a_paso.md` — el registro narrativo completo, útil para repasar decisiones y errores que surgieron en el camino (y cómo se resolvieron).

## Ejercicio final sugerido

Sin mirar el código: diseñá en papel un sistema RAG para un caso propio (ej. buscar en tus apuntes de clase, en la documentación interna de tu trabajo, etc.). Para cada paso del pipeline, definí:
- ¿Qué tamaño de chunk tendría sentido para tus documentos?
- ¿Texto, vectorial o híbrida sería tu estrategia de búsqueda por defecto, y por qué?
- ¿Qué tier/modelo usarías para mantener el costo en $0 mientras prototipás?

Después, compará tu diseño con las decisiones reales tomadas en este proyecto.
