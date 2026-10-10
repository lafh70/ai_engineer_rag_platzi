# Guía paso a paso: RAG sobre Azure AI Search (todo lo hecho hoy)

Cada paso incluye **qué hicimos**, **por qué lo hicimos así** y **para qué sirve** dentro del proyecto completo.

---

## Paso 1 — Entorno de trabajo (Jupyter + VS Code)

**Qué hicimos:** instalamos `jupyter` e `ipykernel` en el venv del proyecto, registramos un kernel llamado `clase7`, e instalamos la extensión de Jupyter en VS Code (`ms-toolsai.jupyter`).

**Por qué:** para poder escribir y ejecutar código Python de forma interactiva, celda por celda, en vez de solo scripts que corren de punta a punta. Esto es clave para experimentar y ver resultados intermedios.

**Para qué:** era el requisito previo para crear `clase7.ipynb`, el notebook donde se construyó y probó todo el pipeline de RAG.

---

## Paso 2 — Almacenamiento de documentos (Azure Blob Storage)

**Qué hicimos:** creamos una Storage Account (`stragdocs9203`, SKU `Standard_LRS`) y un contenedor llamado `rag-documents`, y subimos los 7 PDFs de ejemplo (`curso-rag-main/data_sample`).

**Por qué:** un sistema RAG necesita una fuente de documentos centralizada y accesible por API, no solo archivos locales. `Standard_LRS` es la redundancia más económica, suficiente para un proyecto de prueba.

**Para qué:** sirvió como el "origen de la verdad" de todo el pipeline: todo lo que vectorizamos después salió de ahí.

---

## Paso 3 — Permisos (RBAC)

**Qué hicimos:** asignamos el rol **Storage Blob Data Contributor** a tu usuario sobre la Storage Account, porque la subida de blobs con `--auth-mode login` (autenticación por identidad, sin claves) fallaba por falta de permisos.

**Por qué:** es la forma recomendada por Azure de trabajar con Storage sin depender de compartir claves de cuenta. El rol se asignó vía API REST porque el comando normal de la CLI (`az role assignment create`) daba un error propio de la herramienta.

**Para qué:** permitió subir y gestionar los PDFs sin tener que usar la clave de la cuenta cada vez.

---

## Paso 4 — Vectorización de los documentos

**Qué hicimos:**
- Evaluamos usar Azure OpenAI para generar los embeddings, pero decidimos usar un modelo **local y gratuito**: `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensiones).
- Creamos un servicio de **Azure AI Search** en tier **Free (F0)** (`srch-ragdocs-9203`), con un índice vectorial `rag-documents-index`.
- Construimos el pipeline: descarga de PDFs → extracción de texto (`pypdf`) → chunking (800 caracteres, overlap 150) → generación de embeddings → creación del índice → carga de los 977 chunks resultantes.

**Por qué:** priorizamos siempre la opción de costo $0 sobre crear recursos de pago. Azure OpenAI tiene costo por token; el modelo local no.

**Para qué:** convertir el contenido de los PDFs (texto plano) en vectores numéricos que capturan su significado, de forma que después se pueda buscar por "parecido semántico" y no solo por palabras exactas.

---

## Paso 5 — Consulta RAG (búsqueda + respuesta con LLM)

**Qué hicimos:**
- Revisamos un notebook de referencia del curso (`clase9/libro1.ipynb`) que usaba Azure OpenAI contra otro índice distinto al nuestro, y adaptamos su lógica a nuestro propio stack.
- Armamos una celda que vectoriza la pregunta del usuario y busca los chunks más parecidos en `rag-documents-index`.
- Agregamos un LLM (`gpt-4o-mini`, usando tu propia API key de OpenAI directo, no Azure OpenAI) que lee esos chunks y redacta una respuesta en lenguaje natural.
- Hicimos la pregunta **interactiva** (`input()`) en vez de hardcodeada, para que cada persona pueda preguntar lo que quiera al correr el notebook manualmente.

**Por qué:** la búsqueda sola devuelve fragmentos de texto crudos, difíciles de leer; el LLM los convierte en una respuesta directa. Usamos tu cuenta personal de OpenAI (no Azure OpenAI) para evitar crear otro recurso de pago en Azure.

**Para qué:** esta es la experiencia final de un sistema RAG: el usuario pregunta algo en lenguaje natural y recibe una respuesta basada únicamente en los documentos que subimos.

---

## Paso 6 — Script Python reutilizable (`rag_pipeline/`)

**Qué hicimos:** extrajimos todo el flujo del notebook a un script standalone (`rag_pipeline.py`), movimos las claves a un archivo `.env` (vía `python-dotenv`), y organizamos todo en su propia carpeta `rag_pipeline/` con un `.gitignore`.

**Por qué:** separar credenciales del código es una buena práctica de seguridad (evita subir claves a un repositorio por error). Un script además es más fácil de automatizar o programar que un notebook.

**Para qué:** poder correr todo el pipeline de principio a fin con un solo comando (`python rag_pipeline.py`), sin depender de abrir Jupyter.

---

## Paso 7 — Comparación de estrategias de búsqueda (`search_comparison/`)

**Qué hicimos:** armamos un script (`compare_search.py`) que corre la misma pregunta usando **tres formas de buscar** en el mismo índice:
1. **Texto tradicional (BM25):** coincidencia de palabras clave.
2. **Vectorial:** similitud semántica entre embeddings.
3. **Híbrida:** combina ambas (Azure AI Search las fusiona automáticamente con Reciprocal Rank Fusion).

Probamos con dos tipos de pregunta: una conceptual (sin palabras clave exactas) y otra con un término exacto y distintivo ("PerksPlus").

**Por qué:** cada estrategia tiene fortalezas distintas. La de texto es rápida y precisa cuando hay palabras clave exactas, pero falla con preguntas conceptuales. La vectorial entiende el significado, pero puede perder precisión con términos exactos poco comunes. La híbrida busca combinar lo mejor de ambas.

**Para qué:** entender en la práctica cuándo conviene cada estrategia, en vez de asumir que "vectorial siempre es mejor". Resultado concreto: con la pregunta conceptual, texto y vector casi no coincidieron (1/5); con la pregunta de término exacto, las tres coincidieron 100% (5/5).

---

## Paso 8 — Indexación masiva con múltiples configuraciones (`bulk_indexing/`)

**Qué hicimos:** automatizamos el proceso completo (descarga → chunking → embeddings → carga) para generar **3 índices distintos** con la misma información fuente, variando el tamaño de chunk y el algoritmo de búsqueda vectorial:

| Índice | Chunk / overlap | Algoritmo |
|---|---|---|
| `rag-documents-index` | 800 / 150 | HNSW (aproximado) |
| `rag-index-chunks-small` | 300 / 50 | HNSW (aproximado) |
| `rag-index-exhaustive` | 1500 / 200 | ExhaustiveKNN (exacto) |

Todo dentro del límite de 3 índices del tier Free (sin costo).

**Por qué:** el tamaño de chunk y el algoritmo de búsqueda vectorial afectan la calidad de los resultados (fragmentos chicos = más precisión local pero menos contexto; HNSW = más rápido pero aproximado; ExhaustiveKNN = exacto pero más lento). Probar varias configuraciones en paralelo permite comparar cuál funciona mejor para este contenido específico, en vez de adivinar.

**Para qué:** tener una base para decidir, con datos reales, qué configuración de chunking/algoritmo usar en un sistema RAG de producción.

---

## Paso 9 — CORS para la demo app de Azure AI Search

**Qué hicimos:** habilitamos `corsOptions` (orígenes permitidos `*`) en los 3 índices.

**Por qué:** Azure permite generar una app web de demostración directamente desde el portal con un clic ("Create Demo App"), pero requiere que el índice tenga CORS habilitado para que el navegador pueda llamarlo desde otro dominio.

**Para qué:** poder probar visualmente las consultas RAG (ver los fragmentos de texto y los vectores de fondo) sin tener que programar una interfaz propia.

---

## Paso 10 — Prueba de Azure OpenAI + GPT en AI Foundry (creado y luego eliminado)

**Qué hicimos:**
- Creamos un recurso Azure OpenAI (`openai-ragdocs-9203`) para poder usar un modelo de chat en el Chat Playground de AI Foundry conectado a nuestro índice.
- Descubrimos que `gpt-4o-mini` ya estaba deprecado (desde el 31/03/2026) y desplegamos en su lugar `gpt-5-nano`, el modelo más económico disponible en ese momento.
- Probamos el deployment con una llamada directa (`curl`) para confirmar que el endpoint y la autenticación funcionaban.
- **Eliminamos y purgamos** el recurso completo apenas terminamos la prueba.

**Por qué:** querías explorar la integración con AI Foundry, pero sin dejar un recurso de pago corriendo indefinidamente. Crear → probar → destruir es el patrón correcto para experimentos puntuales.

**Para qué:** validar que la integración (índice + modelo de chat) es técnicamente viable, dejando documentado el proceso exacto (incluyendo los problemas que surgieron: modelo deprecado, SKU incorrecto) para no tener que redescubrirlo la próxima vez.

---

## Paso 11 — Auditoría completa de costos y limpieza total

**Qué hicimos:**
- Revisamos **todos** los recursos de la suscripción, incluyendo uno que no habíamos creado nosotros (`luisalfonso634-2958-resource`, un recurso de Cognitive Services preexistente).
- A pedido explícito, eliminamos **absolutamente todo**: ese recurso preexistente, la Storage Account (con los PDFs), el servicio de Search (los 3 índices) y el Automation Account.
- Cancelamos la suscripción Pay-As-You-Go completa (quedó en estado transicional `Warned`, camino a `Disabled`).

**Por qué:** decidiste que no querías dejar nada activo en Azure, ni siquiera recursos de costo insignificante como el storage con 3.84 MB de datos. Es una decisión válida de control total de costos, aunque implicó borrar el trabajo del día (PDFs, índices).

**Para qué:** garantizar que no queda ningún cargo, ni potencial ni real, corriendo en tu cuenta de Azure después de esta sesión de aprendizaje.

---

## Instrucciones permanentes que quedaron establecidas durante el proceso

1. **Pedir autorización y mostrar el costo estimado** antes de crear cualquier recurso de Azure que genere costos.
2. **Usar siempre, por defecto, la opción más económica disponible** (tier, modelo, SKU), sin necesidad de preguntar cada vez cuál usar.

Estas instrucciones guiaron casi todas las decisiones técnicas de este documento: por qué elegimos embeddings locales en vez de Azure OpenAI, por qué usamos tu API key de OpenAI directo en vez de desplegar un recurso nuevo, por qué el tier Free de Search, y por qué el ciclo completo de crear-probar-destruir en el experimento de AI Foundry.
