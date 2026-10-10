# ai_engineer_rag_platzi

Curso de RAG de Platzi — pipeline de Retrieval-Augmented Generation construido paso a paso sobre Azure (Storage + AI Search), con embeddings locales y consulta vía LLM.

## Qué hicimos, paso a paso, y para qué

### 1. Entorno de trabajo
Instalamos Jupyter + un kernel dedicado (`clase7`) y la extensión de Jupyter en VS Code.
**Para qué:** poder desarrollar y probar el pipeline celda por celda antes de consolidarlo en scripts.

### 2. Almacenamiento de documentos (Azure Blob Storage)
Creamos una Storage Account (`Standard_LRS`, la redundancia más económica) y un contenedor `rag-documents`, y subimos los 7 PDFs de ejemplo del curso.
**Para qué:** tener una fuente de documentos centralizada y accesible por API — el origen de todo lo que se vectoriza después.

### 3. Permisos (RBAC)
Asignamos el rol **Storage Blob Data Contributor** para poder subir archivos autenticando con la identidad del usuario, sin depender de la clave de la cuenta.
**Para qué:** seguir la práctica recomendada de Azure para acceso a datos sin compartir secretos de cuenta.

### 4. Vectorización de los documentos
Elegimos generar los embeddings **localmente** con `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensiones) en vez de usar Azure OpenAI, para evitar costos. Creamos un servicio de **Azure AI Search tier Free (F0)** y el pipeline completo: descarga de PDFs → extracción de texto → chunking (800 caracteres, overlap 150) → embeddings → creación de índice → carga masiva.
**Para qué:** convertir el contenido de los documentos en vectores que permiten buscar por significado, no solo por palabras exactas.

### 5. Consulta RAG (búsqueda + LLM)
Armamos una consulta interactiva: el usuario escribe una pregunta, se vectoriza, se buscan los chunks más relevantes en el índice, y un LLM (`gpt-4o-mini`, usando una API key de OpenAI directo para evitar desplegar Azure OpenAI) redacta la respuesta final a partir de esos chunks.
**Para qué:** esta es la experiencia completa de un sistema RAG — preguntar en lenguaje natural y recibir una respuesta basada únicamente en los documentos propios.

### 6. Script reutilizable (`rag_pipeline/`)
Extrajimos todo el flujo del notebook a un script (`rag_pipeline.py`), con las credenciales movidas a un `.env` (no versionado).
**Para qué:** poder correr el pipeline completo con un solo comando, sin depender de abrir Jupyter, y mantener las claves fuera del código fuente.

### 7. Comparación de estrategias de búsqueda (`search_comparison/`)
Comparamos, sobre el mismo índice, tres formas de buscar: **texto tradicional (BM25)**, **vectorial (semántica)** e **híbrida** (ambas combinadas). Probamos con una pregunta conceptual y otra con un término exacto.
**Para qué:** entender con evidencia real cuándo conviene cada estrategia — con la pregunta conceptual, texto y vector casi no coincidieron; con el término exacto, las tres coincidieron al 100%.

### 8. Indexación masiva multi-configuración (`bulk_indexing/`)
Automatizamos la generación de **3 índices** con la misma información fuente, variando tamaño de chunk y algoritmo vectorial (HNSW aproximado vs. ExhaustiveKNN exacto), todo dentro del límite gratuito de 3 índices del tier Free.
**Para qué:** comparar con datos reales qué configuración de chunking/algoritmo funciona mejor, en vez de adivinar.

### 9. CORS para la demo app de Azure
Habilitamos CORS en los índices.
**Para qué:** poder usar la función "Create Demo App" del portal de Azure y probar las consultas visualmente, viendo texto y vectores de fondo.

### 10. Prueba de Azure OpenAI / AI Foundry (creado y luego eliminado)
Desplegamos temporalmente un modelo de chat (`gpt-5-nano`, tras descubrir que `gpt-4o-mini` ya estaba deprecado) para validar la integración con el Chat Playground de AI Foundry, y **eliminamos y purgamos** el recurso apenas terminó la prueba.
**Para qué:** validar la integración LLM + índice sin dejar un recurso de pago corriendo indefinidamente — patrón "crear → probar → destruir".

### 11. Auditoría y limpieza de costos
Revisamos todos los recursos de la suscripción (incluyendo uno preexistente no reconocido) y, a pedido explícito, eliminamos todo y cancelamos la suscripción Pay-As-You-Go.
**Para qué:** garantizar que no queda ningún cargo activo ni potencial en la cuenta de Azure después de la sesión de aprendizaje.

## Reglas que guiaron el proceso

1. **Pedir autorización y mostrar el costo estimado** antes de crear cualquier recurso de Azure que genere costos.
2. **Usar siempre, por defecto, la opción más económica disponible** (tier, modelo, SKU).

## Estructura del repo

```
rag_pipeline/          # Script completo del pipeline RAG (Storage -> Search -> LLM)
search_comparison/     # Comparación texto vs. vectorial vs. híbrida
bulk_indexing/         # Indexación en 3 configuraciones (chunk size + algoritmo)
curso-rag-main/        # Material del curso (PDFs de ejemplo, notebooks de referencia)
memoria.md             # Registro narrativo de decisiones tomadas y el por qué
guia_paso_a_paso.md    # Guía detallada paso a paso de todo el proceso
rag-azure-skill.md     # Prompt reutilizable tipo "skill" para replicar el proceso en otro entorno
```

Cada carpeta de script (`rag_pipeline/`, `search_comparison/`, `bulk_indexing/`) tiene su propio `.env.example`: copiarlo a `.env` y completar con las credenciales propias antes de ejecutar.
