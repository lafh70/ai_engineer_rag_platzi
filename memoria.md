# Memoria del proyecto: RAG sobre documentos en Azure

Registro paso a paso de lo construido en esta carpeta, las decisiones tomadas y el por qué.

## 1. Entorno de trabajo (Jupyter + VS Code)

- Se instaló `jupyter` e `ipykernel` en el venv del proyecto.
- Se registró un kernel llamado **"clase7"** (`python -m ipykernel install --user --name=clase7`) para que el notebook `clase7.ipynb` corra con las dependencias del proyecto.
- Se instaló la extensión de Jupyter en VS Code (`ms-toolsai.jupyter`) para poder ejecutar notebooks desde el editor.

**Por qué:** era la base necesaria para trabajar con notebooks y conectarlos al entorno Python correcto del proyecto.

## 2. Almacenamiento de documentos (Azure Blob Storage)

- Se creó una **Storage Account** (`stragdocs9203`) en el resource group `rg-luisalfonso634-9203`, región `westus3`, SKU `Standard_LRS` (la más económica).
- Se creó un contenedor llamado **`rag-documents`** dentro de esa cuenta.
- Se subieron los 7 PDFs de `curso-rag-main/data_sample` (primera_ronda y segunda_ronda) a ese contenedor, preservando la estructura de carpetas.

**Por qué:** se necesitaba un lugar centralizado en la nube para los documentos fuente antes de vectorizarlos, en lugar de trabajar solo con archivos locales.

**Decisión de costo:** Standard_LRS es la opción más barata de redundancia de almacenamiento, suficiente para un caso de prueba/aprendizaje sin necesidad de alta disponibilidad geográfica.

## 3. Permisos (RBAC)

- Inicialmente la subida de blobs con `--auth-mode login` falló por falta de rol de datos.
- Se asignó el rol **Storage Blob Data Contributor** al usuario sobre la storage account (vía API REST de Azure, porque el comando normal de `az role assignment create` daba un error de la propia CLI).
- Luego de la propagación del rol, la subida con `--auth-mode login` (autenticación con la identidad del usuario, sin claves) funcionó correctamente.

**Por qué:** es la forma recomendada de trabajar con Storage sin depender de claves de cuenta compartidas.

## 4. Vectorización de los documentos

- Se exploró primero crear un crédito de $200 USD de Azure (cuenta "free account"), pero se determinó que la suscripción activa es **Pay-As-You-Go** (sin ese crédito) y que la suscripción que sí tenía el crédito fue **eliminada** (recuperable solo dentro de la ventana de gracia de Azure).
- Para vectorizar, se evaluaron dos caminos: Azure OpenAI (de pago) vs. un modelo local de embeddings. **Se eligió el modelo local** `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensiones) para evitar costos.
- Se creó un servicio de **Azure AI Search** en tier **Free (F0)** ($0/mes) llamado `srch-ragdocs-9203`, con un índice vectorial `rag-documents-index` (HNSW).
- Pipeline de vectorización:
  1. Descarga de los 7 PDFs desde el contenedor `rag-documents`.
  2. Extracción de texto con `pypdf` y chunking (800 caracteres, overlap de 150).
  3. Generación de embeddings localmente con `sentence-transformers` (977 chunks en total).
  4. Creación del índice vectorial en Azure AI Search.
  5. Carga de los 977 documentos vectorizados al índice (en lotes de 100).

**Por qué:** se priorizó siempre la opción de costo $0 sobre crear recursos de pago (Azure OpenAI), siguiendo la instrucción explícita del usuario de usar siempre lo más económico por defecto.

## 5. Consulta RAG (búsqueda + LLM)

- Se revisó un notebook de referencia del curso (`curso-rag-main/clase9/libro1.ipynb`) que usaba Azure OpenAI (embeddings `ada-002` + chat `gpt-4o`) contra otro índice/servicio de Search que no era el nuestro.
- Se adaptó esa lógica a nuestro stack: la pregunta del usuario se vectoriza con el mismo modelo local, se busca por similitud vectorial en `rag-documents-index`, y se recuperan los chunks más relevantes.
- Para la respuesta en lenguaje natural, el usuario aportó su propia **API key de OpenAI** (cuenta personal, no Azure OpenAI), evitando así crear un recurso nuevo de pago en Azure.
- Se usó el modelo **`gpt-4o-mini`** (el más económico de los disponibles) para sintetizar la respuesta final a partir de los chunks recuperados.

**Por qué:** se evitó desplegar Azure OpenAI (recurso de pago que requiere aprovisionamiento) reutilizando una cuenta ya existente del usuario, y se eligió el modelo de chat más barato disponible.

## 6. Control de costos y permisos de Azure

- Se detectó que la suscripción no tenía el crédito de $200 USD activo (es Pay-As-You-Go, `spendingLimit: Off`), y que la suscripción que sí lo tenía fue eliminada.
- Se evaluó (sin ejecutar) un mecanismo de budget + acción automática para cortar gasto a partir de $1 USD, incluyendo la creación de un **Automation Account** (`aa-budget-guard`) como primer paso de esa cadena. Se pausó esa construcción cuando el usuario estableció la instrucción permanente de pedir autorización y costo antes de crear cualquier recurso de pago.
- Instrucciones permanentes establecidas por el usuario durante el proceso:
  - Pedir autorización y mostrar el costo estimado antes de crear cualquier recurso de Azure que genere costos.
  - Usar siempre, por defecto, la opción más económica disponible (tier, modelo, SKU) sin necesidad de preguntar cada vez.
- Se confirmó que cancelar la suscripción Pay-As-You-Go es posible pero el usuario decidió **esperar** antes de hacerlo, dado que afectaría los recursos ya creados (storage con los PDFs, Automation Account).

**Por qué:** mantener el proyecto dentro de costos controlados y con el usuario siempre en control de decisiones que impliquen gasto.

## 7. Script Python reutilizable

- Se extrajo todo el flujo del notebook a un script standalone: `rag_pipeline.py`, con las claves movidas a un archivo `.env` (vía `python-dotenv`) en lugar de estar hardcodeadas.
- Se agregó un `.gitignore` para excluir `.env`, `downloaded_pdfs/` y archivos de caché, por si el proyecto se versiona con git más adelante.
- Todo esto se organizó en una carpeta dedicada: **`rag_pipeline/`**, conteniendo `rag_pipeline.py`, `.env`, `.gitignore` y `downloaded_pdfs/`.

**Por qué:** separar credenciales del código es una buena práctica de seguridad, y encapsular el flujo en su propia carpeta facilita reutilizarlo o moverlo sin arrastrar el resto del proyecto del curso.

## Recursos de Azure creados (estado actual)

| Recurso | Nombre | Costo |
|---|---|---|
| Resource Group | `rg-luisalfonso634-9203` | — |
| Storage Account | `stragdocs9203` | Standard_LRS (bajo costo por uso) |
| Contenedor Blob | `rag-documents` | incluido en Storage |
| Azure AI Search | `srch-ragdocs-9203` | Free (F0), $0/mes |
| Índice de búsqueda | `rag-documents-index` | incluido en Search Free |
| Automation Account | `aa-budget-guard` | creado pero sin uso activo (pendiente de decisión del usuario) |

## Archivos del proyecto

- `clase7.ipynb` — notebook con todo el flujo (instalación, vectorización, consulta RAG con LLM).
- `rag_pipeline/rag_pipeline.py` — mismo flujo como script Python ejecutable.
- `rag_pipeline/.env` — credenciales (no debe compartirse ni subirse a git).
- `curso-rag-main/` — material del curso, incluye los PDFs originales (`data_sample/`) y el notebook de referencia (`clase9/libro1.ipynb`).
