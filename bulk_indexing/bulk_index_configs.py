"""
Toma los documentos estructurados (PDFs en el contenedor rag-documents), los
parte en chunks y los sube de forma masiva a Azure AI Search, repitiendo el
proceso en varios indices con distintas configuraciones de chunking y
distinto algoritmo de busqueda vectorial, usando siempre la misma
informacion fuente.

Reutiliza el servicio Azure AI Search Free (F0) ya existente (costo $0).
El tier Free permite hasta 3 indices; aqui se usan los 3:

  1. rag-documents-index      -> chunks de 800/overlap 150, algoritmo HNSW
                                  (el indice "base" que ya existia)
  2. rag-index-chunks-small   -> chunks de 300/overlap 50,  algoritmo HNSW
                                  (fragmentos chicos, mas precision local)
  3. rag-index-exhaustive     -> chunks de 1500/overlap 200, ExhaustiveKNN
                                  (fragmentos grandes + busqueda exacta,
                                   sin aproximacion, para comparar contra HNSW)
"""
import os
from dataclasses import dataclass

import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.storage.blob import BlobServiceClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()

STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT"]
STORAGE_KEY = os.environ["STORAGE_KEY"]
CONTAINER = os.environ["CONTAINER"]

SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["SEARCH_KEY"]

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

LOCAL_DIR = "downloaded_pdfs"
BATCH_SIZE = 100


@dataclass
class IndexConfig:
    index_name: str
    chunk_size: int
    overlap: int
    algorithm: str  # "hnsw" o "exhaustive"


CONFIGS = [
    IndexConfig("rag-documents-index", chunk_size=800, overlap=150, algorithm="hnsw"),
    IndexConfig("rag-index-chunks-small", chunk_size=300, overlap=50, algorithm="hnsw"),
    IndexConfig("rag-index-exhaustive", chunk_size=1500, overlap=200, algorithm="exhaustive"),
]


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def download_pdfs():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    blob_service = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=STORAGE_KEY,
    )
    container_client = blob_service.get_container_client(CONTAINER)

    local_paths = []
    for blob in container_client.list_blobs():
        if blob.name.lower().endswith(".pdf"):
            dest_path = os.path.join(LOCAL_DIR, blob.name.replace("/", "_"))
            with open(dest_path, "wb") as f:
                f.write(container_client.download_blob(blob.name).readall())
            local_paths.append((blob.name, dest_path))
    print(f"PDFs descargados: {len(local_paths)}")
    return local_paths


def extract_full_texts(local_paths):
    """Extrae el texto completo una sola vez por PDF (se reutiliza para los 3 chunkings)."""
    full_texts = {}
    for blob_name, path in local_paths:
        reader = PdfReader(path)
        full_texts[blob_name] = "\n".join(page.extract_text() or "" for page in reader.pages)
    return full_texts


def build_documents(full_texts, chunk_size, overlap):
    documents = []
    for blob_name, text in full_texts.items():
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            documents.append(
                {
                    "id": f"{blob_name.replace('/', '_').replace('.pdf', '')}_{i}",
                    "blob_name": blob_name,
                    "chunk_index": i,
                    "content": chunk,
                }
            )
    return documents


def embed_documents(model, documents):
    contents = [doc["content"] for doc in documents]
    embeddings = model.encode(contents, show_progress_bar=True, normalize_embeddings=True)
    for doc, vector in zip(documents, embeddings):
        doc["content_vector"] = vector.tolist()
    return documents


def create_index(index_client, config, embedding_dim):
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="blob_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name="default-vector-profile",
        ),
    ]

    if config.algorithm == "hnsw":
        algo = HnswAlgorithmConfiguration(name="algo-config")
    elif config.algorithm == "exhaustive":
        algo = ExhaustiveKnnAlgorithmConfiguration(name="algo-config")
    else:
        raise ValueError(f"Algoritmo desconocido: {config.algorithm}")

    vector_search = VectorSearch(
        algorithms=[algo],
        profiles=[
            VectorSearchProfile(name="default-vector-profile", algorithm_configuration_name="algo-config")
        ],
    )

    index = SearchIndex(name=config.index_name, fields=fields, vector_search=vector_search)
    try:
        index_client.delete_index(config.index_name)
    except Exception:
        pass
    index_client.create_index(index)


def upload_documents(search_client, documents):
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        result = search_client.upload_documents(documents=batch)
        failed = [r for r in result if not r.succeeded]
        if failed:
            print(f"  Lote {i // BATCH_SIZE + 1}: {len(batch) - len(failed)}/{len(batch)} OK ({len(failed)} fallaron)")


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))

    local_paths = download_pdfs()
    full_texts = extract_full_texts(local_paths)

    print(f"\n{'=' * 70}")
    print(f"Procesando {len(CONFIGS)} configuraciones de indice")
    print("=" * 70)

    for config in CONFIGS:
        print(f"\n--- {config.index_name} "
              f"(chunk_size={config.chunk_size}, overlap={config.overlap}, algoritmo={config.algorithm}) ---")

        documents = build_documents(full_texts, config.chunk_size, config.overlap)
        print(f"  Chunks generados: {len(documents)}")

        documents = embed_documents(model, documents)

        create_index(index_client, config, embedding_dim)
        print(f"  Indice '{config.index_name}' creado/actualizado.")

        search_client = SearchClient(SEARCH_ENDPOINT, config.index_name, AzureKeyCredential(SEARCH_KEY))
        upload_documents(search_client, documents)
        print(f"  {len(documents)} documentos subidos a '{config.index_name}'.")

    print(f"\n{'=' * 70}")
    print("RESUMEN")
    print("=" * 70)
    for config in CONFIGS:
        n_chunks = len(build_documents(full_texts, config.chunk_size, config.overlap))
        print(f"  {config.index_name}: {n_chunks} chunks, algoritmo={config.algorithm}")


if __name__ == "__main__":
    main()
