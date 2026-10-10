"""
Pipeline RAG completo: descarga PDFs de Blob Storage, los vectoriza localmente
(sentence-transformers), los sube a Azure AI Search y responde una pregunta
usando un LLM (OpenAI gpt-4o-mini).

Equivalente en script al notebook clase7.ipynb.
"""
import os

import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()

STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT"]
STORAGE_KEY = os.environ["STORAGE_KEY"]
CONTAINER = os.environ["CONTAINER"]

SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["SEARCH_KEY"]
INDEX_NAME = os.environ["INDEX_NAME"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

LOCAL_DIR = "downloaded_pdfs"
BATCH_SIZE = 100


def chunk_text(text, chunk_size=800, overlap=150):
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
            print("Descargado:", blob.name)
    print(f"Total PDFs descargados: {len(local_paths)}")
    return local_paths


def extract_and_chunk(local_paths):
    documents = []
    for blob_name, path in local_paths:
        reader = PdfReader(path)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for i, chunk in enumerate(chunk_text(full_text)):
            documents.append(
                {
                    "id": f"{blob_name.replace('/', '_').replace('.pdf', '')}_{i}",
                    "blob_name": blob_name,
                    "chunk_index": i,
                    "content": chunk,
                }
            )
    print(f"Total de chunks generados: {len(documents)}")
    return documents


def embed_documents(model, documents):
    contents = [doc["content"] for doc in documents]
    embeddings = model.encode(contents, show_progress_bar=True, normalize_embeddings=True)
    for doc, vector in zip(documents, embeddings):
        doc["content_vector"] = vector.tolist()
    return documents


def create_index(embedding_dim):
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))
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
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="default-vector-profile", algorithm_configuration_name="default-hnsw"
            )
        ],
    )
    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"Índice '{INDEX_NAME}' creado/actualizado.")


def upload_documents(search_client, documents):
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        result = search_client.upload_documents(documents=batch)
        failed = [r for r in result if not r.succeeded]
        print(f"Lote {i // BATCH_SIZE + 1}: {len(batch) - len(failed)}/{len(batch)} subidos correctamente")


def ask(model, search_client, openai_client, question, k=5):
    question_vector = model.encode(question, normalize_embeddings=True).tolist()
    results = list(
        search_client.search(
            search_text=None,
            vector_queries=[
                VectorizedQuery(vector=question_vector, k_nearest_neighbors=k, fields="content_vector")
            ],
            select=["id", "blob_name", "chunk_index", "content"],
        )
    )

    context = "\n\n".join(r["content"] for r in results)
    system_message = (
        "You are an AI Assistant.\n"
        "Be brief in your answers. Answer ONLY with the facts listed in the retrieved text.\n\n"
        f"Context:\n{context}"
    )
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content, results


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()

    local_paths = download_pdfs()
    documents = extract_and_chunk(local_paths)
    documents = embed_documents(model, documents)

    create_index(embedding_dim)

    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    upload_documents(search_client, documents)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    question = "What is included in my Northwind Health Plus plan that is not in standard?"
    answer, results = ask(model, search_client, openai_client, question)

    print("\nPregunta:", question)
    print("Respuesta:", answer)


if __name__ == "__main__":
    main()
