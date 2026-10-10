"""
Compara tres estrategias de búsqueda en Azure AI Search sobre el mismo índice
(rag-documents-index, ya poblado por rag_pipeline.py):

1. Búsqueda tradicional por texto (BM25 sobre el campo "content").
2. Búsqueda vectorial (similitud semántica sobre "content_vector").
3. Búsqueda híbrida (texto + vector combinados con Reciprocal Rank Fusion,
   que es lo que hace Azure AI Search automáticamente cuando se pasan ambos).

No crea ningún recurso nuevo de Azure: reutiliza el servicio Free (F0) y el
índice ya existentes.
"""
import os

import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()

SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["SEARCH_KEY"]
INDEX_NAME = os.environ["INDEX_NAME"]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

TOP_K = 5


def search_text_only(search_client, query):
    """1. Búsqueda tradicional por texto (keyword / BM25)."""
    results = search_client.search(
        search_text=query,
        top=TOP_K,
        select=["id", "blob_name", "content"],
    )
    return list(results)


def search_vector_only(search_client, model, query):
    """2. Búsqueda vectorial pura (similitud semántica)."""
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = search_client.search(
        search_text=None,
        vector_queries=[
            VectorizedQuery(vector=query_vector, k_nearest_neighbors=TOP_K, fields="content_vector")
        ],
        top=TOP_K,
        select=["id", "blob_name", "content"],
    )
    return list(results)


def search_hybrid(search_client, model, query):
    """3. Búsqueda híbrida: texto + vector combinados (RRF automático de Azure AI Search)."""
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = search_client.search(
        search_text=query,
        vector_queries=[
            VectorizedQuery(vector=query_vector, k_nearest_neighbors=TOP_K, fields="content_vector")
        ],
        top=TOP_K,
        select=["id", "blob_name", "content"],
    )
    return list(results)


def print_results(title, results):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)
    for rank, r in enumerate(results, start=1):
        score = r.get("@search.score")
        reranker_score = r.get("@search.reranker_score")
        score_str = f"score={score:.4f}" if score is not None else ""
        print(f"{rank}. [{r['id']}] {score_str}  ({r['blob_name']})")
        print(f"   {r['content'][:160].strip()}...")


def compare(query):
    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    model = SentenceTransformer(EMBEDDING_MODEL)

    text_results = search_text_only(search_client, query)
    vector_results = search_vector_only(search_client, model, query)
    hybrid_results = search_hybrid(search_client, model, query)

    print(f"\nPregunta: {query}")
    print_results("1) BÚSQUEDA TRADICIONAL POR TEXTO (keyword / BM25)", text_results)
    print_results("2) BÚSQUEDA VECTORIAL (semántica)", vector_results)
    print_results("3) BÚSQUEDA HÍBRIDA (texto + vector)", hybrid_results)

    # Comparación rápida de qué tan distintos son los top-5 de cada método
    text_ids = {r["id"] for r in text_results}
    vector_ids = {r["id"] for r in vector_results}
    hybrid_ids = {r["id"] for r in hybrid_results}

    print(f"\n{'=' * 70}")
    print("SOLAPAMIENTO ENTRE ESTRATEGIAS (IDs en común dentro del top-5)")
    print("=" * 70)
    print(f"Texto y Vector:  {len(text_ids & vector_ids)}/{TOP_K}")
    print(f"Texto y Híbrido: {len(text_ids & hybrid_ids)}/{TOP_K}")
    print(f"Vector y Híbrido: {len(vector_ids & hybrid_ids)}/{TOP_K}")


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What is included in my Northwind Health Plus plan that is not in standard?"
    )
    compare(query)
