from src.models.chunks import Chunk
from src.retrieval.bm25_index import BM25Index


def search(
        query: str,
        index: BM25Index,
        top_k: int = 5
) -> list[tuple[Chunk, float]]:
    results = index.score(query)
    return results[:top_k]
