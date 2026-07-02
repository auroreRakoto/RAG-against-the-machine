from src.models.chunks import Chunk
from src.retrieval.bm25_index import BM25Index


def build_index(chunks: list[Chunk]) -> BM25Index:
    return BM25Index(chunks)
