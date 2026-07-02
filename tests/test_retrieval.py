from src.models.chunks import Chunk
from src.services.indexing import build_index
from src.services.searching import search


def test_search_returns_the_most_relevant_chunk() -> None:
    chunks = [
        Chunk(
            content="The quick brown fox jumps over the lazy dog",
            file_path="doc1.txt",
            first_character_index=0,
            last_character_index=43,
            chunk_type="text",
        ),
        Chunk(
            content="A cat sleeps near the window",
            file_path="doc2.txt",
            first_character_index=0,
            last_character_index=28,
            chunk_type="text",
        ),
        Chunk(
            content="Cats are playful pets",
            file_path="doc3.txt",
            first_character_index=0,
            last_character_index=21,
            chunk_type="text",
        ),
    ]

    index = build_index(chunks)
    results = search("cat", index, top_k=2)

    assert len(results) == 2
    assert results[0][0].file_path == "doc2.txt"
    assert results[0][0].content == "A cat sleeps near the window"
