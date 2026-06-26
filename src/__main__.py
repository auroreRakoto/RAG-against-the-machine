from pathlib import Path

from src.chunking.text_chunker import TextChunker


def test_chunking(
    file_path: str,
    max_chunk_size: int = 500,
) -> None:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    chunker = TextChunker()

    chunks = chunker.chunk(
        text=text,
        file_path=str(path),
        max_chunk_size=max_chunk_size,
    )

    print(f"File: {path}")
    print(f"Text length: {len(text)}")
    print(f"Chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print()
        print(
            f"Chunk {index}: "
            f"[{chunk.first_character_index}:"
            f"{chunk.last_character_index}]"
        )
        print(f"Size: {len(chunk.content)}")
        print(repr(chunk.content[:100]))

        original_content = text[
            chunk.first_character_index:
            chunk.last_character_index
        ]

        assert chunk.content == original_content
        assert len(chunk.content) <= max_chunk_size


# if __name__ == "__main__":
#     test_chunking(
#         file_path="en.subject.pdf",
#         max_chunk_size=500,
#     )

if __name__ == "__main__":
    test_chunking(
        file_path="src/services/searching.py",
        max_chunk_size=500,
    )