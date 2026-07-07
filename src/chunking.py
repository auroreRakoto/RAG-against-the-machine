# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// CHUNKING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod

from src.logging_config import chunks_loaded_logger, chunks_logger
from src.models import Chunk

class Chunker(ABC):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(
        self,
        text: str,
        file_path: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0

        while start < len(text):
            max_end = min(
                start + self.max_chunk_size,
                len(text),
            )

            end = self.find_split_index(
                text=text,
                start=start,
                max_end=max_end,
            )
            chunk = Chunk(
                text=text[start:end],
                file_path=file_path,
                first_character_index=start,
                last_character_index=end,
            )

            chunks_logger.info(
                "File=%s | start=%d | end=%d\n%s\n",
                file_path,
                start,
                end,
                chunk.text,
            )

            chunks.append(chunk)

            start = end
        chunks_loaded_logger.info(
            "Created %d chunks for file: %s",
            len(chunks),
            file_path,
        )
        return chunks

    @abstractmethod
    def find_split_index(
        self,
        text: str,
        start: int,
        max_end: int,
    ) -> int:
        pass


class PythonChunker(Chunker):
    def find_split_index(
        self,
        text: str,
        start: int,
        max_end: int,
    ) -> int:
        """
        Finds a safe line boundary for Python code.
        """
        if max_end >= len(text):
            return len(text)

        split_index = text.rfind("\n", start, max_end)

        if split_index <= start:
            return max_end

        return split_index + 1


class TextChunker(Chunker):
    def find_split_index(
        self,
        text: str,
        start: int,
        max_end: int,
    ) -> int:
        """
        Finds the index to split the text into chunks.
        """
        if max_end >= len(text):
            return len(text)

        split_index = text.rfind("\n", start, max_end)

        if split_index <= start:
            split_index = text.rfind(" ", start, max_end)

        if split_index <= start:
            return max_end

        return split_index + 1


class ChunkStorage:
    pass

