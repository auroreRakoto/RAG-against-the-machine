# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// CHUNKING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod

from src.models import Chunk


class Chunker(ABC):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than zero")

        if max_chunk_size > 2000:
            raise ValueError("max_chunk_size cannot be greater than 2000")

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

            chunks.append(chunk)

            start = end
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
