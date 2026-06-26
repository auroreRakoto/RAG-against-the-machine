from src.chunking.base import BaseChunker
from src.models.chunks import Chunk


class TextChunker(BaseChunker):
    def __init__(self) -> None:
        self.separators = (
            "\n\n",
            "\n",
            ". ",
            " ",
        )

    def chunk(
        self,
        text: str,
        file_path: str,
        max_chunk_size: int,
    ) -> list[Chunk]:
        self.validate_inputs(
            text=text,
            file_path=file_path,
            max_chunk_size=max_chunk_size,
        )

        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0

        while start < len(text):
            maximum_end = min(
                start + max_chunk_size,
                len(text),
            )

            end = self._find_split_position(
                text=text,
                start=start,
                maximum_end=maximum_end,
            )

            chunk = self.create_chunk(
                text=text,
                file_path=file_path,
                start=start,
                end=end,
                chunk_type="text",
            )

            chunks.append(chunk)
            start = end

        return chunks

    def _find_split_position(
        self,
        text: str,
        start: int,
        maximum_end: int,
    ) -> int:
        if maximum_end >= len(text):
            return len(text)

        for separator in self.separators:
            position = text.rfind(
                separator,
                start,
                maximum_end,
            )

            if position == -1:
                continue

            end = position + len(separator)

            if end > start:
                return end

        return maximum_end