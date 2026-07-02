from abc import ABC, abstractmethod
from typing import Any

from src.models.chunks import Chunk


class BaseChunker(ABC):
    MAX_ALLOWED_CHUNK_SIZE = 2000

    @abstractmethod
    def chunk(
        self,
        text: str,
        file_path: str,
        max_chunk_size: int,
    ) -> list[Chunk]:
        raise NotImplementedError

    def validate_inputs(
        self,
        text: str,
        file_path: str,
        max_chunk_size: int,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")

        if not file_path.strip():
            raise ValueError("file_path cannot be empty")

        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than zero"
            )

        if max_chunk_size > self.MAX_ALLOWED_CHUNK_SIZE:
            raise ValueError(
                "max_chunk_size cannot be greater than 2000"
            )

    def create_chunk(
        self,
        text: str,
        file_path: str,
        start: int,
        end: int,
        chunk_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        if start < 0:
            raise ValueError("start cannot be negative")

        if end <= start:
            raise ValueError("end must be greater than start")

        if end > len(text):
            raise ValueError("end cannot exceed the text length")

        return Chunk(
            content=text[start:end],
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
            chunk_type=chunk_type,
            metadata=metadata or {},
        )
