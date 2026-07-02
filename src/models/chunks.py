from typing import Any

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    content: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(gt=0)
    chunk_type: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_indexes(self) -> "Chunk":
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                "last_character_index must be greater than "
                "first_character_index"
            )

        expected_length = (
            self.last_character_index
            - self.first_character_index
        )

        if len(self.content) != expected_length:
            raise ValueError(
                "The chunk content length does not match its indexes"
            )

        return self
