# ////////////////////////////////////////////////////////////////// #
# ///////////////////////////// MODELS ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
import uuid

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represent a chunk of text extracted from a file."""

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSource(BaseModel):
    """Represent the location of a retrieved source."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question without an expected answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with expected sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieved sources and a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represent the structured output of search operations."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Represent the structured output of answer operations."""

    search_results: list[MinimalAnswer]  # type: ignore[assignment]
