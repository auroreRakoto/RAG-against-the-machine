
# ////////////////////////////////////////////////////////////////// #
# ///////////////////////////// MODELS ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from typing import Generic, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class Chunk(BaseModel):
    """
    represents a chunk of text extracted from a file, along with its metadata.
    """
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class SearchResults(BaseModel, Generic[T]):
    search_results: list[T]
    k: int


class SearchResultsWithAnswers(SearchResults[MinimalAnswer]):
    pass
