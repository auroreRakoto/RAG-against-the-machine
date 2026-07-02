from abc import ABC, abstractmethod
from pydantic import BaseModel


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// CHUNKING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class Chunk(BaseModel):
    """
    represents a chunk of text extracted from a file, along with its metadata.
    """
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunker(ABC):
    """
    represents a chunker that can split text into chunks.
    """
    @abstractmethod
    def chunk(self, text: str, file_path: str) -> list[Chunk]:
        pass


class PythonChunker(Chunker):
    """
    represents a chunker that can split Python code into chunks.
    """
    pass


class TextChunker(Chunker):
    """
    represents a chunker that can split Markdown or text into chunks.
    """
    pass


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// SEARCHING //////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class SearchIndex(ABC):
    """
    represents a search index.
    """
    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> list[Chunk]:
        pass


class BM25Index(SearchIndex):
    """
    represents a search index using the BM25 ranking algorithm.
    """
    pass


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// OTHERS //////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class RepositoryLoader:
    pass


class Retriever:
    pass


class DatasetLoader:
    pass


class RecallEvaluator:
    pass


class ContextBuilder:
    pass


class AnswerGenerator:
    pass


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// MODELS /////////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
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


class SearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class SearchResultsWithAnswers(SearchResults):
    search_results: list[MinimalAnswer]


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// OTHER 2 ////////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class FileReader:
    pass


class ChunkStorage:
    pass


class IndexStorage:
    pass


class PromptBuilder:
    pass


class LanguageModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class QwenLanguageModel(LanguageModel):
    pass


class DatasetWriter:
    pass


class CLI:
    pass
