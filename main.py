# ////////////////////////////////////////////////////////////////// #
# ///////////////////////////// MODELS ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from pydantic import BaseModel

from classes import MinimalAnswer


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


class SearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class SearchResultsWithAnswers(SearchResults):
    search_results: list[MinimalAnswer]


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// INGESTION //////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class FileReader:
    pass


class RepositoryLoader:
    pass


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// CHUNKING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod


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


class ChunkStorage:
    pass


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// INDEXING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod


class SearchIndex(ABC):
    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None:
        """
        Builds the search index from chunks.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> list[Chunk]:
        """
        Returns the top-k chunks matching the query.
        """
        pass

    @abstractmethod
    def is_built(self) -> bool:
        """
        Returns whether the index is ready to be searched.
        """
        pass


class BM25Index(SearchIndex):
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.tokenized_chunks: list[list[str]] = []
        self.bm25: object | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """
        Tokenizes the chunks and builds the BM25 index.
        """
        pass

    def search(self, query: str, k: int) -> list[Chunk]:
        """
        Ranks chunks with BM25 and returns the top-k results.
        """
        print(f"[BM25Index] Searching for query: {query} with top-{k} results")

    def tokenize(self, text: str) -> list[str]:
        """
        Converts text into tokens used by BM25.
        """
        pass

    def is_built(self) -> bool:
        """
        Returns whether the BM25 index has been built.
        """
        pass


class IndexStorage:
    def save(
        self,
        search_index: SearchIndex,
        directory_path: str,
    ) -> None:
        """
        Saves the index data to a directory.
        """
        pass

    def load(
        self,
        directory_path: str,
    ) -> SearchIndex:
        print(
            f"[IndexStorage] Loading index from: "
            f"{directory_path}"
        )

        return BM25Index()

    def exists(self, directory_path: str) -> bool:
        """
        Checks whether a saved index exists.
        """
        pass

    def create_directory(self, directory_path: str) -> None:
        """
        Creates the storage directory if necessary.
        """
        pass


# ////////////////////////////////////////////////////////////////// #
# ////////////////////////// RETRIEVAL ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class Retriever:
    def __init__(self, search_index: SearchIndex) -> None:
        self.search_index = search_index
        print(f"[Retriever] Initialized with search index: {type(search_index).__name__}")

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """
        Retrieves the top-k most relevant chunks for a query.
        """
        print(f"[Retriever] Retrieving top-{k} chunks for query: {query}")
        return self.search_index.search(query=query, k=k)

    def retrieve_sources(
        self,
        query: str,
        k: int = 10,
    ) -> list[MinimalSource]:
        """
        Retrieves the top-k results as minimal sources.
        """
        pass

    def chunk_to_source(
        self,
        chunk: Chunk,
    ) -> MinimalSource:
        """
        Converts a chunk into a minimal source.
        """
        pass

    def validate_query(
        self,
        query: str,
        k: int,
    ) -> None:
        """
        Validates the query and the number of requested results.
        """
        pass


# ////////////////////////////////////////////////////////////////// #
# ////////////////////////// GENERATION //////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class ContextBuilder:
    pass


class PromptBuilder:
    pass


class LanguageModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class QwenLanguageModel(LanguageModel):
    pass


class AnswerGenerator:
    pass


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// DATASETS ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class DatasetLoader:
    pass


class DatasetWriter:
    pass


# ////////////////////////////////////////////////////////////////// #
# ////////////////////////// EVALUATION //////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class RecallEvaluator:
    MINIMUM_OVERLAP_RATIO = 0.05

    def evaluate(
        self,
        expected_dataset: RagDataset,
        search_results: SearchResults,
        k_values: list[int],
    ) -> dict[int, float]:
        """
        Evaluates the recall@k for the given expected dataset
            and search results.
        """
        pass

    def evaluate_question(
        self,
        expected_question: AnsweredQuestion,
        retrieved_result: MinimalSearchResults,
        k: int,
    ) -> float:
        """
        Evaluates the recall@k for a single question."""
        pass

    def is_source_found(
        self,
        expected_source: MinimalSource,
        retrieved_sources: list[MinimalSource],
    ) -> bool:
        """
        Checks if the expected source is found in the retrieved sources.
        """
        pass

    def calculate_overlap_ratio(
        self,
        first_source: MinimalSource,
        second_source: MinimalSource,
    ) -> float:
        """
        Calculates the overlap ratio between two sources based on
            their character indices.
        """
        pass

    def count_found_sources(
        self,
        expected_sources: list[MinimalSource],
        retrieved_sources: list[MinimalSource],
    ) -> int:
        """
        Counts the number of expected sources found in the retrieved sources.
        """
        pass


# ////////////////////////////////////////////////////////////////// #
# ////////////////////////////// CLI /////////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class CLI:
    def index(
        self,
        repository_path: str,
        max_chunk_size: int = 2000,
    ) -> None:
        print(
            f"[CLI] Indexing repository: {repository_path} "
            f"with max chunk size: {max_chunk_size}"
        )

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> None:
        print(f"[CLI] Search requested: {query}")

        print("[CLI] Loads the saved search index")
        # create an instance of IndexStorage and load the index
        index_storage = IndexStorage()
        search_index = index_storage.load(directory_path="data/index")
        print("[CLI] Creates a Retriever with that index")
        # create an instance of Retriever with the loaded search index
        retriever = Retriever(search_index=search_index)
        print("[CLI] Calls Retriever.retrieve()")
        # retrieve the top-k chunks for the query
        retrieved_chunks = retriever.retrieve(query=query, k=k)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        pass

    def answer(
        self,
        question: str,
        k: int = 10,
    ) -> None:
        print(f"[CLI] Answer requested: {question}")

    def answer_dataset(
        self,
        search_results_path: str,
        save_directory: str,
    ) -> None:
        pass

    def evaluate(
        self,
        answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> None:
        pass


def main():
    """
    Entry point for the CLI.
    """
    repository_path = "test_repository.txt"
    test_query = "What is the purpose of the Chunk class?"
    cli = CLI()

    print("\n=== INDEXING ===")
    cli.index(
        repository_path=repository_path,
        max_chunk_size=2000,
    )

    print("\n=== SEARCHING ===")
    cli.search(query=test_query, k=10)

    print("\n=== ANSWERING ===")
    cli.answer(
        question=test_query,
        k=10,
    )


main()
