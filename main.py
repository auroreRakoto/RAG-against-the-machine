import logging
from pathlib import Path


def create_file_logger(
    name: str,
    file_path: str,
) -> logging.Logger:
    """
    Creates a logger that writes messages to a specific file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    path = Path(file_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handler = logging.FileHandler(
        path,
        mode="w",
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(handler)

    return logger


files_logger = create_file_logger(
    "files",
    "data/logs/files_read.log",
)

chunks_logger = create_file_logger(
    "chunks",
    "data/logs/chunks.log",
)

chunks_loaded_logger = create_file_logger(
    "chunks_loaded",
    "data/logs/chunks_loaded.log",
)

steps_logger = create_file_logger(
    "steps",
    "data/logs/steps.log",
)


# ////////////////////////////////////////////////////////////////// #
# ///////////////////////////// MODELS ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from pydantic import BaseModel


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
from pathlib import Path


class FileReader:
    def read(self, file_path: str) -> str:
        """
        Reads and returns the content of a text file.
        """
        path = Path(file_path)

        # print(f"[FileReader] Reading file: {path}")

        with path.open(
            mode="r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            return file.read()

class RepositoryLoader:
    def __init__(self, file_reader: FileReader) -> None:
        self.file_reader = file_reader

    def load(
        self,
        repository_path: str,
    ) -> dict[str, str]:
        """
        Loads Python and Markdown files from a repository.
        """
        repository = Path(repository_path)

        if not repository.exists():
            raise FileNotFoundError(
                f"Repository not found: {repository_path}"
            )

        files: dict[str, str] = {}

        for file_path in repository.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix in {".py", ".md"}
            ):
                files[str(file_path)] = self.file_reader.read(
                    str(file_path)
                )

                files_logger.info(
                    "Read file: %s",
                    file_path,
                )

        steps_logger.info(
            "[RepositoryLoader] Loaded %d files",
            len(files)
        )

        return files


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// CHUNKING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod


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


# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// INDEXING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod
from rank_bm25 import BM25Okapi
import pickle



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
        if not chunks:
            raise ValueError("Cannot build an index without chunks")

        steps_logger.info(
            "[BM25Index] Starting BM25 index construction"
        )

        self.chunks = chunks

        self.tokenized_chunks = [
            self.tokenize(chunk.text)
            for chunk in chunks
        ]

        output_path = Path("data/output/BM25_indexes.txt")
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            for index, tokens in enumerate(
                self.tokenized_chunks
            ):
                file.write(f"--- CHUNK {index} ---\n")
                file.write(" ".join(tokens))
                file.write("\n\n")

        self.bm25 = BM25Okapi(self.tokenized_chunks)

        steps_logger.info(
            "[BM25Index] BM25 index construction completed"
        )

    def search(
        self,
        query: str,
        k: int,
    ) -> list[Chunk]:
        """
        Ranks chunks with BM25 and returns the top-k results.
        """
        if not self.is_built():
            raise RuntimeError("BM25 index has not been built")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be greater than zero")

        query_tokens = self.tokenize(query)

        if not isinstance(self.bm25, BM25Okapi):
            raise RuntimeError("Invalid BM25 index")

        steps_logger.info(
            "[BM25Index] Searching query=%r with k=%d",
            query,
            k,
        )

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        selected_indices = ranked_indices[:min(k, len(ranked_indices))]

        steps_logger.info(
            "[BM25Index] Retrieved %d chunks",
            len(selected_indices),
        )

        return [
            self.chunks[index]
            for index in selected_indices
        ]

    def tokenize(self, text: str) -> list[str]:
        """
        Converts text into lowercase tokens used by BM25.
        """
        return text.lower().split()

    def is_built(self) -> bool:
        """
        Returns whether the BM25 index has been built.
        """
        return self.bm25 is not None


class IndexStorage:
    def save(
        self,
        search_index: SearchIndex,
        directory_path: str,
    ) -> None:
        """
        Saves the search index to disk.
        """
        directory = Path(directory_path)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path = directory / "bm25_index.pkl"

        with file_path.open("wb") as file:
            pickle.dump(search_index, file)

        steps_logger.info(
            "[IndexStorage] Index saved to: %s",
            file_path,
        )

    def load(
        self,
        directory_path: str,
    ) -> SearchIndex:
        """
        Loads the search index from disk.
        """
        file_path = (
            Path(directory_path)
            / "bm25_index.pkl"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Index not found: {file_path}"
            )

        with file_path.open("rb") as file:
            search_index = pickle.load(file)

        if not isinstance(search_index, SearchIndex):
            raise TypeError(
                "Loaded object is not a SearchIndex"
            )

        steps_logger.info(
            "[IndexStorage] Index loaded from: %s",
            file_path,
        )

        return search_index

    def exists(self, directory_path: str) -> bool:
        """
        Checks whether a saved index exists.
        """
        file_path = (
            Path(directory_path)
            / "bm25_index.pkl"
        )

        return file_path.exists()

    def create_directory(
        self,
        directory_path: str,
    ) -> None:
        """
        Creates the storage directory if necessary.
        """
        Path(directory_path).mkdir(
            parents=True,
            exist_ok=True,
        )


# ////////////////////////////////////////////////////////////////// #
# ////////////////////////// RETRIEVAL ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
class Retriever:
    def __init__(self, search_index: SearchIndex) -> None:
        self.search_index = search_index

        steps_logger.info(
            "[Retriever] Initialized with search index: %s",
            type(search_index).__name__,
        )

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """
        Retrieves the top-k most relevant chunks for a query.
        """
        steps_logger.info(
            "[Retriever] Retrieving top-%d chunks for query: %s",
            k,
            query,
        )
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
        max_chunk_size: int = 2000,
        repository_path: str = "data/raw/vllm-0.10.1",
    ) -> None:
        steps_logger.info(
            "[CLI] Starting repository indexing: %s",
            repository_path,
        )

        reader = FileReader()
        loader = RepositoryLoader(file_reader=reader)

        files = loader.load(repository_path)

        steps_logger.info(
            "[CLI] Repository loading completed"
        )

        text_chunker = TextChunker(
            max_chunk_size=max_chunk_size
        )

        python_chunker = PythonChunker(
            max_chunk_size=max_chunk_size
        )

        all_chunks: list[Chunk] = []

        for file_path, text in files.items():
            if file_path.endswith(".py"):
                file_chunks = python_chunker.chunk(
                    text=text,
                    file_path=file_path,
                )
            else:
                file_chunks = text_chunker.chunk(
                    text=text,
                    file_path=file_path,
                )

            all_chunks.extend(file_chunks)

        search_index = BM25Index()
        search_index.build(all_chunks)

        index_storage = IndexStorage()
        index_storage.save(
            search_index=search_index,
            directory_path="data/index",
        )

        steps_logger.info(f"[CLI] Created {len(all_chunks)} chunks")

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> None:
        steps_logger.info(
            "[CLI] Search requested: %s",
            query,
        )

        steps_logger.info(
            "[CLI] Loading saved search index"
        )

        index_storage = IndexStorage()
        search_index = index_storage.load(
            directory_path="data/index"
        )

        steps_logger.info(
            "[CLI] Creating Retriever"
        )

        retriever = Retriever(
            search_index=search_index
        )

        steps_logger.info(
            "[CLI] Calling Retriever.retrieve()"
        )

        retrieved_chunks = retriever.retrieve(
            query=query,
            k=k,
        )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        steps_logger.info(
            "[CLI] Searching dataset from: %s with top-%d results",
            dataset_path,
            k
        )

    def answer(
        self,
        question: str,
        k: int = 10,
    ) -> None:
        steps_logger.info(
            "[CLI] Answer requested: %s",
            question
        )

    def answer_dataset(
        self,
        search_results_path: str,
        save_directory: str,
    ) -> None:
        steps_logger.info(
            "[CLI] Answering dataset from: %s",
            search_results_path
        )

    def evaluate(
        self,
        answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> None:
        steps_logger.info(
            "[CLI] Evaluating answers from: %s against dataset: %s",
            answer_path,
            dataset_path
        )


import fire

def main():
    """
    Entry point for the CLI.
    """
    fire.Fire(CLI, name="RAG CLI")

if  __name__ == "__main__":
    main()
