# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// INDEXING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod
from pathlib import Path
import pickle

from rank_bm25 import BM25Okapi

from src.logging_config import steps_logger
from src.models import Chunk


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

