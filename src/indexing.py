# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// INDEXING ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod
from pathlib import Path
import re
import pickle

from rank_bm25 import BM25Okapi

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

        self.chunks = chunks

        self.tokenized_chunks = []

        for chunk in chunks:
            searchable_text = f"{chunk.file_path} {chunk.text}"
            tokens = self.tokenize(searchable_text)
            self.tokenized_chunks.append(tokens)

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

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        selected_indices = ranked_indices[:min(k, len(ranked_indices))]

        return [
            self.chunks[index]
            for index in selected_indices
        ]

    def tokenize(self, text: str) -> list[str]:
        """
        Converts text into normalized tokens used by BM25.
        """
        text = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1 \2",
            text,
        )
        text = text.replace("_", " ")

        return re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

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
