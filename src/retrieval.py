# ////////////////////////////////////////////////////////////////// #
# ////////////////////////// RETRIEVAL ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from src.indexing import SearchIndex
from src.logging_config import steps_logger
from src.models import Chunk, MinimalSource


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
        self.validate_query(
            query=query,
            k=k,
        )

        steps_logger.info(
            "[Retriever] Retrieving top-%d chunks for query: %s",
            k,
            query,
        )

        return self.search_index.search(
            query=query,
            k=k,
        )

    def retrieve_sources(
        self,
        query: str,
        k: int = 10,
    ) -> list[MinimalSource]:
        """
        Retrieves the top-k results as minimal sources.
        """
        chunks = self.retrieve(
            query=query,
            k=k,
        )

        return [
            self.chunk_to_source(chunk)
            for chunk in chunks
        ]

    def chunk_to_source(
        self,
        chunk: Chunk,
    ) -> MinimalSource:
        """
        Converts a chunk into a minimal source.
        """
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )

    def validate_query(
        self,
        query: str,
        k: int,
    ) -> None:
        """
        Validates the query and the number of requested results.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be greater than zero")
