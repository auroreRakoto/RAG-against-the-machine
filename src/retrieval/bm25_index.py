from collections import Counter
from math import log

from src.models.chunks import Chunk
from src.retrieval.tokenizer import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0

        if not chunks:
            return

        for chunk in chunks:
            tokens = tokenize(chunk.content)
            self.term_freqs.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        for counter in self.term_freqs:
            for term in counter:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def score(
            self,
            query: str,
            k1: float = 1.5,
            b: float = 0.75
    ) -> list[tuple[Chunk, float]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        results: list[tuple[Chunk, float]] = []
        total_documents = len(self.chunks)

        for index, chunk in enumerate(self.chunks):
            score = 0.0
            counter = self.term_freqs[index]
            document_length = self.doc_lengths[index]
            exact_matches = 0

            for term in query_terms:
                if term not in counter:
                    continue

                term_frequency = counter[term]
                document_frequency = self.doc_freqs.get(term, 0)
                inverse_document_frequency = log(
                    (1 + (total_documents - document_frequency + 0.5))
                    / (document_frequency + 0.5)
                    + 1
                )
                denominator = term_frequency + k1 * (
                    1 - b + b * document_length / self.avg_doc_length
                ) if self.avg_doc_length > 0 else term_frequency + k1
                score += inverse_document_frequency * (
                    term_frequency * (k1 + 1) / denominator
                )

                if term in tokenize(self.chunks[index].content, normalize=False):
                    exact_matches += 1

            if score > 0:
                score += exact_matches * 0.5
                results.append((chunk, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results
