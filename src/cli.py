# ////////////////////////////////////////////////////////////////// #
# ////////////////////////////// CLI /////////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
import json
from pathlib import Path

from pydantic import BaseModel

from src.chunking import PythonChunker, TextChunker
from src.evaluation import RecallEvaluator
from src.generation import ContextBuilder, PromptBuilder, QwenLanguageModel
from src.indexing import BM25Index, IndexStorage
from src.ingestion import FileReader, RepositoryLoader
from src.logging_config import steps_logger
from src.models import (
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    SearchResults,
    SearchResultsWithAnswers,
    UnansweredQuestion,
)
from src.retrieval import Retriever
from tqdm import tqdm

class Saving:
    @staticmethod
    def save_text_file(file_path: str, content: str) -> None:
        """
        Saves text content to a file.
        """
        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

        steps_logger.info(
            "[CLI] Saved text file to: %s",
            path,
        )

    @staticmethod
    def save_retrieved_chunks(
        retrieved_chunks: list[Chunk],
        file_path: str = "data/output/retrieved_chunks.txt"
    ) -> None:
        """
        Saves retrieved chunks to a readable debug file.
        """
        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            for index, chunk in enumerate(retrieved_chunks):
                file.write(f"--- RESULT {index + 1} ---\n")
                file.write(f"File: {chunk.file_path}\n")
                file.write(
                    f"Start: {chunk.first_character_index}\n"
                )
                file.write(
                    f"End: {chunk.last_character_index}\n\n"
                )
                file.write(chunk.text)
                file.write("\n\n")

        steps_logger.info(
            "[CLI] Saved %d retrieved chunks to %s",
            len(retrieved_chunks),
            path,
        )

    @staticmethod
    def log_info(msg: str, arg: str) -> None:
        steps_logger.info(
            msg,
            arg
        )

class CLI:
    def _retrieve_chunks(
        self,
        query: str,
        k: int,
    ) -> list[Chunk]:
        """
        Loads the saved index and creates a retriever.
        Retrieves chunks for a query using the saved index.
        """
        index_storage = IndexStorage()
        search_index = index_storage.load(
            directory_path="data/index"
        )

        retriever = Retriever(search_index)

        return retriever.retrieve(
            query=query,
            k=k,
        )

    def _save_json_file(
        self,
        file_path: str,
        content: BaseModel,
    ) -> None:
        """
        Saves a pydantic model to a JSON file.
        """
        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content.model_dump_json(indent=4),
            encoding="utf-8",
        )

        steps_logger.info(
            "[CLI] Saved JSON file to: %s",
            path,
        )

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

        for file_path, text in tqdm(
            files.items(),
            desc="Chunking files",
        ):
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

    def _search_results(
        self,
        retrieved_chunks: list[Chunk],
        query: str,
        k: int
    ) -> SearchResults:
        retrieved_sources = [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            for chunk in retrieved_chunks
        ]

        search_result = SearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="manual",
                    question=query,
                    retrieved_sources=retrieved_sources,
                )
            ],
            k=k,
        )
        return search_result

    def _search_result_for_question(
        self,
        question: UnansweredQuestion,
        retrieved_chunks: list[Chunk],
    ) -> MinimalSearchResults:
        """
        Builds one structured search result for one dataset question.
        """
        retrieved_sources = []

        for chunk in retrieved_chunks:
            retrieved_sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )

        search_results = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=retrieved_sources
        )

        return search_results

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> None:
        """
        Searches the index and saves retrieved chunks and context.
        """
        Saving.log_info("[CLI] Search requested: %s", query)
        retrieved_chunks = self._retrieve_chunks(query, k)
        Saving.save_retrieved_chunks(retrieved_chunks)

        search_result = self._search_results(retrieved_chunks, query, k)
        self._save_json_file("data/output/search_result.json", search_result)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """
        Searches all questions from a dataset and saves structured results.
        """
        steps_logger.info(
            "[CLI] Searching dataset from: %s with top-%d results",
            dataset_path,
            k,
        )

        dataset = self._load_rag_dataset(
            dataset_path=dataset_path,
        )

        search_results: list[MinimalSearchResults] = []

        for question in tqdm(
            dataset.rag_questions,
            desc="Searching questions",
        ):
            retrieved_chunks = self._retrieve_chunks(
                query=question.question,
                k=k,
            )

            search_results.append(
                self._search_result_for_question(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                )
            )

        result = SearchResults(
            search_results=search_results,
            k=k,
        )

        dataset_name = Path(dataset_path).name
        output_path = str(
            Path(save_directory) / dataset_name
        )

        self._save_json_file(
            file_path=output_path,
            content=result,
        )

        steps_logger.info(
            "[CLI] Saved dataset search results to: %s",
            output_path,
        )

        print(f"Saved search results to {output_path}")

    def _answer_results(
        self,
        retrieved_chunks: list[Chunk],
        question: str,
        answer: str,
        k: int,
    ) -> SearchResultsWithAnswers:
        """
        Builds structured answer results from retrieved chunks.
        """
        retrieved_sources = [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            for chunk in retrieved_chunks
        ]

        return SearchResultsWithAnswers(
            search_results=[
                MinimalAnswer(
                    question_id="manual",
                    question=question,
                    retrieved_sources=retrieved_sources,
                    answer=answer,
                )
            ],
            k=k,
        )

    def _build_context(
        self,
        retrieved_chunks: list[Chunk],
        max_context_length: int = 8000,
    ) -> str:
        """
        Builds context from retrieved chunks.
        """
        context_builder = ContextBuilder()

        context = context_builder.build(retrieved_chunks, max_context_length)

        return context

    def answer(
        self,
        question: str,
        k: int = 10,
    ) -> None:
        """
        Answers a question using retrieved context and Qwen.
        """
        Saving.log_info("[CLI] Answer requested: %s", question)

        retrieved_chunks = self._retrieve_chunks(question, k)

        context = self._build_context(retrieved_chunks, 8000)

        prompt_builder = PromptBuilder()

        prompt = prompt_builder.build(question, context)

        Saving.save_retrieved_chunks(retrieved_chunks)
        Saving.save_text_file("data/output/context.txt", context)
        Saving.save_text_file("data/output/prompt.txt", prompt)

        language_model = QwenLanguageModel()

        language_model.save_prompt_tokens(prompt, "data/output/qwen_tokens.txt")

        answer = language_model.generate(prompt)

        answer_res = self._answer_results(retrieved_chunks, question, answer, k)

        self._save_json_file("data/output/answer_result.json", answer_res)

        Saving.save_text_file("data/output/answer.txt", answer)

    def _source_to_chunk(
        self,
        source: MinimalSource,
    ) -> Chunk:
        """
        Rebuilds a Chunk from a MinimalSource by reading the source file.
        """
        path = Path(source.file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Source file not found: {source.file_path}"
            )

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if source.first_character_index < 0:
            raise ValueError(
                "first_character_index cannot be negative"
            )

        if source.last_character_index > len(text):
            raise ValueError(
                f"last_character_index is outside file: {source.file_path}"
            )

        if source.last_character_index <= source.first_character_index:
            raise ValueError(
                "last_character_index must be greater than "
                "first_character_index"
            )

        return Chunk(
            text=text[
                source.first_character_index:
                source.last_character_index
            ],
            file_path=source.file_path,
            first_character_index=source.first_character_index,
            last_character_index=source.last_character_index,
        )

    def _sources_to_chunks(
        self,
        sources: list[MinimalSource],
    ) -> list[Chunk]:
        """
        Rebuilds chunks from minimal source metadata.
        """
        chunks: list[Chunk] = []

        for source in sources:
            chunks.append(
                self._source_to_chunk(source)
            )

        return chunks

    def _answer_result_for_search_result(
        self,
        search_result: MinimalSearchResults,
        answer: str,
    ) -> MinimalAnswer:
        """
        Builds one answer result while preserving the question_id.
        """
        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer,
        )

    def answer_dataset(
        self,
        search_results_path: str,
        save_directory: str,
        max_context_length: int = 8000,
        limit: int | None = None,
    ) -> None:
        """
        Generates answers for all questions from saved search results.
        """
        steps_logger.info(
            "[CLI] Answering dataset from: %s",
            search_results_path,
        )

        search_results = self._load_search_results(
            answer_path=search_results_path
        )

        results_to_answer = search_results.search_results

        if limit is not None:
            results_to_answer = results_to_answer[:limit]

        prompt_builder = PromptBuilder()
        language_model = QwenLanguageModel()

        answered_results: list[MinimalAnswer] = []

        total_questions = len(results_to_answer)

        print(f"Loaded {total_questions} questions from {search_results_path}")

        for search_result in tqdm(
            results_to_answer,
            desc="Answering questions",
        ):
            retrieved_chunks = self._sources_to_chunks(
                search_result.retrieved_sources
            )

            if not retrieved_chunks:
                answer = (
                    "The provided sources do not contain enough information."
                )
            else:
                context = self._build_context(
                    retrieved_chunks=retrieved_chunks,
                    max_context_length=max_context_length,
                )

                prompt = prompt_builder.build(
                    question=search_result.question,
                    context=context,
                )

                answer = language_model.generate(prompt)

            answered_results.append(
                self._answer_result_for_search_result(
                    search_result=search_result,
                    answer=answer,
                )
            )

            print(f"Processed {index} of {total_questions} questions")

        result = SearchResultsWithAnswers(
            search_results=answered_results,
            k=search_results.k,
        )

        dataset_name = Path(search_results_path).name
        output_path = str(
            Path(save_directory) / dataset_name
        )

        self._save_json_file(
            file_path=output_path,
            content=result,
        )

        Saving.log_info(
            "[CLI] Saved dataset answers to: ",
            output_path
        )

    def _load_rag_dataset(
        self,
        dataset_path: str,
    ) -> RagDataset:
        """
        Loads a RAG dataset from a JSON file.
        """
        path = Path(dataset_path)

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        return RagDataset.model_validate(data)

    def _load_search_results(
        self,
        answer_path: str,
    ) -> SearchResults:
        """
        Loads search results from a JSON file.
        """
        path = Path(answer_path)

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        return SearchResults.model_validate(data)

    def evaluate(
        self,
        answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> None:
        """
        Evaluates search results against an answered dataset.
        """
        steps_logger.info(
            "[CLI] Evaluating results from: %s against dataset: %s",
            answer_path,
            dataset_path,
        )

        expected_dataset = self._load_rag_dataset(dataset_path)

        search_results = self._load_search_results(answer_path)

        k_values = []
        for value in tqdm(
            [1, 3, 5, k], 
            desc="Evaluating k-values"
        ):
            if value <= k:
                k_values.append(value)

        evaluator = RecallEvaluator()

        scores = evaluator.evaluate(expected_dataset, search_results, k_values)

        lines = [
            "Evaluation Results",
            "========================================",
            f"Max context length: {max_context_length}",
        ]

        for k_value, score in scores.items():
            lines.append(f"Recall@{k_value}: {score:.3f}")

        result_text = "\n".join(lines)

        print(result_text)

        Saving.save_text_file("data/output/evaluation.txt", result_text)
