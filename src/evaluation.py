# ////////////////////////////////////////////////////////////////// #
# ////////////////////////// EVALUATION //////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    SearchResults,
)


class RecallEvaluator:
    MINIMUM_OVERLAP_RATIO = 0.05

    def evaluate(
        self,
        expected_dataset: RagDataset,
        search_results: SearchResults,
        k_values: list[int],
    ) -> dict[int, float]:
        """
        Evaluates recall@k for multiple k values.
        """
        if not expected_dataset.rag_questions:
            raise ValueError("Expected dataset cannot be empty")

        results_by_question_id = {
            result.question_id: result
            for result in search_results.search_results
        }

        scores: dict[int, float] = {}

        for k in k_values:
            total_score = 0.0
            evaluated_questions = 0

            for expected_question in expected_dataset.rag_questions:
                if not isinstance(expected_question, AnsweredQuestion):
                    continue

                retrieved_result = results_by_question_id.get(
                    expected_question.question_id
                )

                if retrieved_result is None:
                    question_score = 0.0
                else:
                    question_score = self.evaluate_question(
                        expected_question=expected_question,
                        retrieved_result=retrieved_result,
                        k=k,
                    )

                total_score += question_score
                evaluated_questions += 1

            if evaluated_questions == 0:
                scores[k] = 0.0
            else:
                scores[k] = total_score / evaluated_questions

        return scores

    def evaluate_question(
        self,
        expected_question: AnsweredQuestion,
        retrieved_result: MinimalSearchResults,
        k: int,
    ) -> float:
        """
        Evaluates recall@k for a single question.
        """
        if not expected_question.sources:
            return 0.0

        retrieved_sources = retrieved_result.retrieved_sources[:k]

        found_sources = self.count_found_sources(
            expected_sources=expected_question.sources,
            retrieved_sources=retrieved_sources,
        )

        return found_sources / len(expected_question.sources)

    def is_source_found(
        self,
        expected_source: MinimalSource,
        retrieved_sources: list[MinimalSource],
    ) -> bool:
        """
        Checks whether an expected source is found in retrieved sources.
        """
        for retrieved_source in retrieved_sources:
            if expected_source.file_path != retrieved_source.file_path:
                continue

            overlap_ratio = self.calculate_overlap_ratio(
                first_source=expected_source,
                second_source=retrieved_source,
            )

            if overlap_ratio >= self.MINIMUM_OVERLAP_RATIO:
                return True

        return False

    def calculate_overlap_ratio(
        self,
        first_source: MinimalSource,
        second_source: MinimalSource,
    ) -> float:
        """
        Calculates overlap ratio between two source intervals.
        """
        first_length = (
            first_source.last_character_index
            - first_source.first_character_index
        )

        if first_length <= 0:
            return 0.0

        overlap_start = max(
            first_source.first_character_index,
            second_source.first_character_index,
        )

        overlap_end = min(
            first_source.last_character_index,
            second_source.last_character_index,
        )

        overlap_length = max(
            0,
            overlap_end - overlap_start,
        )

        return overlap_length / first_length

    def count_found_sources(
        self,
        expected_sources: list[MinimalSource],
        retrieved_sources: list[MinimalSource],
    ) -> int:
        """
        Counts how many expected sources are found.
        """
        found_sources = 0

        for expected_source in expected_sources:
            if self.is_source_found(
                expected_source=expected_source,
                retrieved_sources=retrieved_sources,
            ):
                found_sources += 1

        return found_sources
