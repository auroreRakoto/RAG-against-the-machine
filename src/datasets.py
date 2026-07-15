# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// DATASETS ///////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from __future__ import annotations

import json
from pathlib import Path

from src.models import RagDataset


class DatasetLoader:
    def load(self, dataset_path: str) -> RagDataset:
        """
        Loads a RAG dataset from a JSON file.
        """
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}"
            )

        with path.open(mode="r", encoding="utf-8") as file:
            data = json.load(file)

        return RagDataset.model_validate(data)

    def load_many(self, dataset_paths: list[str]) -> list[RagDataset]:
        """
        Loads multiple datasets and returns them in the same order.
        """
        return [self.load(path) for path in dataset_paths]


class DatasetWriter:
    def save(self, dataset: RagDataset, output_path: str) -> None:
        """
        Saves a RagDataset into a JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="w", encoding="utf-8") as file:
            json.dump(dataset.model_dump(mode="json"), file, indent=4)

    def save_many(self, datasets: list[RagDataset], output_path: str) -> None:
        """
        Saves multiple datasets in a single file.
        """
        if not datasets:
            raise ValueError("datasets cannot be empty")

        combined = {
            "rag_questions": [
                question
                for dataset in datasets
                for question in dataset.rag_questions
            ]
        }

        self.save(RagDataset.model_validate(combined), output_path)
