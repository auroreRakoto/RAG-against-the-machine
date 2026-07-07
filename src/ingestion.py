# ////////////////////////////////////////////////////////////////// #
# /////////////////////////// INGESTION //////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from pathlib import Path

from src.logging_config import files_logger, steps_logger


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
