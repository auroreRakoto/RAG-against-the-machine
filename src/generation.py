# ////////////////////////////////////////////////////////////////// #
# ////////////////////////// GENERATION //////////////////////////// #
# ////////////////////////////////////////////////////////////////// #
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from transformers import AutoModelForCausalLM, AutoTokenizer

# from src.logging_config import steps_logger
from src.models import Chunk


class ContextBuilder:
    def build(
        self,
        chunks: list[Chunk],
        max_context_length: int = 8000,
    ) -> str:
        """
        Builds a text context from retrieved chunks.
        """
        if max_context_length <= 0:
            raise ValueError(
                "max_context_length must be greater than zero"
            )

        context_parts: list[str] = []
        current_length = 0

        # steps_logger.info(
        #     "[ContextBuilder] Building context from %d chunks",
        #     len(chunks),
        # )

        for index, chunk in enumerate(chunks):
            chunk_context = (
                f"--- SOURCE {index + 1} ---\n"
                f"File: {chunk.file_path}\n"
                f"Characters: "
                f"{chunk.first_character_index}-"
                f"{chunk.last_character_index}\n\n"
                f"{chunk.text}\n\n"
            )

            remaining_length = (
                max_context_length - current_length
            )

            if remaining_length <= 0:
                break

            if len(chunk_context) > remaining_length:
                chunk_context = chunk_context[
                    :remaining_length
                ]

            context_parts.append(chunk_context)
            current_length += len(chunk_context)

        context = "".join(context_parts)

        # steps_logger.info(
        #     "[ContextBuilder] Context created with %d characters",
        #     len(context),
        # )

        return context


class PromptBuilder:
    def build(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Builds the final prompt sent to the language model.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not context.strip():
            raise ValueError("Context cannot be empty")

        prompt = (
            "You are a retrieval-augmented assistant.\n"
            "Answer the question using only the provided context.\n"
            "Do not use external knowledge.\n"
            "If the context does not contain the answer, say that "
            "the provided sources do not contain enough information.\n"
            "Cite the source file names when useful.\n\n"
            "CONTEXT:\n"
            f"{context}\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "Write a concise answer in one paragraph.\n"
            "Do not repeat the word ANSWER.\n"
            "FINAL ANSWER:\n"
        )

        # steps_logger.info(
        #     "[PromptBuilder] Prompt created with %d characters",
        #     len(prompt),
        # )

        return prompt


class LanguageModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class QwenLanguageModel(LanguageModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        self.model_name = model_name

        # steps_logger.info(
        #     "[QwenLanguageModel] Loading tokenizer: %s",
        #     model_name,
        # )

        self.tokenizer = cast(
            Any,
            AutoTokenizer.from_pretrained(
                model_name,
            ),
        )

        # steps_logger.info("[QwenLanguageModel] Tokenizer loaded")

        # steps_logger.info(
        #     "[QwenLanguageModel] Loading model: %s",
        #     model_name,
        # )

        self.model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
                model_name,
            ),
        )

        # steps_logger.info("[QwenLanguageModel] Model loaded")

    def tokenize_prompt(
        self,
        prompt: str,
    ) -> list[int]:
        """
        Tokenizes the prompt with the Qwen tokenizer.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        token_ids = cast(
            list[int],
            self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
            ),
        )

        # steps_logger.info(
        #     "[QwenLanguageModel] Prompt tokenized into %d tokens",
        #     len(token_ids),
        # )

        return token_ids

    def save_prompt_tokens(
        self,
        prompt: str,
        output_path: str = "data/output/qwen_tokens.txt",
    ) -> None:
        """
        Saves Qwen token ids and readable tokens to a file.
        """
        token_ids = self.tokenize_prompt(prompt)

        tokens = cast(
            list[str],
            self.tokenizer.convert_ids_to_tokens(
                token_ids,
            ),
        )

        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            file.write("MODEL:\n")
            file.write(f"{self.model_name}\n\n")

            file.write("TOKEN COUNT:\n")
            file.write(f"{len(token_ids)}\n\n")

            file.write("TOKENS:\n")
            for index, token in enumerate(tokens):
                file.write(
                    f"{index}: {token_ids[index]} -> {token!s}\n"
                )

        # steps_logger.info(
        #     "[QwenLanguageModel] Qwen tokens saved to: %s",
        #     path,
        # )

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generates an answer from a prompt.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

        answer = cast(
            str,
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ),
        )

        answer = answer.strip()

        if "FINAL ANSWER:" in answer:
            answer = answer.split("FINAL ANSWER:", 1)[-1].strip()

        if "\nQUESTION:" in answer:
            answer = answer.split("\nQUESTION:", 1)[0].strip()

        stop_markers = [
            "\nAnswer the question",
            "\nCite the source",
            "\n```python",
            "\nAnswer:",
        ]

        for marker in stop_markers:
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()

        answer = answer.strip("`").strip()

        # steps_logger.info(
        #     "[QwenLanguageModel] Generated answer with %d characters",
        #     len(answer),
        # )

        return answer.strip()


class AnswerGenerator:
    pass
