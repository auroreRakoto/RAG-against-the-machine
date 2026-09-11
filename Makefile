.PHONY: install run debug clean lint \
	run_index run_search run_search_dataset run_answer \
	run_answer_dataset run_evaluate

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;

run_index:
	uv run python -m src index --max_chunk_size 2000

run_search:
	uv run python -m src search "How is CUDA initialized?" --k 5

run_search_dataset:
	uv run python -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 5 \
		--save_directory data/output/search_results

run_answer:
	uv run python -m src answer "How to configure OpenAI server?" --k 10

run_answer_dataset:
	uv run python -m src answer_dataset \
		--search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer \
		--limit 3

run_evaluate:
	uv run python -m src evaluate \
		--answer_path data/output/search_results/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 5
