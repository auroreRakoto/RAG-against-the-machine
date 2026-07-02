install:
	uv sync

run:
	uv run python3 main.py

run_index:
	uv run python main.py index --max_chunk_size 2000

run_search:
	uv run python3 main.py search "How is CUDA initialized?" --k=5

run_search_dataset:
	uv run python3 main.py search "data/raw/vllm-0.10.1" --k=10

run_answer:
	uv run python3 main.py answer "How to configure OpenAI server ?" --k=10

run_answer_dataset:
	uv run python3 main.py answer "data/output/search_results" "data/raw/vllm-0.10.1" --k=10

run_evaluate:
	uv run python3 main.py evaluate "data/output/search_results" "data/raw/vllm-0.10.1" --k=10

debug:
	uv run python3 -m pdb main.py

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {}
