install:
	uv sync

run:
	uv run python3 main.py

run_index:
	uv run python main.py index --max_chunk_size 2000

run_search:
	uv run python3 main.py search "How is CUDA initialized?" --k=5

run_search_dataset:
# 	uv run python3 main.py search "data/raw/vllm-0.10.1" --k=10
	uv run python3 -m src.main search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k=5 \
    --save_directory data/output/search_results

run_answer:
	uv run python3 main.py answer "How to configure OpenAI server ?" --k=10

run_answer_dataset:
	uv run python -m src.main answer_dataset \
    --search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer \
    --limit 3

run_evaluate:
# 	uv run python3 main.py evaluate "data/output/search_results" "data/raw/vllm-0.10.1" --k=10
# 	uv run python3 main.py evaluate --answer_path data/output/search_result.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k=5
	uv run python3 main.py evaluate \
    --answer_path data/output/search_results/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k=5

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


cat_index:
	uv run python main.py index \
		--repository_path data/raw/cats_felines_test \
		--max_chunk_size 500

cat_search1:
	uv run python main.py search \
		"Which feline has rosettes with central spots?" \
		--k 5

cat_search2:
	uv run python main.py search \
		"Which wild cat lives in prides?" \
		--k 5