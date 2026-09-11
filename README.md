*This project has been created as part of the 42 curriculum by aurrakot.*

# RAG against the machine

A local Retrieval-Augmented Generation (RAG) system that indexes the supplied
vLLM repository, retrieves relevant Python/documentation chunks with BM25, and
uses `Qwen/Qwen3-0.6B` to answer questions from retrieved context.

## System architecture

The pipeline is:

1. **Ingestion** — `RepositoryLoader` recursively reads useful `.py` and `.md`
   files from the vLLM repository.
2. **Chunking** — files are split into bounded chunks while preserving source
   character offsets.
3. **Indexing** — chunks are tokenized and stored in a BM25 index.
4. **Retrieval** — a query is ranked against the BM25 index and the top-k
   chunks are returned.
5. **Context building** — retrieved chunks and their source metadata are
   assembled within a bounded context size.
6. **Generation** — `Qwen/Qwen3-0.6B` receives the question and retrieved
   context and generates a source-grounded answer.
7. **Evaluation** — Recall@k is calculated by comparing retrieved source
   intervals against ground-truth intervals. A source counts as found when
   there is at least 5% overlap.

Main modules:

- `src/ingestion.py`: repository loading
- `src/chunking.py`: Python and text chunking
- `src/indexing.py`: BM25 index and persistence
- `src/retrieval.py`: top-k retrieval
- `src/generation.py`: context, prompt and Qwen inference
- `src/evaluation.py`: Recall@k evaluation
- `src/cli.py`: Python Fire CLI
- `src/models.py`: Pydantic data models

## Chunking strategy

Two chunking strategies are implemented.

### Python code

`PythonChunker` tries to split at a newline before the maximum chunk size.
This avoids cutting a line of Python code when possible.

### Markdown/text

`TextChunker` first tries to split at a newline, then at a space, and finally
falls back to the maximum boundary when no suitable separator exists.

The default maximum chunk size is **2000 characters** and can be changed with
`--max_chunk_size`.

Smaller chunks give more precise matches but may lose surrounding context and
increase the total number of indexed chunks. Larger chunks preserve more
context but can add irrelevant terms, reduce retrieval precision, and consume
more of the LLM context window.

## Retrieval method

The mandatory lexical retriever uses **BM25** through `rank-bm25`.

During indexing, every chunk is lowercased and tokenized. At query time, the
query is tokenized with the same strategy and BM25 assigns a relevance score
to every chunk. Results are sorted by decreasing score and the top-k chunks
are returned.

BM25 was chosen because it is simple, CPU-friendly, fast to index and query,
and provides a strong lexical baseline for technical documentation and source
code.

## Design decisions and trade-offs

- **BM25 instead of embeddings:** lower setup cost and fast CPU execution, at
  the cost of weaker semantic matching for queries whose wording differs from
  the source.
- **Separate Python/text chunkers:** both keep exact source offsets while
  allowing file-type-specific splitting.
- **Persisted index:** avoids rebuilding the index for every query.
- **Grounded prompting:** the generation prompt instructs Qwen to use only
  retrieved context and to state when the sources are insufficient.
- **Character offsets:** each retrieved result preserves `file_path`,
  `first_character_index` and `last_character_index` for evaluation.

## Usage

Install dependencies:

```bash
make install
```

Index the supplied vLLM repository:

```bash
uv run python -m src index --max_chunk_size 2000
```

Search a single question:

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

Answer a single question:

```bash
uv run python -m src answer "How to configure OpenAI server?" --k 10
```

Search a dataset:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results
```

Evaluate retrieved results:

```bash
uv run python -m src evaluate \
  --answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5
```

Run lint:

```bash
make lint
```

## Performance analysis

The mandatory targets are:

| Metric | Required |
| --- | ---: |
| Docs Recall@5 | >= 80% |
| Code Recall@5 | >= 50% |
| Indexing time | <= 300 s |
| Retrieval throughput | correction-script limit |


| Metric | Measured |
| --- | ---: |
| Docs Recall@5 | TODO |
| Code Recall@5 | TODO |
| Indexing time | TODO |
| Retrieval throughput | TODO |

