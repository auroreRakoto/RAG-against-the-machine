import time

import fire

from src.cli import CLI


def main() -> None:
    """Run the command-line interface and report errors without tracebacks."""
    start = time.perf_counter()

    try:
        fire.Fire(CLI, name="RAG CLI")
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1)

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")


if __name__ == "__main__":
    main()
