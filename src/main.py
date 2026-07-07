import time

import fire

from src.cli import CLI

def main():
    """
    Entry point for the CLI.
    """
    fire.Fire(CLI, name="RAG CLI")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")
