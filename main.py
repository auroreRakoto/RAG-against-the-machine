# parourir les files text + py
#   chunking 2000 char

#   algo de recherche + ingestion ==> classer les chunks

        # One of the two following retrieving methods must be implemented, the choice is up to you:
        # • TF-IDF
        # • BM25

        # You can also explore other retrieving methods, as long as one of the two above is imple
        # mented.

# import un model -> ollama / qwen



"""
Step 1: Read all database
    - Print all valid files
    - Print the content of some of the files

Step 2: Make the database usable
    are all files of the same value?
    do I keep the information of path?
    - Chunk the files in 2000 char strings
    - Upgrade chunking
        - don't cut words
        - avoid cutting paragraphs
        - chunk by context?
    - Indexing
        - check multi-layers indexing

Step 3: See later
"""


"""
• Build an indexed knowledge base from the project attached files.
    - Implement a robust ingestion pipeline to process various file types (e.g., .txt, .py).

• Retrieve and rank the most relevant pieces of information.
• Pass them to the LLM within context limitations.
• Generate structured JSON output as described in the output section.
• Implement intelligent chunking strategies for the different file types.
• Provide a comprehensive CLI interface for all operations.
• Include evaluation metrics and performance analysis.
"""

from pathlib import Path
import os

def ingest():
    pass


def retrieve_docs_listdir():
    directory = "vllm-0.10.1"
    lst_dir = os.listdir(directory)
    # print(f"Files and dirs in {directory}:\n{lst_dir}")
    for f in lst_dir:
        if os.path.isfile(directory+'/'+f) and (f.endswith(".txt") or f.endswith(".md")):
            print(f"file: {f}")
        # if os.path.isdir(directory+'/'+f):
        #     print(f"dir: {f}")


def retrieve_docs_walk():
    directory = "vllm-0.10.1"
    lst_dir = os.listdir(directory)

    lst = []
    for _, dirs, files in os.walk(directory):
        # print(files)
        for f in files:
            if f.endswith(".txt") or f.endswith(".md"):
                print(f"file: {f}")
                # print(dirs)



def main():
    # retrieve_docs_listdir()
    retrieve_docs_walk()


if __name__ == "__main__":
    print("--- Before main ---\n")
    main()
    print("\n--- After  main ---")
