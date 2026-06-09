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

def ingest():
    pass


def retrieve_docs():
    """Go through all the documents and separate them into chunks of 2000 characters. Then, classify the chunks using a retrieval method (TF-IDF or BM25)."""
    directory = Path("vllm-0.10.1")
    with open("data.txt", "r") as f:
        data = f.read()



def main():
    pass


if __name__ == "__main__":
    main()
