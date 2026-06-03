"""Dev utility: ingest a PDF from disk into Qdrant.

Standalone maintenance script, not part of the application. Reads a PDF from
the local filesystem and runs it through the same ingestion pipeline the API
uses (extract -> chunk -> embed -> upsert), writing to the configured Qdrant
collection.

Runs the embedding in-process on the local machine rather than through the
HTTP endpoint, which avoids request timeouts and the memory limits of the
deployed instance when ingesting a large corpus.

Run from the repo root with the project env active:
    uv run python scripts/ingest_file.py "path/to/document.pdf"
"""

import sys
import time
from pathlib import Path

from document_brain.ingestion import ingest_document
from document_brain.vector_store import ensure_collection_exists


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: uv run python scripts/ingest_file.py "path/to/document.pdf"')
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    pdf_bytes = path.read_bytes()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    print(f"Ingesting {path.name} ({size_mb:.1f} MB)...")

    ensure_collection_exists()

    start = time.perf_counter()
    chunks = ingest_document(path.name, pdf_bytes)
    elapsed = time.perf_counter() - start

    print(f"Done. Ingested {chunks} chunks in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
