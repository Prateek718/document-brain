"""Dev utility: empty the Qdrant collection.

Standalone maintenance script, not part of the application. Deletes all points
from the configured collection so the next ingestion starts from a clean slate.

Deletes points (a collection-level operation) rather than dropping the whole
collection, since the application's API key is scoped to the collection and
does not have the cluster-level access that delete_collection requires.

Run from the repo root with the project env active:
    uv run python scripts/wipe_collection.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FilterSelector

from document_brain.config import settings


def main() -> None:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30,
    )
    name = settings.qdrant_collection

    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        print(f"Collection {name!r} does not exist. Nothing to do.")
        return

    count = client.count(collection_name=name).count
    print(f"Collection {name!r} currently holds {count} points.")
    if count == 0:
        print("Already empty. Nothing to do.")
        return

    confirm = input(f"Delete all {count} points from {name!r}? Type 'yes' to proceed: ")
    if confirm.strip().lower() != "yes":
        print("Aborted. Nothing changed.")
        return

    client.delete(
        collection_name=name,
        points_selector=FilterSelector(filter=Filter(must=[])),
        wait=True,
    )
    remaining = client.count(collection_name=name).count
    print(f"Deleted points. Collection {name!r} now holds {remaining} points.")


if __name__ == "__main__":
    main()
