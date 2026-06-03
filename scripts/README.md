# Scripts

Local maintenance utilities. Not part of the deployed application; run manually
from the repo root with the project environment active. Both read configuration
(Qdrant URL, API key, collection) from `.env` via the app's settings.

## `wipe_collection.py`

Deletes all points from the configured Qdrant collection, leaving it empty.
Prompts for confirmation and reports the point count before and after. Deletes
points rather than dropping the collection, since the application's API key is
scoped to the collection and lacks cluster-level access.

```
uv run python scripts/wipe_collection.py
```

## `ingest_file.py`

Ingests a PDF from disk through the same pipeline the API uses
(extract → chunk → embed → upsert). Runs the embedding in-process, which avoids
the request timeouts and instance memory limits hit when ingesting a large
corpus through the deployed `/documents` endpoint.

```
uv run python scripts/ingest_file.py "path/to/document.pdf"
```
