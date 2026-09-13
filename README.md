# Corpus Platform

Corpus is a monorepo for provider-neutral document intelligence and guided ingestion planning.

## Workspace

- `packages/corpus-engine`: deterministic extraction, chunking, indexing, and retrieval library
- `packages/corpus-agent`: provider-independent ingestion planning and control library
- `apps/api`: FastAPI composition and deployment layer
- `apps/web`: React and TypeScript document workbench

## Development

```shell
uv sync --all-packages
uv run uvicorn corpus_api.main:app --reload
```

```shell
npm install
npm run dev:web
```
