# Corpus Engine

Deterministic, provider-neutral document extraction, chunking, indexing, and retrieval.

The core package contains no cloud, model-provider, or web-framework dependency. Install
adapter groups only when they are needed:

```shell
pip install corpus-engine
pip install "corpus-engine[parsers]"
pip install "corpus-engine[aws,qdrant]"
```

## Local parsing and chunking

```python
from corpus_engine import (
    DEFAULT_CHUNKERS,
    DEFAULT_PARSERS,
    ChunkerConfig,
    ChunkingStrategy,
    Document,
    DocumentSource,
)

source = DocumentSource(id="guide", uri="guide.md", filename="guide.md")
elements = DEFAULT_PARSERS.create_for(source.filename).parse(
    b"# Guide\n\nCorpus runs locally.",
    source,
)
document = Document(id=source.id, source=source, elements=tuple(elements))
chunks = DEFAULT_CHUNKERS.chunk(
    document,
    ChunkerConfig(strategy=ChunkingStrategy.RECURSIVE),
)
```

Built-in formats are `.txt`, `.md`, `.csv`, `.pdf`, `.xlsx`, `.docx`, and `.pptx`.
The binary formats require the `parsers` extra. Adapter capabilities are exposed by
registries so applications and agents can recommend only installed strategies.

## Entry points

`CorpusEngine` is the primary Python entry point:

```python
from corpus_engine import CorpusEngine

result = CorpusEngine().process_file("guide.md")
print(result.chunks)
```

For local command-line processing:

```shell
corpus-engine guide.md --preview-limit 5
# Equivalent:
python -m corpus_engine guide.md --preview-limit 5
```

The optional FastAPI router is available with the `server` extra:

```python
from fastapi import FastAPI
from corpus_engine.interfaces.api import create_router

app = FastAPI()
app.include_router(create_router())
```

It exposes `GET /v1/engine/capabilities` and
`POST /v1/engine/documents/preview`. The monorepo's `apps/api` application
mounts this router for deployment.
