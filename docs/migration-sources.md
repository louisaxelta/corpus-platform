# Migration sources

## `rag-pipelines`

- Source: `C:\FILES\repos\rag-pipelines`
- Revision: `7075137013a6a6a0a2b7170d7d640e4002640e32`
- License: MIT, copyright Louis Axel
- Status: read-only implementation reference

### Inventory

- Parser behavior: redesigned and ported to `corpus_engine.ingestion.parsers`
- Chunker behavior: redesigned and ported to `corpus_engine.ingestion.chunkers`
- Legacy models: replaced by versioned, immutable domain models
- AWS clients and AWS-bound workflow: rejected from the engine core
- LLM chunking: redesigned as an optional, provider-neutral LiteLLM adapter
- API and application configuration: not migrated into the engine

Corrections made during the port include native text and Markdown parsers, removal of
unsupported legacy Office extensions, explicit parser capabilities, provenance-preserving
elements and chunks, and independent implementations for deterministic chunking strategies.
