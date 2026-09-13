from corpus_engine.interfaces.api import create_router
from fastapi import FastAPI

app = FastAPI(title="Corpus API", version="0.1.0")
app.include_router(create_router())


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
