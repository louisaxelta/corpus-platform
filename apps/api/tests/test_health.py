from corpus_api.main import app


def test_application_routes() -> None:
    route_paths = set(app.openapi()["paths"])

    assert "/health" in route_paths
    assert "/v1/engine/capabilities" in route_paths
    assert "/v1/engine/documents/preview" in route_paths
