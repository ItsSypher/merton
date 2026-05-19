"""FastAPI server tests via TestClient (no real HTTP server boot)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from merton.excel.server import build_functions_metadata, make_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(make_app())


class TestFunctionsMetadata:
    def test_count_matches_registry(self) -> None:
        meta = build_functions_metadata()
        assert len(meta["functions"]) == 10

    def test_functions_have_description_and_params(self) -> None:
        meta = build_functions_metadata()
        for fn in meta["functions"]:
            assert fn["name"]
            assert fn["description"]
            assert isinstance(fn["parameters"], list)


class TestEndpoints:
    def test_healthz(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_functions_json(self, client: TestClient) -> None:
        r = client.get("/functions.json")
        assert r.status_code == 200
        data = r.json()
        names = {fn["name"] for fn in data["functions"]}
        assert {"DD", "PD", "SPREAD", "GREEKS", "BACKTEST"}.issubset(names)

    def test_functions_js(self, client: TestClient) -> None:
        r = client.get("/static/functions.js")
        assert r.status_code == 200
        assert "CustomFunctions.associate" in r.text

    def test_functions_html(self, client: TestClient) -> None:
        r = client.get("/static/functions.html")
        assert r.status_code == 200
        assert "office.js" in r.text

    def test_taskpane(self, client: TestClient) -> None:
        r = client.get("/taskpane.html")
        assert r.status_code == 200
        assert "merton" in r.text.lower()


class TestCallEndpoint:
    def test_dd_call(self, client: TestClient) -> None:
        r = client.post("/call", json={"function": "DD", "args": [100, 0.30, 35, 0.04, 1.0]})
        assert r.status_code == 200
        body = r.json()
        assert "result" in body
        assert 0 < body["result"] < 20

    def test_pd_call(self, client: TestClient) -> None:
        r = client.post("/call", json={"function": "PD", "args": [100, 0.30, 35, 0.04, 1.0]})
        assert r.status_code == 200
        assert 0 <= r.json()["result"] <= 1

    def test_greeks_call(self, client: TestClient) -> None:
        r = client.post("/call", json={"function": "GREEKS", "args": [100, 0.30, 35, 0.04, 1.0]})
        assert r.status_code == 200
        result = r.json()["result"]
        assert isinstance(result, list)
        assert result[0][0] == "delta"

    def test_unknown_function_returns_404(self, client: TestClient) -> None:
        r = client.post("/call", json={"function": "NOPE", "args": []})
        assert r.status_code == 404

    def test_invalid_inputs_return_error_field(self, client: TestClient) -> None:
        r = client.post("/call", json={"function": "DD", "args": [100, 0.30, -1.0, 0.04, 1.0]})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_pd_term_call(self, client: TestClient) -> None:
        r = client.post(
            "/call",
            json={"function": "PD_TERM", "args": [100, 0.30, 35, 0.04, [0.25, 1.0, 5.0]]},
        )
        assert r.status_code == 200
        result = r.json()["result"]
        assert result[0] == ["horizon_years", "pd"]
        assert len(result) == 4

    def test_backtest_call(self, client: TestClient) -> None:
        preds = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        defs = [0, 0, 0, 1, 1, 1]
        r = client.post("/call", json={"function": "BACKTEST", "args": [preds, defs, "AUC"]})
        assert r.status_code == 200
        assert r.json()["result"] == 1.0
