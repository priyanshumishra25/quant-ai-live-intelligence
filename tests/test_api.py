from fastapi.testclient import TestClient

from api.main import app


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/token",
        json={"username": "demo", "password": "quantai-demo"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_auth_and_prediction_end_to_end():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        body = health.json()
        assert body["models_loaded"] is True
        assert body["model_count"] == 15
        assert body["data_mode"] == "bundled_synthetic_fixture"

        denied = client.post("/api/v1/predict", json={"ticker": "AAPL", "horizon": "5d"})
        assert denied.status_code == 401

        headers = auth_headers(client)
        pred = client.post(
            "/api/v1/predict",
            headers=headers,
            json={"ticker": "AAPL", "horizon": "5d"},
        )
        assert pred.status_code == 200
        payload = pred.json()
        assert payload["ticker"] == "AAPL"
        assert payload["data_mode"] == "bundled_sample_model"
        assert payload["model_weights"] == {"ridge_linear_return": 1.0}


def test_history_overview_and_backtest_end_to_end():
    with TestClient(app) as client:
        headers = auth_headers(client)
        history = client.get("/api/v1/market/history/SPY?limit=30", headers=headers)
        assert history.status_code == 200
        assert len(history.json()["rows"]) == 30

        overview = client.get("/api/v1/market/overview", headers=headers)
        assert overview.status_code == 200
        assert overview.json()["top_movers"]

        bt = client.post(
            "/api/v1/backtest",
            headers=headers,
            json={"ticker": "SPY", "horizon": "5d", "train_size": 300, "step_size": 63, "n_splits": 2},
        )
        assert bt.status_code == 200
        payload = bt.json()
        assert payload["method"] == "walk_forward_ridge_linear"
        assert payload["equity_curve"]
        assert "total_return_pct" in payload["stats"]
        assert payload["warnings"]

        csv_export = client.get("/api/v1/backtest/SPY/equity.csv?horizon=5d", headers=headers)
        assert csv_export.status_code == 200
        assert csv_export.headers["content-type"].startswith("text/csv")
        assert csv_export.text.startswith("date,equity\n")


def test_experiment_lab_and_system_overview():
    with TestClient(app) as client:
        headers = auth_headers(client)
        comparison = client.post(
            "/api/v1/experiments/compare",
            headers=headers,
            json={
                "ticker": "SPY", "horizon": "5d",
                "models": ["ridge", "momentum"],
                "train_size": 300, "step_size": 63, "n_splits": 1,
                "commission_bps": 8, "slippage_bps": 3,
            },
        )
        assert comparison.status_code == 200
        payload = comparison.json()
        assert [item["model"] for item in payload["comparisons"]] == ["ridge", "momentum"]
        assert payload["costs"] == {"commission_bps": 8.0, "slippage_bps": 3.0}
        assert all("stats" in item and "equity_curve" in item for item in payload["comparisons"])

        system = client.get("/api/v1/system", headers=headers)
        assert system.status_code == 200
        body = system.json()
        assert body["version"] == "3.0.0"
        assert body["models_loaded"] == 15
        assert "walk-forward evaluation" in body["capabilities"]


def test_live_intelligence_is_explicitly_gated_without_provider_key():
    with TestClient(app) as client:
        headers = auth_headers(client)
        response = client.post(
            "/api/v1/intelligence/analyze",
            headers=headers,
            json={"query": "Apple", "horizon": "5d"},
        )
        assert response.status_code == 503
        assert "ALPHA_VANTAGE_KEY" in response.json()["detail"]
