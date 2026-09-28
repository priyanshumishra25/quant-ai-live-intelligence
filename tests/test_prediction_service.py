from pathlib import Path

import math

from services.market_data import MarketDataService
from services.prediction_service import ArtifactRegistry, PredictionService

ROOT = Path(__file__).resolve().parents[1]


def test_persisted_model_inference_is_deterministic():
    market = MarketDataService("bundled", ROOT / "data" / "sample")
    service = PredictionService(market, ArtifactRegistry(ROOT / "artifacts" / "models"))
    first = service.predict("AAPL", "5d")
    second = service.predict("AAPL", "5d")

    assert first["data_mode"] == "bundled_sample_model"
    assert first["predicted_price"] == second["predicted_price"]
    assert first["model_metadata"]["n_samples"] > 100
    assert set(first["feature_importance"]) == {
        "ret_1d", "ret_5d", "ma_gap_5", "ma_gap_20", "vol_10", "momentum_10"
    }
    assert math.isclose(
        first["buy_probability"] + first["hold_probability"] + first["sell_probability"],
        1.0,
        abs_tol=2e-4,
    )


def test_fixture_tickers_match_artifact_registry():
    market = MarketDataService("bundled", ROOT / "data" / "sample")
    registry = ArtifactRegistry(ROOT / "artifacts" / "models")
    for ticker in market.available_tickers():
        for horizon in ("1d", "5d", "20d"):
            assert registry.path_for(ticker, horizon).exists()
