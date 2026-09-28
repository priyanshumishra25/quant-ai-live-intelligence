from config import APISettings, InfraSettings


def test_api_environment_aliases(monkeypatch):
    monkeypatch.setenv("API_PORT", "9123")
    monkeypatch.setenv("API_WORKERS", "2")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    cfg = APISettings(_env_file=None)
    assert cfg.port == 9123
    assert cfg.workers == 2
    assert cfg.jwt_secret == "test-secret"


def test_infra_environment_aliases(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://example:6379/3")
    monkeypatch.setenv("KAFKA_BROKERS", "kafka:29092")
    cfg = InfraSettings(_env_file=None)
    assert cfg.redis_url == "redis://example:6379/3"
    assert cfg.kafka_brokers == "kafka:29092"
