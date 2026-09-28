"""Fast, network-free release integrity checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"release check failed: {message}")


def main() -> None:
    # A local `.env` is expected during development. It is a release problem
    # only if Git tracks it; `.gitignore` already excludes it. Archive packaging
    # performs an additional hard check that `.env` is absent from the ZIP.
    git_dir = ROOT / ".git"
    if git_dir.exists():
        import subprocess
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if tracked.returncode == 0:
            fail(".env is tracked by git; remove it from the index immediately")
    required = [
        "api/main.py", "frontend/index.html", "frontend/src/app.ts", "frontend/src/api.ts", "frontend/dist/app.js",
        "deployment/Dockerfile", "deployment/docker-compose.yml",
        "data/sample/README.md", "artifacts/models/registry.json",
    ]
    for rel in required:
        if not (ROOT / rel).exists():
            fail(f"missing {rel}")
    registry = json.loads((ROOT / "artifacts/models/registry.json").read_text())
    if registry.get("data_source") != "deterministic synthetic research fixture":
        fail("bundled registry must identify the synthetic fixture")
    for ticker, horizons in registry.get("tickers", {}).items():
        for horizon, filename in horizons.items():
            path = ROOT / "artifacts" / "models" / filename
            if not path.exists():
                fail(f"missing artifact {ticker}/{horizon}: {filename}")
            payload = json.loads(path.read_text())
            if payload.get("data_source") != "bundled_synthetic_fixture":
                fail(f"artifact {filename} is not labelled as bundled synthetic data")

    manifest_path = ROOT / "artifacts" / "MANIFEST.sha256"
    if not manifest_path.exists():
        fail("missing artifacts/MANIFEST.sha256")
    for line in manifest_path.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.exists():
            fail(f"manifest path missing: {rel}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            fail(f"hash mismatch: {rel}")
    print("release integrity checks passed")


if __name__ == "__main__":
    main()
