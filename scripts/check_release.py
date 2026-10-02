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

    forbidden_prototypes = [
        "models", "features", "risk", "database", "data/ingestion",
        "data/scraping", "data/streaming", "services/retrainer.py",
        "requirements-research.txt",
    ]
    for rel in forbidden_prototypes:
        if (ROOT / rel).exists():
            fail(f"disconnected prototype path must not ship on main: {rel}")

    public_copy = [ROOT / "README.md", ROOT / "docs" / "ENGINEERING.md", ROOT / "docs" / "MODEL_CARD.md"]
    for path in public_copy:
        if "interview" in path.read_text().lower():
            fail(f"interview-specific copy remains in {path.relative_to(ROOT)}")

    frontend_html = (ROOT / "frontend" / "index.html").read_text()
    frontend_css = (ROOT / "frontend" / "styles.css").read_text()
    frontend_ts = (ROOT / "frontend" / "src" / "app.ts").read_text()
    forbidden_frontend_tokens = {
        "linear-gradient": "gradients",
        "radial-gradient": "radial gradients",
        "box-shadow": "drop shadows",
        "space grotesk": "Space Grotesk",
        "font-family: inter": "Inter",
        "font-family: geist": "Geist",
        "lucide": "Lucide icons",
        "\u2014": "em dashes",
    }
    combined_frontend = "\n".join([frontend_html.lower(), frontend_css.lower(), frontend_ts.lower()])
    for token, label in forbidden_frontend_tokens.items():
        if token.lower() in combined_frontend:
            fail(f"frontend design exclusion violated: {label}")
    if "#fff" in frontend_css.lower() or "#ffffff" in frontend_css.lower():
        fail("frontend design exclusion violated: pure white color token")
    for required_copy in ["Privacy Policy", "Terms of Use", "Model Card", "Security"]:
        if required_copy.lower() not in frontend_html.lower():
            fail(f"frontend trust page missing: {required_copy}")
    if "software + ai engineering portfolio" in frontend_html.lower():
        fail("portfolio-oriented marketing copy remains in frontend")
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
