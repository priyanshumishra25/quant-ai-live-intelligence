.PHONY: bootstrap verify experiment analyze api frontend frontend-build docker docker-observability

bootstrap:
	PYTHONPATH=. python scripts/bootstrap_artifacts.py

experiment:
	PYTHONPATH=. python scripts/run_sample_experiment.py

analyze:
	@test -n "$(QUERY)" || (echo "Usage: make analyze QUERY=Apple HORIZON=5d" && exit 2)
	PYTHONPATH=. python scripts/analyze_stock.py "$(QUERY)" --horizon "$(or $(HORIZON),5d)"

frontend-build:
	cd frontend && npm exec -- tsc -p tsconfig.json

verify:
	python -m compileall -q .
	pytest -q
	cd frontend && npm exec -- tsc -p tsconfig.json --noEmit
	node --check frontend/dist/app.js
	node --check frontend/dist/api.js
	node --check frontend/dist/chart.js
	PYTHONPATH=. python scripts/check_release.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && python -m http.server 8080

docker:
	docker compose -f deployment/docker-compose.yml up --build

docker-observability:
	docker compose -f deployment/docker-compose.yml --profile observability up --build
