.PHONY: dev api web db test lint

dev:
	docker compose up -d postgres minio
	cd apps/api && uvicorn rag_platform.main:app --reload --host 0.0.0.0 --port 8000

api:
	cd apps/api && uvicorn rag_platform.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd apps/web && npm run dev

db:
	docker compose up -d postgres minio

test:
	cd apps/api && python -m pytest

lint:
	cd apps/api && ruff check src tests

