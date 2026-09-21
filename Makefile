PORT ?= 8000
LT_CACHE ?= $(HOME)/.cache/live-transcribe

.DEFAULT_GOAL := help
.PHONY: help setup setup-mt run dev test check clean cache-clean

help:
	@echo "live-transcribe"
	@echo
	@echo "  make setup       install base deps (en-en, ja-ja, ja-en-fast)"
	@echo "  make setup-mt    install base + torch/transformers for ja-en"
	@echo "  make run         start the server on :$(PORT)"
	@echo "  make dev         start with auto-reload"
	@echo "  make test        run the test suite"
	@echo "  make check       byte-compile the python modules"
	@echo "  make clean       remove .venv and caches"
	@echo "  make cache-clean remove derived PCM/remuxed media ($(LT_CACHE))"
	@echo
	@echo "Override the port with: make run PORT=9000"
	@echo "Pass a token with   : HF_TOKEN=hf_xxx make run"

setup:
	uv sync

setup-mt:
	uv sync --extra mt

run:
	uv run uvicorn app:app --port $(PORT)

dev:
	uv run uvicorn app:app --port $(PORT) --reload

test:
	uv run pytest

check:
	uv run python -m py_compile app.py pipeline.py backends.py

clean:
	rm -rf .venv __pycache__ .pytest_cache tests/__pycache__

cache-clean:
	rm -rf "$(LT_CACHE)"
