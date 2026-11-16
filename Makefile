# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Whisper Transcription Manager — Makefile                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
#  make setup          Full setup (GPU auto-detect)
#  make setup-cpu      Setup without GPU
#  make setup-dev      Setup with developer tools
#  make setup-locked   NVIDIA setup with the exact pinned versions
#  make hf-token       Set up speaker diarization (Hugging Face token)
#  make run            Start server
#  make test           Run tests
#  make test-cov       Tests with coverage
#  make check          Hardware check
#  make clean          Delete caches and build artifacts
#  make reset          Delete the venv (then: make setup)

SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────

.PHONY: help
help: ## This help
	@echo ""
	@echo "Whisper Transcription Manager — available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────

.PHONY: setup
setup: ## Full setup (GPU auto-detect)
	@bash setup.sh

.PHONY: setup-nvidia
setup-nvidia: ## Setup with NVIDIA CUDA
	@bash setup.sh --gpu nvidia

.PHONY: setup-amd
setup-amd: ## Setup with AMD ROCm
	@bash setup.sh --gpu amd

.PHONY: setup-cpu
setup-cpu: ## CPU-only setup (no GPU)
	@bash setup.sh --gpu cpu

.PHONY: setup-dev
setup-dev: ## Setup with developer tools
	@bash setup.sh --dev

.PHONY: setup-full
setup-full: ## Setup with everything (dev tools + llama.cpp)
	@bash setup.sh --dev --llama

# The exact environment of requirements-lock-nvidia.txt. Torch's +cu128
# builds come from the PyTorch index; torchcodec is then reinstalled as the
# CPU build from PyPI (the cu128 one needs NVIDIA's NPP library to load);
# requirements-lock-overrides.txt settles one conflict between two pins.
.PHONY: setup-locked
setup-locked: ## NVIDIA setup with the exact versions of requirements-lock-nvidia.txt
	@command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
	@test -d $(VENV) || uv venv $(VENV) --python 3.12 --relocatable --seed
	uv pip install --python $(PYTHON) --index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cu128 \
		--override requirements-lock-overrides.txt -r requirements-lock-nvidia.txt
	uv pip install --python $(PYTHON) --index-url https://pypi.org/simple --reinstall-package torchcodec \
		"torchcodec==$$(sed -n 's/^torchcodec==//p' requirements-lock-nvidia.txt)"
	uv pip install --python $(PYTHON) -e .
	@echo ""
	@echo "Speaker diarization needs a free Hugging Face token once: make hf-token"
	@echo "(docs/GUIDE.md: Speaker diarization: the Hugging Face token). Then: make run"

.PHONY: hf-token
hf-token: ## Speaker diarization: check and save the Hugging Face token, download the model
	@$(PYTHON) -m whisper_tm --hf-token

# ── Server ────────────────────────────────────────────────────────────────

.PHONY: run
run: ## Start server
	@$(PYTHON) -m whisper_tm

.PHONY: run-debug
run-debug: ## Start server in debug mode
	@$(PYTHON) -m whisper_tm --debug

.PHONY: run-mcp
run-mcp: ## Start MCP server (HTTP, for a local AI)
	@$(PYTHON) -m whisper_tm.mcp_server --http

.PHONY: run-openapi
run-openapi: ## Expose tools as an OpenAPI service (Open WebUI)
	@$(PYTHON) -m whisper_tm.mcp_server --openapi

.PHONY: mcp-check
mcp-check: ## Check whether the MCP server can reach the app
	@$(PYTHON) -m whisper_tm.mcp_server --check

.PHONY: check
check: ## Hardware diagnostics
	@$(PYTHON) -m whisper_tm --check

# ── Tests ─────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run tests
	@$(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov: ## Tests with coverage report
	@$(PYTHON) -m pytest tests/ --cov=whisper_tm --cov-report=term-missing -v --tb=short

# ── Code quality ──────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Check code with ruff
	@$(PYTHON) -m ruff check .

.PHONY: lint-fix
lint-fix: ## Auto-fix code with ruff
	@$(PYTHON) -m ruff check --fix .

.PHONY: typecheck
typecheck: ## Type check with mypy
	@$(PYTHON) -m mypy whisper_tm/ --ignore-missing-imports


# ── Browser extension ─────────────────────────────────────────────────────

.PHONY: extension-zip
extension-zip: ## Zip the browser extension into dist/ (load unpacked, and a store copy)
	@$(PYTHON) tools/extension_zip.py

# ── Cleanup ───────────────────────────────────────────────────────────────

# Not into the venv (its own caches) or the downloaded llama.cpp
.PHONY: clean
clean: ## Delete caches and build artifacts
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage build dist *.egg-info
	@find . \( -path ./$(VENV) -o -path ./.git -o -path ./llama.cpp \) -prune \
		-o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned up."

.PHONY: reset
reset: clean ## Delete the venv (then: make setup)
	@rm -rf $(VENV)
	@echo "venv deleted. Please run 'make setup'."

# ── Database ──────────────────────────────────────────────────────────────

.PHONY: db-sync
db-sync: ## Import JSON transcripts into the DB
	@$(PYTHON) -c "from whisper_tm.config import config; config.load(); \
		from whisper_tm.database import Database; \
		db = Database(config.db_path); db.initialize(); \
		r = db.sync_from_json_dir(config.transcript_dir); \
		print(f\"Imported: {r['imported']}, Skipped: {r['skipped']}, Errors: {r['errors']}\")"

.PHONY: db-rebuild
db-rebuild: ## Rebuild the FTS index
	@$(PYTHON) -c "from whisper_tm.config import config; config.load(); \
		from whisper_tm.database import Database; \
		db = Database(config.db_path); db.initialize(); \
		db.rebuild_fts_index(); print('FTS index rebuilt.')"
