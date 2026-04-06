SHELL := /bin/bash

.PHONY: setup install login model prompt run clean #Declaration that tells make that these are commands

# Installing UV 
setup:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	source ~/.cargo/env  # or restart shell
	uv python install 3.13.12
	uv python pin 3.13.12
	uv sync


install:
	uv sync

# HuggingFace login (needed for Llama gated model)
login:
	uv run huggingface-cli login

# Download model weights ahead of time
model:
	uv run python -c "\
	from transformers import AutoModelForCausalLM, AutoTokenizer; \
	AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B'); \
	AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')"

# Run main experiment
run:
	uv run python src/cmaes_loop.py

# Run a quick inference with a test prompt
prompt:
	uv run python src/load_model.py --prompt "hey there"

clean:
	rm -rf .venv __pycache__ src/__pycache__