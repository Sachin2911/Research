SHELL := /bin/bash

.PHONY: setup install login model prompt run clean #Declaration that tells make that these are commands

# Installing UV 
setup:
	apt-get install -y btop


	curl -LsSf https://astral.sh/uv/install.sh | sh
	. ~/.local/bin/env && uv python install 3.13.12
	. ~/.local/bin/env && uv python pin 3.13.12
	. ~/.local/bin/env && uv sync


	git config --global user.name "Sachin"
	git config --global user.email "sachinmohan29@gmail.com"


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