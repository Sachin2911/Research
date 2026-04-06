SHELL := /bin/bash

.PHONY: setup install login model prompt run clean #Declaration that tells make that these are commands

setup:
# 	Installing useful software
	apt-get install -y btop nvtop

#   Installing uv, python and then all the packages
	curl -LsSf https://astral.sh/uv/install.sh | sh
	. ~/.local/bin/env && uv python install 3.13.12
	. ~/.local/bin/env && uv python pin 3.13.12
	. ~/.local/bin/env && uv sync

#   Configuring the git credentials
	git config --global user.name "Sachin"
	git config --global user.email "sachinmohan29@gmail.com"

	@read -p "Enter GitHub token: " token; \
	git remote set-url origin https://Sachin2911:$$token@github.com/Sachin2911/Research.git
#   HuggingFace login
	@read -p "Enter HuggingFace token: " hf_token; \
	. ~/.local/bin/env && uv run hf auth login --token $$hf_token

# Install the environment
install:
	uv sync

# Run a quick inference with a test prompt
prompt:
	uv run python src/model_test.py

# Deleting the environment
clean:
	rm -rf .venv __pycache__ src/__pycache__