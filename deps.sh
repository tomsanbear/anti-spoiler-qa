#!/usr/bin/env bash -e

# Setup python
pyenv install -s

# Setup virtual environment
pyenv exec python -m venv env

# Install dev dependencies
env/bin/pip install -U -r requirements-dev.txt

# Generate requirements.txt to ensure we stay up to date
env/bin/pip-compile

# Install dependencies
env/bin/pip install -U -r requirements.txt