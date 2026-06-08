#!/bin/bash

# This script runs after the container is created.
# The 'set -e' command ensures that the script will exit immediately if a command fails.
set -e

echo "--- Running post-create script ---"
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y ripgrep vim python3.13-venv python3-pip

# Install anti-gravity
if [ ! -f /home/vscode/.local/bin/agy ]; then
    echo "Installing anti-gravity CLI..."
    curl -fsSL https://antigravity.google/cli/install.sh | bash
fi

# Activating the virtual environment
echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip

# Install Python dependencies from requirements.txt
echo "Installing requirements..."
find . -name "requirements.txt" -exec ./.venv/bin/pip install -r {} \;

if [ -f "pyproject.toml" ]; then
    echo "Installing project in editable mode..."
    ./.venv/bin/pip install -e .
fi