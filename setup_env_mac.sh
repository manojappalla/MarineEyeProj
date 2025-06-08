#!/bin/bash

# Configuration
PYTHON_VERSION="3.10.12"
VENV_DIR=".venv"

# Resolve the directory this script is in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📂 Project folder: $SCRIPT_DIR"
echo "🐍 Target Python version: $PYTHON_VERSION"

# -------------------------------
# Step 1: Install pyenv if missing
# -------------------------------
if ! command -v pyenv &> /dev/null; then
  echo "📥 pyenv not found. Installing..."

  # Install pyenv build dependencies (Ubuntu)
  sudo apt-get update && sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

  curl https://pyenv.run | bash

  # Load pyenv into current shell
  export PATH="$HOME/.pyenv/bin:$PATH"
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"

  echo "✅ pyenv installed."
else
  echo "✅ pyenv already installed."
  export PATH="$HOME/.pyenv/bin:$PATH"
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
fi

# -------------------------------
# Step 2: Install Python version
# -------------------------------
if ! pyenv versions --bare | grep -q "^$PYTHON_VERSION$"; then
  echo "📦 Installing Python $PYTHON_VERSION..."
  pyenv install "$PYTHON_VERSION"
else
  echo "✅ Python $PYTHON_VERSION already installed."
fi

# Set local Python version
pyenv local "$PYTHON_VERSION"
echo "$PYTHON_VERSION" > .python-version

# -------------------------------
# Step 3: Install uv if missing
# -------------------------------
if ! command -v uv &> /dev/null; then
  echo "📥 Installing uv..."
  curl -Ls https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.cargo/bin:$PATH"

# -------------------------------
# Step 4: Create and activate venv
# -------------------------------
echo "🧪 Creating virtual environment at: $SCRIPT_DIR/$VENV_DIR"
uv venv --python "$(pyenv which python)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# -------------------------------
# Step 5: Install dependencies
# -------------------------------
if [ -f requirements.txt ]; then
  echo "📦 Installing dependencies from requirements.txt..."
  uv pip install -r requirements.txt
else
  echo "⚠️ No requirements.txt found — skipping dependency installation."
fi

# -------------------------------
# Done
# -------------------------------
echo "🎉 Setup complete!"
echo "👉 Activate your environment with:"
echo "   source $SCRIPT_DIR/$VENV_DIR/bin/activate"
