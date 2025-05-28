#!/bin/bash

# Self-chmod if not already executable (future-proof)
if [ ! -x "$0" ]; then
  echo "🔐 Making script executable for future use..."
  chmod +x "$0"
fi

# === Step 1: Detect the latest Python version installed ===
PYTHON=$(command -v python3 || command -v python)

if [ -z "$PYTHON" ]; then
  echo "❌ Python is not installed."
  exit 1
fi

echo "✅ Using Python: $PYTHON"

# === Step 2: Create virtual environment ===
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  $PYTHON -m venv venv
else
  echo "✔️  venv already exists. Skipping creation."
fi

# === Step 3: Activate the virtual environment ===
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# === Step 4: Upgrade pip ===
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# === Step 5: Install from requirements.txt if it exists ===
if [ -f "requirements.txt" ]; then
  echo "📄 Installing requirements..."
  pip install -r requirements.txt
else
  echo "⚠️  No requirements.txt found. Skipping package installation."
fi

# === Step 6: Start Ollama with llama3 model ===
if command -v ollama &> /dev/null; then
  echo "🤖 Ollama is installed."

  # Pull llama3 model if not already present
  if ollama list | grep -q llama3; then
    echo "📦 LLaMA3 model already available."
  else
    echo "⬇️ Pulling LLaMA3 model..."
    ollama pull llama3
  fi

  # Start the llama3 model if not already running
  if pgrep -f "ollama run llama3" > /dev/null; then
    echo "🟢 LLaMA3 is already running."
  else
    echo "🚀 Starting LLaMA3 model in background..."
    nohup ollama run llama3 > llama3.log 2>&1 &
    sleep 3
    echo "✅ LLaMA3 model started."
  fi

else
  echo "❌ Ollama is not installed. Please install from https://ollama.com before running this project."
fi

echo "✅ Environment setup complete."
