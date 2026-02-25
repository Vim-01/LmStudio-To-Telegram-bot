#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
LOG_DIR="$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_$(date +%Y-%m-%d_%H-%M-%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo ">>> [$(date)] Arch-AI Auto-Launcher Started"
echo ">>> Working Directory: $PROJECT_DIR"

# ✅ ПЕРЕСОЗДАЕМ VENV (важно для Python 3.14!)
if [ -d "venv" ]; then
    echo ">>> Removing old venv (Python version may have changed)..."
    rm -rf venv
fi

echo ">>> Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ">>> Installing/updating Python dependencies..."
pip install --upgrade pip
pip install -q -r requirements.txt

# --- LM STUDIO CHECK ---
echo ">>> Checking LM Studio Server on port 1234..."

check_server() {
    curl -s --max-time 3 \
        -H "Authorization: Bearer ${LM_API_TOKEN:-lm-studio}" \
        http://localhost:1234/v1/models > /dev/null 2>&1
}

if check_server; then
    echo "✅ LM Studio Server is ALREADY RUNNING."
else
    echo "⚠️ Server is DOWN. Please start LM Studio manually."
    exit 1
fi

# --- BOT START ---
echo ">>> Starting Telegram Bot..."
trap "echo '>>> Stopping bot...'; exit" SIGINT SIGTERM

python bot.py

echo ">>> Bot process stopped."
