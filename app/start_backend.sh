#!/bin/bash
# Start script for Telesense backend

cd "$(dirname "$0")/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Run the server from the repo root so the app package is importable
cd ../..
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"
echo "Starting Telesense backend server..."
# Localhost only: the API has no authentication
python -m uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
