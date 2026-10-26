#!/bin/bash
# Start script for Telesense backend using conda environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Activate conda environment (override with CONDA_ENV=<name>)
CONDA_ENV="${CONDA_ENV:-fantasy}"
echo "Activating conda environment '$CONDA_ENV'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV'"
    echo "Please ensure the environment exists: conda env list"
    exit 1
fi

# Navigate to the repo root
cd "$REPO_DIR"

# Set PYTHONPATH to include the repo root and src/
export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$PYTHONPATH"

# Set OpenMP workaround if needed (for any libraries that might have conflicts)
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import fastapi, uvicorn, transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    cd "$SCRIPT_DIR/backend"
    pip install -r requirements_conda.txt
    cd "$REPO_DIR"
fi

# Run the server
echo "Starting Telesense backend server..."
echo "Backend will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""
# Localhost only: the API has no authentication
python -m uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
