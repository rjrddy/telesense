#!/bin/bash
# Start script for Telesense frontend

cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start development server
echo "Starting Telesense frontend..."
npm run dev
