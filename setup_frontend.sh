#!/bin/bash
# Setup script for Face Recognition System Frontend

echo "================================================"
echo "Face Recognition System - Frontend Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )[^ ]*')
echo "Python version: $python_version"

if [[ ! "$python_version" > "3.8" ]]; then
    echo "Error: Python 3.8+ required"
    exit 1
fi

echo ""
echo "Installing dependencies..."
pip install -r requirements_frontend.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run the application:"
echo "  python run_frontend.py"
echo ""
echo "Or directly:"
echo "  python -m frontend.main_window"
echo ""
