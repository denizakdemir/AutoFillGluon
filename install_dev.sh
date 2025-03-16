#!/bin/bash
# Script to install AutoFillGluon in development mode

# Create and activate a virtual environment (optional)
# python -m venv venv
# source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package in development mode
pip install -e .

# Run tests to verify installation
python -m unittest discover tests

echo "Installation complete! If all tests passed, the package is ready to use."