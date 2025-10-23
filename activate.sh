#!/bin/bash
# Activation script for the virtual environment

echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""
echo "Development tools available:"
echo "  - pytest      : Run tests"
echo "  - ruff        : Lint and format code"
echo "  - mypy        : Type checking"
echo ""
echo "Note: Full geospatial dependencies (GDAL, rioxarray, etc.) are only available in Docker."
echo "      Use 'docker-compose up' for running the service with full functionality."
