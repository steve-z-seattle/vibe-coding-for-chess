#!/bin/bash
# Run all tests for the chess project

echo "Running Chess Game Tests..."
echo "============================"

cd backend

# Check if pytest is available
if ! python -c "import pytest" 2>/dev/null; then
    echo "Installing test dependencies..."
    pip install -r ../requirements.txt
fi

# Run tests
echo ""
echo "Running tests with pytest..."
python -m pytest tests/ -v

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Some tests failed!"
    exit 1
fi
