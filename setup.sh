#!/bin/bash
# setup.sh - Setup environment for SAXS dataset generation
# ========================================================

set -euo pipefail

echo "SAXS Dataset Generator - Environment Setup"
echo "=========================================="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python $PYTHON_VERSION"

# Check for ATSAS
echo -n "Checking for ATSAS tools... "
MISSING_TOOLS=()
for tool in bodies mixture crysol imsim im2dat; do
    if ! command -v $tool &> /dev/null; then
        MISSING_TOOLS+=($tool)
    fi
done

if [ ${#MISSING_TOOLS[@]} -eq 0 ]; then
    echo "OK"
else
    echo "MISSING"
    echo "The following ATSAS tools are not found in PATH:"
    printf '  - %s\n' "${MISSING_TOOLS[@]}"
    echo ""
    echo "Please install ATSAS from: https://www.embl-hamburg.de/biosaxs/software.html"
    echo "And ensure the binaries are in your PATH"
    exit 1
fi

# Create virtual environment
echo -n "Creating Python virtual environment... "
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "OK"
else
    echo "Already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install pandas numpy pyyaml tqdm matplotlib > /dev/null 2>&1
echo "Python dependencies installed"

# Create directory structure
echo "Creating directory structure..."
mkdir -p configs scripts dataset/{saxs,logs,temp}

# Make scripts executable
chmod +x scripts/*.py scripts/*.sh 2>/dev/null || true

# Check for GNU Parallel (optional)
if command -v parallel &> /dev/null; then
    echo "GNU Parallel found (recommended for large datasets)"
else
    echo "GNU Parallel not found (optional, but recommended for performance)"
    echo "Install with: sudo apt-get install parallel  # on Ubuntu/Debian"
fi

# Test ATSAS functionality
echo -n "Testing ATSAS functionality... "
if command -v bodies >/dev/null 2>&1 && command -v imsim >/dev/null 2>&1; then
    echo "OK"
else
    echo "ISSUES DETECTED"
    echo "ATSAS tools not found in PATH"
    echo "Check your ATSAS installation and ensure tools are in PATH"
fi

# Summary
echo ""
echo "Setup complete!"
echo "=============="
echo ""
echo "To generate a dataset:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Test the generator (small test):"
echo "     python scripts/generate.py --n 10 --jobs 1 --out test/"
echo ""
echo "  3. Run the generator:"
echo "     python scripts/generate.py --n 1000 --jobs 4 --atsas-parallel 2 --out dataset/"
echo ""
echo "  4. For large datasets, use the parallel runner:"
echo "     bash scripts/run_parallel.sh -n 37656 -j 20 -a 2"
echo ""
echo "  5. Check data quality:"
echo "     python scripts/check_quality.py dataset/"
echo ""
echo "  6. See interactive examples:"
echo "     python scripts/example_usage.py"
echo ""
echo "Configuration files are in: configs/"
echo "Output will be saved to: dataset/"
