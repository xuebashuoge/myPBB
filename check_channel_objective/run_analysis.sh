#!/bin/bash

# Script to run channel vs vanilla analysis in torch28 conda environment

echo "=========================================="
echo "Channel vs Vanilla Objective Analysis"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating conda environment: torch28"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch28

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'torch28'"
    echo "Please make sure the environment exists:"
    echo "  conda env list"
    exit 1
fi

echo "Environment activated successfully"
echo ""

# Install required packages if needed
echo "Checking required packages..."
pip install -q pandas matplotlib seaborn tabulate 2>/dev/null

# Run the analysis
echo ""
echo "Running analysis..."
echo ""

cd "$(dirname "$0")"
python analyze_channel_vs_vanilla.py

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo ""
echo "Results saved in: $(pwd)"
echo ""
echo "Generated files:"
echo "  - all_comparisons.csv"
echo "  - successful_improvements.csv"
echo "  - summary_improvements.png"
echo "  - channel_spec_effects.png"
echo "  - parameter_effects.png"
echo "  - bound_validation.png"
echo "  - ANALYSIS_REPORT.md"
