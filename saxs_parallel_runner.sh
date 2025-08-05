#!/bin/bash
# run_parallel.sh - Enhanced parallel execution using GNU Parallel
# This script provides better control over ATSAS license locks

set -euo pipefail

# Default parameters
N_TOTAL=37656
N_JOBS=20
ATSAS_PARALLEL=2  # ATSAS license limit
OUTPUT_DIR="dataset"
BATCH_SIZE=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--total)
            N_TOTAL="$2"
            shift 2
            ;;
        -j|--jobs)
            N_JOBS="$2"
            shift 2
            ;;
        -a|--atsas-parallel)
            ATSAS_PARALLEL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-n total] [-j jobs] [-a atsas_parallel] [-o output_dir] [-b batch_size]"
            exit 1
            ;;
    esac
done

# Check dependencies
command -v parallel >/dev/null 2>&1 || { echo "GNU Parallel required but not found"; exit 1; }
command -v bodies >/dev/null 2>&1 || { echo "ATSAS not found in PATH"; exit 1; }

# Create output directories
mkdir -p "$OUTPUT_DIR"/{saxs,logs,temp}

# Calculate batches
N_BATCHES=$((N_TOTAL / BATCH_SIZE))
REMAINDER=$((N_TOTAL % BATCH_SIZE))

echo "Generating $N_TOTAL SAXS curves in $N_BATCHES batches of $BATCH_SIZE"
echo "Using $N_JOBS parallel jobs (limited to $ATSAS_PARALLEL for ATSAS operations)"
echo "Output directory: $OUTPUT_DIR"

# Function to process a single batch
process_batch() {
    local batch_id=$1
    local start_uid=$((batch_id * BATCH_SIZE))
    local end_uid=$((start_uid + BATCH_SIZE - 1))
    
    # Handle last batch with remainder
    if [[ $batch_id -eq $((N_BATCHES - 1)) ]] && [[ $REMAINDER -gt 0 ]]; then
        end_uid=$((start_uid + BATCH_SIZE + REMAINDER - 1))
    fi
    
    echo "Processing batch $batch_id: UIDs $start_uid to $end_uid"
    
    # Use semaphore to control ATSAS license access
    # This ensures only ATSAS_PARALLEL processes use ATSAS at once
    sem --id atsas_lock -j "$ATSAS_PARALLEL" python scripts/generate_batch.py \
        --start-uid "$start_uid" \
        --end-uid "$end_uid" \
        --output "$OUTPUT_DIR" \
        --batch-id "$batch_id" \
        --atsas-parallel "$ATSAS_PARALLEL"
}

export -f process_batch
export BATCH_SIZE N_BATCHES REMAINDER OUTPUT_DIR ATSAS_PARALLEL

# Run batches in parallel
seq 0 $((N_BATCHES - 1)) | parallel -j "$N_JOBS" process_batch {}

# Merge metadata files
echo "Merging metadata..."
python -c "
import pandas as pd
import glob
import os

output_dir = '$OUTPUT_DIR'
meta_files = glob.glob(os.path.join(output_dir, 'temp', 'meta_batch_*.csv'))
if meta_files:
    dfs = [pd.read_csv(f) for f in meta_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, 'meta.csv'), index=False)
    print(f'Merged {len(meta_files)} metadata files')
    # Clean up temporary files
    for f in meta_files:
        os.remove(f)
"

# Generate summary statistics
echo "Generating summary..."
python -c "
import pandas as pd
import os

output_dir = '$OUTPUT_DIR'
meta_file = os.path.join(output_dir, 'meta.csv')
if os.path.exists(meta_file):
    df = pd.read_csv(meta_file)
    print('\nDataset Summary:')
    print(f'Total curves: {len(df)}')
    print(f'Shape distribution:')
    print(df['shape_class'].value_counts())
    print(f'\nInstrument distribution:')
    print(df['instrument_cfg'].value_counts())
    
    # Check for failures
    failed_file = os.path.join(output_dir, 'failed.txt')
    if os.path.exists(failed_file):
        with open(failed_file, 'r') as f:
            n_failed = len(f.readlines())
        print(f'\nFailed curves: {n_failed}')
"

echo "Dataset generation complete!"
