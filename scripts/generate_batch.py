#!/usr/bin/env python3
"""
generate_batch.py - Process a batch of SAXS curves
==================================================

This script is called by run_parallel.sh to process individual batches
of curves, allowing for better parallelization and ATSAS license management.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Import functions from main generate.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate import (
    SHAPE_SPECS, ATSASWrapper, ParameterSampler, 
    generate_single_curve
)


def process_batch(start_uid: int, end_uid: int, output_dir: Path, 
                  batch_id: int, atsas_parallel: int = 2, seed: int = 42) -> None:
    """Process a batch of UIDs"""
    
    # Setup batch-specific logging
    log_file = output_dir / 'logs' / f'batch_{batch_id:04d}.log'
    log_file.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(f'batch_{batch_id}')
    
    # Load instrument configurations
    configs = {
        'Xeuss1800HR': {
            'name': 'Xeuss1800HR',
            'config_file': Path('configs/xeuss.yml'),
            'mask_file': Path('masks/eiger1m.msk'),
            'qmin': 0.0031,
            'qmax': 0.149,
            'bins': 890
        },
        'NanoInXiderHR': {
            'name': 'NanoInXiderHR', 
            'config_file': Path('configs/nanoinx.yml'),
            'mask_file': Path('masks/eiger1m.msk'),
            'qmin': 0.0019,
            'qmax': 0.445,
            'bins': 890
        }
    }
    
    # Initialize RNG with batch-specific seed
    batch_seed = seed + batch_id * 1000
    rng = np.random.default_rng(batch_seed)
    
    # Calculate shape distribution for this batch
    n_curves = end_uid - start_uid + 1
    n_shapes = len(SHAPE_SPECS)
    shape_names = list(SHAPE_SPECS.keys())
    
    # Generate curves
    metadata_list = []
    failed_list = []
    
    logger.info(f"Processing batch {batch_id}: UIDs {start_uid} to {end_uid}")
    
    for uid in range(start_uid, end_uid + 1):
        # Determine shape class (balanced distribution)
        shape_idx = uid % n_shapes
        shape_name = shape_names[shape_idx]
        
        # Select random instrument configuration
        config_name = rng.choice(list(configs.keys()))
        config = configs[config_name]
        
        # Generate unique seed for this curve
        curve_seed = int(rng.integers(2**32))
        
        try:
            # Process single curve
            result = generate_single_curve(
                (uid, shape_name, SHAPE_SPECS[shape_name], 
                 config, output_dir, curve_seed)
            )
            
            if 'error' in result:
                failed_list.append(uid)
                logger.error(f"Failed UID {uid}: {result['error']}")
            else:
                metadata_list.append(result)
                
        except Exception as e:
            failed_list.append(uid)
            logger.error(f"Exception for UID {uid}: {str(e)}")
    
    # Save batch metadata
    if metadata_list:
        batch_meta_file = output_dir / 'temp' / f'meta_batch_{batch_id:04d}.csv'
        batch_meta_file.parent.mkdir(exist_ok=True)
        df = pd.DataFrame(metadata_list)
        df.to_csv(batch_meta_file, index=False)
        logger.info(f"Saved {len(metadata_list)} metadata entries")
    
    # Append failed UIDs to global failed file
    if failed_list:
        failed_file = output_dir / 'failed.txt'
        with open(failed_file, 'a') as f:
            for uid in failed_list:
                f.write(f"{uid}\n")
        logger.warning(f"{len(failed_list)} curves failed in this batch")
    
    logger.info(f"Batch {batch_id} complete: {len(metadata_list)} successful, "
                f"{len(failed_list)} failed")


def main():
    parser = argparse.ArgumentParser(description='Process a batch of SAXS curves')
    parser.add_argument('--start-uid', type=int, required=True,
                        help='Starting UID for this batch')
    parser.add_argument('--end-uid', type=int, required=True,
                        help='Ending UID for this batch (inclusive)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--batch-id', type=int, required=True,
                        help='Batch identifier')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--atsas-parallel', type=int, default=2,
                        help='Max parallel ATSAS processes')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    process_batch(
        start_uid=args.start_uid,
        end_uid=args.end_uid,
        output_dir=output_dir,
        batch_id=args.batch_id,
        atsas_parallel=args.atsas_parallel,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
