#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo version of generator that creates mock SAXS data for all shape types.
"""
import sys
import json
import random
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from param_sampler import available_shapes, draw_shape

def create_mock_saxs_data(q_range=(0.01, 0.5), n_points=100):
    """Create mock SAXS intensity data"""
    q = np.logspace(np.log10(q_range[0]), np.log10(q_range[1]), n_points)
    # Mock Guinier-like scattering
    I = np.exp(-q**2 * random.uniform(100, 1000)) * random.uniform(1e6, 1e8)
    # Add some noise
    noise = I * 0.01 * np.random.random(len(I))
    I_noise = I + noise
    errors = np.sqrt(I_noise) + I_noise * 0.01
    return q, I_noise, errors

def main():
    print("Starting demo generation of 5k samples per shape type...")
    
    out_dir = Path("dataset_all_shapes_5k_demo")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "saxs").mkdir(exist_ok=True)
    
    shapes = available_shapes()
    print(f"Will generate 5000 samples for each of {len(shapes)} shape types")
    print(f"Output directory: {out_dir.absolute()}")
    
    meta_rows = []
    rng = random.Random(42)
    
    uid = 0
    for shape_idx, shape in enumerate(shapes):
        print(f"Generating {shape} ({shape_idx+1}/{len(shapes)})...")
        
        for i in range(5000):
            sample = draw_shape(rng, shape=shape)
            
            # Create mock SAXS data
            q, I, err = create_mock_saxs_data()
            
            # Write .dat file
            dat_file = out_dir / "saxs" / f"{uid:06d}__nanoinx.dat"
            with open(dat_file, 'w') as f:
                f.write(f'Sample description: Demo {shape} #{i+1}\n')
                f.write('Sample:   c= 1.000 mg/ml  Code: \n')
                for qi, Ii, erri in zip(q, I, err):
                    f.write(f'{qi:.6e}   {Ii:.6e}   {erri:.6e}\n')
            
            # Add to metadata
            meta_rows.append({
                "uid": uid,
                "shape_class": sample.shape_class,
                "generator": "pythonDAM+demo_saxs",
                "true_params": json.dumps(sample.params, ensure_ascii=False),
                "instrument_cfg": "nanoinx",
                "seed": 42 + uid
            })
            
            uid += 1
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/5000 samples generated")
    
    # Write metadata
    df = pd.DataFrame(meta_rows)
    meta_file = out_dir / "meta.csv"
    df.to_csv(meta_file, index=False)
    
    print("Generation complete!")
    print(f"Total samples generated: {len(meta_rows)}")
    print(f"Metadata saved to: {meta_file}")
    print(f"SAXS curves saved to: {out_dir / 'saxs'}")
    
    # Statistics
    shape_counts = df['shape_class'].value_counts()
    print("\nShape distribution:")
    for shape, count in shape_counts.items():
        print(f"  {shape}: {count} samples")

if __name__ == "__main__":
    main()