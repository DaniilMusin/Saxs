#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick dataset validation - simple checks without emoji
"""
import json
from pathlib import Path
import pandas as pd
import re
from collections import Counter

def check_dataset(dataset_dir):
    print(f"\nChecking dataset: {dataset_dir}")
    print("-" * 50)
    
    # 1. File count check
    saxs_dir = dataset_dir / "saxs"
    if not saxs_dir.exists():
        print("ERROR: saxs directory not found")
        return
        
    dat_files = list(saxs_dir.glob("*.dat"))
    print(f"Total files: {len(dat_files)}")
    print(f"Expected: 85,000 (17 shapes x 5,000 each)")
    
    if len(dat_files) == 85000:
        print("OK: File count correct")
    else:
        print("WARNING: File count mismatch")
    
    # 2. Shape distribution from filenames
    shape_pattern = r'\d+__(.+?)__.*\.dat'  # Match until second underscore (non-greedy)
    shape_counts = Counter()
    
    for file in dat_files:
        match = re.search(shape_pattern, file.name)
        if match:
            shape = match.group(1)
            shape_counts[shape] += 1
    
    print(f"\nShape distribution ({len(shape_counts)} shapes):")
    for shape, count in sorted(shape_counts.items()):
        status = "OK" if count == 5000 else "MISMATCH"
        print(f"  {shape}: {count} [{status}]")
    
    # 3. Metadata check
    meta_file = dataset_dir / "meta.csv"
    if meta_file.exists():
        df = pd.read_csv(meta_file)
        print(f"\nMetadata: {len(df)} rows")
        
        required_cols = ['uid', 'shape_class', 'true_params']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Missing columns: {missing}")
        else:
            print("OK: Required columns present")
            
        # Check a few parameter examples
        print("\nParameter examples:")
        for shape in list(shape_counts.keys())[:3]:
            shape_data = df[df['shape_class'] == shape].iloc[0]
            try:
                params = json.loads(shape_data['true_params'])
                param_str = ", ".join([f"{k}={v:.1f}" for k, v in list(params.items())[:3]])
                print(f"  {shape}: {param_str}")
            except:
                print(f"  {shape}: Error parsing parameters")
    else:
        print("ERROR: meta.csv not found")
    
    # 4. Sample file content check
    print("\nSample file check:")
    sample_files = dat_files[:3] if dat_files else []
    
    for file in sample_files:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            # Count data lines (skip headers)
            data_lines = [l for l in lines[2:] if l.strip() and not l.startswith('Sample')]
            
            if len(data_lines) >= 50:
                # Try to parse first data line
                parts = data_lines[0].split()
                if len(parts) >= 3:
                    q, I, err = float(parts[0]), float(parts[1]), float(parts[2])
                    print(f"  {file.name[:40]}... OK ({len(data_lines)} points)")
                else:
                    print(f"  {file.name[:40]}... Bad format")
            else:
                print(f"  {file.name[:40]}... Too few points ({len(data_lines)})")
                
        except Exception as e:
            print(f"  {file.name[:40]}... ERROR: {e}")

def main():
    datasets = [
        Path("dataset_all_shapes_5k_labeled"),
        Path("dataset_all_shapes_5k_demo")
    ]
    
    for dataset in datasets:
        if dataset.exists():
            check_dataset(dataset)
        else:
            print(f"\nDataset not found: {dataset}")

if __name__ == "__main__":
    main()