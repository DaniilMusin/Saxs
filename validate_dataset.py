#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset validation script - checks generated SAXS data quality and completeness
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

def check_file_counts(dataset_dir):
    """Check if we have the expected number of files"""
    print("=== FILE COUNT VALIDATION ===")
    saxs_dir = dataset_dir / "saxs"
    
    if not saxs_dir.exists():
        print("ERROR: SAXS directory not found!")
        return False
    
    dat_files = list(saxs_dir.glob("*.dat"))
    print(f"Total .dat files found: {len(dat_files)}")
    
    # Check expected count (17 shapes × 5000 = 85,000)
    expected = 17 * 5000
    if len(dat_files) == expected:
        print(f"✓ File count matches expected: {expected}")
        return True
    else:
        print(f"WARNING: File count mismatch: expected {expected}, found {len(dat_files)}")
        return False

def analyze_filenames(dataset_dir):
    """Analyze filename patterns and extract shape distribution"""
    print("\n=== FILENAME ANALYSIS ===")
    saxs_dir = dataset_dir / "saxs"
    dat_files = list(saxs_dir.glob("*.dat"))
    
    shape_pattern = r'\d+__([^_]+)__.*\.dat'
    shape_counts = Counter()
    
    for file in dat_files:
        match = re.search(shape_pattern, file.name)
        if match:
            shape = match.group(1)
            shape_counts[shape] += 1
        else:
            print(f"⚠️  Unrecognized filename format: {file.name}")
    
    print(f"📊 Shape distribution:")
    total_shapes = len(shape_counts)
    for shape, count in sorted(shape_counts.items()):
        print(f"  {shape}: {count} samples")
    
    # Check if all shapes have 5000 samples
    expected_per_shape = 5000
    all_correct = True
    for shape, count in shape_counts.items():
        if count != expected_per_shape:
            print(f"⚠️  {shape}: expected {expected_per_shape}, got {count}")
            all_correct = False
    
    if all_correct and total_shapes == 17:
        print(f"✅ All {total_shapes} shapes have exactly {expected_per_shape} samples each")
    else:
        print(f"❌ Shape distribution issues found")
    
    return shape_counts

def validate_metadata(dataset_dir):
    """Validate metadata file consistency"""
    print("\n=== METADATA VALIDATION ===")
    meta_file = dataset_dir / "meta.csv"
    
    if not meta_file.exists():
        print("❌ meta.csv not found!")
        return None
    
    df = pd.read_csv(meta_file)
    print(f"📄 Metadata rows: {len(df)}")
    print(f"📋 Metadata columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['uid', 'shape_class', 'true_params', 'generator']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    else:
        print("✅ All required columns present")
    
    # Check shape distribution in metadata
    shape_counts = df['shape_class'].value_counts()
    print(f"📊 Metadata shape distribution:")
    for shape, count in sorted(shape_counts.items()):
        print(f"  {shape}: {count}")
    
    return df

def validate_saxs_files(dataset_dir, sample_size=10):
    """Validate sample SAXS .dat files"""
    print(f"\n=== SAXS FILES VALIDATION (sampling {sample_size} files) ===")
    saxs_dir = dataset_dir / "saxs"
    dat_files = list(saxs_dir.glob("*.dat"))
    
    if len(dat_files) == 0:
        print("❌ No .dat files found!")
        return
    
    # Sample random files
    sample_files = np.random.choice(dat_files, min(sample_size, len(dat_files)), replace=False)
    
    valid_files = 0
    issues = []
    
    for file in sample_files:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines
            data_lines = [line.strip() for line in lines[2:] if line.strip() and not line.startswith('Sample')]
            
            if len(data_lines) < 50:  # Expect at least 50 data points
                issues.append(f"{file.name}: Too few data points ({len(data_lines)})")
                continue
            
            # Try to parse first few data lines
            valid_data = 0
            for line in data_lines[:10]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        q, I, err = float(parts[0]), float(parts[1]), float(parts[2])
                        if q > 0 and I > 0 and err > 0:
                            valid_data += 1
                    except ValueError:
                        pass
            
            if valid_data >= 8:  # At least 8/10 lines should be valid
                valid_files += 1
            else:
                issues.append(f"{file.name}: Invalid data format")
                
        except Exception as e:
            issues.append(f"{file.name}: Error reading file - {e}")
    
    print(f"✅ Valid files: {valid_files}/{len(sample_files)}")
    if issues:
        print("⚠️  Issues found:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues)-5} more issues")

def validate_parameters(dataset_dir):
    """Validate that parameters are within expected ranges"""
    print("\n=== PARAMETER VALIDATION ===")
    meta_file = dataset_dir / "meta.csv"
    
    if not meta_file.exists():
        print("❌ meta.csv not found!")
        return
    
    df = pd.read_csv(meta_file)
    
    # Parameter ranges from specification
    ranges = {
        'radius': (25, 500),
        'outer_radius': (25, 500), 
        'inner_radius': (0, 500),
        'length': (250, 10000),  # 10-20 * radius
        'a': (25, 500),
        'b': (25, 500),
        'c': (25, 500),
        'core_radius': (1, 499),
        'shell_thickness': (1, 499),
        'r1': (25, 500),
        'r2': (25, 500),
        'center_distance': (1, 2000),
        'bilayer_thickness': (1, 100),
        'rmemb': (10, 250),
        'rtail': (1, 100),
        'rhead': (1, 100),
        'delta': (1, 100),
        'zcorona': (0, 200)
    }
    
    issues = []
    
    for idx, row in df.sample(min(1000, len(df))).iterrows():  # Sample for speed
        try:
            params = json.loads(row['true_params'])
            for param_name, value in params.items():
                if param_name in ranges:
                    min_val, max_val = ranges[param_name]
                    if not (min_val <= value <= max_val):
                        issues.append(f"UID {row['uid']}: {param_name}={value:.1f} outside range [{min_val}, {max_val}]")
        except json.JSONDecodeError:
            issues.append(f"UID {row['uid']}: Invalid JSON in true_params")
    
    if not issues:
        print("✅ All sampled parameters within expected ranges")
    else:
        print(f"⚠️  Parameter issues found ({len(issues)} total):")
        for issue in issues[:5]:
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues)-5} more issues")

def create_summary_plot(dataset_dir):
    """Create summary visualization"""
    print("\n=== CREATING SUMMARY PLOT ===")
    meta_file = dataset_dir / "meta.csv"
    
    if not meta_file.exists():
        print("❌ meta.csv not found!")
        return
    
    df = pd.read_csv(meta_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Shape distribution
    shape_counts = df['shape_class'].value_counts()
    axes[0,0].bar(range(len(shape_counts)), shape_counts.values)
    axes[0,0].set_xticks(range(len(shape_counts)))
    axes[0,0].set_xticklabels(shape_counts.index, rotation=45, ha='right')
    axes[0,0].set_title('Shape Distribution')
    axes[0,0].set_ylabel('Count')
    
    # Parameter example (radius distribution for spheres)
    sphere_data = df[df['shape_class'] == 'sphere']['true_params'].apply(json.loads)
    if len(sphere_data) > 0:
        radii = [p.get('radius', 0) for p in sphere_data]
        axes[0,1].hist(radii, bins=50, alpha=0.7)
        axes[0,1].set_title('Sphere Radius Distribution')
        axes[0,1].set_xlabel('Radius (Å)')
        axes[0,1].set_ylabel('Count')
    
    # Generator distribution  
    gen_counts = df['generator'].value_counts()
    axes[1,0].pie(gen_counts.values, labels=gen_counts.index, autopct='%1.1f%%')
    axes[1,0].set_title('Generator Distribution')
    
    # Sample SAXS curve
    saxs_dir = dataset_dir / "saxs"
    sample_file = list(saxs_dir.glob("*sphere*.dat"))[0] if list(saxs_dir.glob("*sphere*.dat")) else None
    if sample_file:
        try:
            with open(sample_file, 'r') as f:
                lines = f.readlines()
            data_lines = [line.strip() for line in lines[2:] if line.strip()]
            
            q_vals, I_vals = [], []
            for line in data_lines[:100]:  # First 100 points
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        q, I = float(parts[0]), float(parts[1])
                        q_vals.append(q)
                        I_vals.append(I)
                    except ValueError:
                        continue
            
            if q_vals and I_vals:
                axes[1,1].loglog(q_vals, I_vals, 'b-', alpha=0.7)
                axes[1,1].set_title(f'Sample SAXS Curve\n{sample_file.name[:30]}...')
                axes[1,1].set_xlabel('q (Å⁻¹)')
                axes[1,1].set_ylabel('I(q)')
        except:
            axes[1,1].text(0.5, 0.5, 'Could not load\nSAXS curve', ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plot_path = dataset_dir / "validation_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Summary plot saved: {plot_path}")
    plt.close()

def main():
    # Check both labeled and demo datasets
    datasets = [
        Path("dataset_all_shapes_5k_labeled"),
        Path("dataset_all_shapes_5k_demo") 
    ]
    
    for dataset_dir in datasets:
        if not dataset_dir.exists():
            print(f"⚠️  Dataset not found: {dataset_dir}")
            continue
            
        print(f"\n{'='*60}")
        print(f"VALIDATING DATASET: {dataset_dir}")
        print(f"{'='*60}")
        
        check_file_counts(dataset_dir)
        analyze_filenames(dataset_dir)
        validate_metadata(dataset_dir)
        validate_saxs_files(dataset_dir)
        validate_parameters(dataset_dir)
        create_summary_plot(dataset_dir)
        
        print(f"\n✅ Validation complete for {dataset_dir}")

if __name__ == "__main__":
    main()