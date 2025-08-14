#!/usr/bin/env python3
"""
Quick verification of dataset - checks key issues and representative samples
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def quick_dataset_check():
    """Quick but thorough dataset check"""
    dataset_dir = Path("corrected_dataset_full_20250814_110527")
    
    print("="*60)
    print("QUICK DATASET VERIFICATION")
    print("="*60)
    
    # Check structure
    meta_file = dataset_dir / "meta.csv"
    saxs_dir = dataset_dir / "saxs"
    
    if not dataset_dir.exists():
        print("ERROR: Dataset directory not found!")
        return False
    
    if not meta_file.exists():
        print("ERROR: Metadata file not found!")
        return False
        
    if not saxs_dir.exists():
        print("ERROR: SAXS directory not found!")
        return False
    
    print("OK Directory structure OK")
    
    # Load metadata
    df = pd.read_csv(meta_file)
    print(f"OK Metadata loaded: {len(df)} entries")
    
    # Check file count
    saxs_files = list(saxs_dir.glob("*.dat"))
    print(f"OK SAXS files found: {len(saxs_files)}")
    
    if len(saxs_files) != len(df):
        print(f"WARNING: File count mismatch! Metadata: {len(df)}, Files: {len(saxs_files)}")
    else:
        print("OK File count matches metadata")
    
    # Check shape distribution
    shape_counts = df['shape_class'].value_counts()
    print(f"\nOK Shape types: {len(shape_counts)}")
    expected_per_shape = 5000
    
    shapes_ok = True
    for shape, count in shape_counts.items():
        if count != expected_per_shape:
            print(f"  WARNING: {shape} has {count} samples (expected {expected_per_shape})")
            shapes_ok = False
        else:
            print(f"  OK {shape}: {count} samples")
    
    if shapes_ok:
        print("OK All shapes have correct sample counts")
    
    # Sample file verification
    print(f"\nChecking sample files...")
    
    # Check each shape type with 2 samples
    sample_results = {}
    total_samples = 0
    ok_samples = 0
    
    for shape in shape_counts.index[:10]:  # Check first 10 shape types
        shape_files = df[df['shape_class'] == shape].head(2)
        
        for _, row in shape_files.iterrows():
            total_samples += 1
            filename = row['filename']
            saxs_file = saxs_dir / filename
            
            if not saxs_file.exists():
                print(f"  ERROR: File not found: {filename}")
                continue
            
            try:
                # Load file
                q, I = [], []
                with open(saxs_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line_num == 1:  # Skip header
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 2:
                            q.append(float(parts[0]))
                            I.append(float(parts[1]))
                
                # Basic checks
                q = np.array(q)
                I = np.array(I)
                
                checks = []
                checks.append(len(q) >= 50)  # Sufficient points
                checks.append(0.009 <= q.min() <= 0.02)  # Good q range start
                checks.append(0.4 <= q.max() <= 0.6)  # Good q range end
                checks.append(np.all(I > 0))  # Positive intensities
                checks.append(np.all(np.isfinite(I)))  # Finite values
                checks.append(I[0] > I[-1])  # Decaying
                checks.append(I.max() / I.min() < 1e15)  # Reasonable dynamic range
                
                if sum(checks) >= 6:  # At least 6/7 checks pass
                    ok_samples += 1
                    if shape not in sample_results:
                        sample_results[shape] = {'ok': 0, 'total': 0}
                    sample_results[shape]['ok'] += 1
                    sample_results[shape]['total'] += 1
                else:
                    print(f"  WARNING: Quality issues in {filename}")
                    print(f"    Checks passed: {sum(checks)}/7")
                
            except Exception as e:
                print(f"  ERROR: Failed to read {filename}: {e}")
    
    # Results
    print(f"\nSample verification results:")
    print(f"Files checked: {total_samples}")
    print(f"Files OK: {ok_samples}")
    sample_success_rate = (ok_samples / total_samples * 100) if total_samples > 0 else 0
    print(f"Sample success rate: {sample_success_rate:.1f}%")
    
    # Sphere accuracy check
    sphere_files = df[df['shape_class'] == 'sphere'].head(5)
    sphere_errors = []
    
    print(f"\nSphere accuracy check:")
    for _, row in sphere_files.iterrows():
        try:
            filename = row['filename']
            saxs_file = saxs_dir / filename
            
            # Get radius
            params_str = row['true_params']
            params = json.loads(params_str.replace('""', '"'))
            R = params['R']
            
            # Load data
            q, I = [], []
            with open(saxs_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num == 1:
                        continue
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            q.append(float(parts[0]))
                            I.append(float(parts[1]))
            
            q = np.array(q)
            I = np.array(I)
            
            # Find first minimum
            q_theoretical = 4.493 / R
            
            # Look for minimum in expected range
            mask = (q >= q_theoretical * 0.7) & (q <= q_theoretical * 1.3)
            if np.any(mask):
                q_region = q[mask]
                I_region = I[mask]
                min_idx = np.argmin(I_region)
                q_observed = q_region[min_idx]
                
                error_pct = abs(q_observed - q_theoretical) / q_theoretical * 100
                sphere_errors.append(error_pct)
                print(f"  {filename}: R={R:.1f}Å, error={error_pct:.1f}%")
        
        except Exception as e:
            print(f"  ERROR checking {filename}: {e}")
    
    if sphere_errors:
        avg_sphere_error = np.mean(sphere_errors)
        max_sphere_error = np.max(sphere_errors)
        print(f"  Average sphere error: {avg_sphere_error:.2f}%")
        print(f"  Maximum sphere error: {max_sphere_error:.2f}%")
        
        sphere_accuracy_ok = avg_sphere_error < 5.0
    else:
        sphere_accuracy_ok = False
        print("  No sphere accuracy data available")
    
    # Final assessment
    print(f"\n" + "="*60)
    print("QUICK VERIFICATION SUMMARY")
    print("="*60)
    
    overall_score = 0
    max_score = 5
    
    # Structure (1 point)
    if len(saxs_files) == len(df) == 85000:
        overall_score += 1
        print("OK Structure and file count: PASS")
    else:
        print("ERROR Structure and file count: FAIL")
    
    # Shape distribution (1 point)
    if shapes_ok:
        overall_score += 1
        print("OK Shape distribution: PASS")
    else:
        print("ERROR Shape distribution: FAIL")
    
    # Sample quality (2 points)
    if sample_success_rate >= 95:
        overall_score += 2
        print("OK Sample quality: EXCELLENT")
    elif sample_success_rate >= 85:
        overall_score += 1
        print("OK Sample quality: GOOD")
    else:
        print("ERROR Sample quality: POOR")
    
    # Sphere accuracy (1 point)
    if sphere_accuracy_ok:
        overall_score += 1
        print("OK Sphere accuracy: PASS")
    else:
        print("ERROR Sphere accuracy: FAIL")
    
    final_score = (overall_score / max_score) * 100
    print(f"\nOverall Score: {overall_score}/{max_score} ({final_score:.1f}%)")
    
    if final_score >= 80:
        verdict = "DATASET VERIFICATION: PASSED"
        success = True
    else:
        verdict = "DATASET VERIFICATION: FAILED"
        success = False
    
    print(f"\n{verdict}")
    
    return success, final_score

if __name__ == "__main__":
    success, score = quick_dataset_check()
    print(f"\nFinal result: {'SUCCESS' if success else 'FAILURE'} (Score: {score:.1f}%)")