#!/usr/bin/env python3
"""
Detailed verification of specific shape types for physical correctness
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

def verify_complex_shapes():
    """Verify complex shapes for physical consistency"""
    dataset_dir = Path("corrected_dataset_full_20250814_110527")
    meta_file = dataset_dir / "meta.csv"
    saxs_dir = dataset_dir / "saxs"
    
    print("="*60)
    print("DETAILED SHAPE VERIFICATION")
    print("="*60)
    
    df = pd.read_csv(meta_file)
    
    # Focus on complex shapes that were problematic before
    complex_shapes = [
        'core_shell_sphere', 'core_shell_cylinder', 'core_shell_prolate',
        'hollow_cylinder', 'hollow_sphere', 'liposome', 'dumbbell'
    ]
    
    results = {}
    
    for shape in complex_shapes:
        print(f"\nVerifying {shape}...")
        shape_files = df[df['shape_class'] == shape].head(3)  # Check 3 samples
        
        shape_results = {'samples': 0, 'passed': 0, 'issues': []}
        
        for _, row in shape_files.iterrows():
            filename = row['filename']
            saxs_file = saxs_dir / filename
            shape_results['samples'] += 1
            
            try:
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
                
                # Physical consistency checks
                checks = []
                
                # 1. Forward scattering should be reasonable
                forward_ok = 0.1 < I[0] < 100
                checks.append(('Forward scattering', forward_ok))
                
                # 2. Monotonic decay overall (allowing for oscillations)
                decay_ok = I[-1] < I[0] * 1.1  # Allow 10% variation
                checks.append(('Overall decay', decay_ok))
                
                # 3. No extreme jumps in log scale
                log_I = np.log(I)
                max_jump = np.max(np.abs(np.diff(log_I)))
                jump_ok = max_jump < 5  # Less than 5 decades jump
                checks.append(('No extreme jumps', jump_ok))
                
                # 4. Reasonable dynamic range
                dynamic_range = I.max() / I.min()
                range_ok = 1 < dynamic_range < 1e12
                checks.append(('Reasonable range', range_ok))
                
                # 5. Smooth at high q (no sharp spikes)
                high_q_mask = q > 0.3
                if np.any(high_q_mask):
                    high_q_I = I[high_q_mask]
                    high_q_smooth = np.std(np.diff(np.log(high_q_I))) < 0.5
                else:
                    high_q_smooth = True
                checks.append(('High-q smoothness', high_q_smooth))
                
                # 6. Positive everywhere
                positive_ok = np.all(I > 0)
                checks.append(('All positive', positive_ok))
                
                passed_checks = sum(check[1] for check in checks)
                
                if passed_checks >= 5:  # At least 5/6 checks pass
                    shape_results['passed'] += 1
                    print(f"  {filename}: PASS ({passed_checks}/6 checks)")
                else:
                    failed_checks = [check[0] for check in checks if not check[1]]
                    shape_results['issues'].append(f"{filename}: FAIL - {failed_checks}")
                    print(f"  {filename}: FAIL ({passed_checks}/6 checks) - {failed_checks}")
                
            except Exception as e:
                shape_results['issues'].append(f"{filename}: ERROR - {str(e)}")
                print(f"  {filename}: ERROR - {str(e)}")
        
        success_rate = shape_results['passed'] / shape_results['samples'] * 100 if shape_results['samples'] > 0 else 0
        results[shape] = {
            'success_rate': success_rate,
            'samples': shape_results['samples'],
            'passed': shape_results['passed'],
            'issues': shape_results['issues']
        }
        
        print(f"  {shape}: {shape_results['passed']}/{shape_results['samples']} passed ({success_rate:.1f}%)")
    
    # Summary
    print(f"\n" + "="*60)
    print("DETAILED VERIFICATION SUMMARY")
    print("="*60)
    
    all_success_rates = [result['success_rate'] for result in results.values()]
    overall_success = np.mean(all_success_rates) if all_success_rates else 0
    
    for shape, result in results.items():
        status = "PASS" if result['success_rate'] >= 66.7 else "FAIL"
        print(f"{shape}: {result['success_rate']:.1f}% - {status}")
    
    print(f"\nOverall complex shape success: {overall_success:.1f}%")
    
    if overall_success >= 80:
        verdict = "COMPLEX SHAPES: EXCELLENT"
    elif overall_success >= 66:
        verdict = "COMPLEX SHAPES: GOOD"
    else:
        verdict = "COMPLEX SHAPES: NEEDS WORK"
    
    print(f"\n{verdict}")
    
    return overall_success >= 66

def verify_simple_shapes():
    """Verify simple shapes (sphere, cylinder) for mathematical accuracy"""
    dataset_dir = Path("corrected_dataset_full_20250814_110527")
    meta_file = dataset_dir / "meta.csv"
    saxs_dir = dataset_dir / "saxs"
    
    print(f"\n" + "="*60)
    print("SIMPLE SHAPES ACCURACY CHECK")
    print("="*60)
    
    df = pd.read_csv(meta_file)
    
    # Check spheres for exact accuracy
    print("Checking sphere accuracy...")
    sphere_files = df[df['shape_class'] == 'sphere'].head(10)
    sphere_errors = []
    
    for _, row in sphere_files.iterrows():
        try:
            filename = row['filename']
            saxs_file = saxs_dir / filename
            
            # Get parameters
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
            
            # Theoretical first minimum
            q_theoretical = 4.493 / R
            
            # Find observed minimum
            mask = (q >= q_theoretical * 0.7) & (q <= q_theoretical * 1.3)
            if np.any(mask):
                q_region = q[mask]
                I_region = I[mask]
                min_idx = np.argmin(I_region)
                q_observed = q_region[min_idx]
                
                error_pct = abs(q_observed - q_theoretical) / q_theoretical * 100
                sphere_errors.append(error_pct)
                
                status = "OK" if error_pct < 5 else "HIGH ERROR"
                print(f"  {filename}: R={R:.1f}A, error={error_pct:.2f}% - {status}")
        
        except Exception as e:
            print(f"  ERROR checking {filename}: {e}")
    
    if sphere_errors:
        avg_error = np.mean(sphere_errors)
        max_error = np.max(sphere_errors)
        print(f"\nSphere statistics:")
        print(f"  Average error: {avg_error:.2f}%")
        print(f"  Maximum error: {max_error:.2f}%")
        
        sphere_quality = "EXCELLENT" if avg_error < 1 else "GOOD" if avg_error < 3 else "ACCEPTABLE" if avg_error < 5 else "POOR"
        print(f"  Sphere accuracy: {sphere_quality}")
        
        return avg_error < 5
    
    return False

def main():
    """Main verification"""
    print("Starting detailed shape verification...")
    
    complex_ok = verify_complex_shapes()
    simple_ok = verify_simple_shapes()
    
    print(f"\n" + "="*60)
    print("FINAL DETAILED VERIFICATION RESULT")
    print("="*60)
    
    print(f"Complex shapes: {'PASS' if complex_ok else 'FAIL'}")
    print(f"Simple shapes: {'PASS' if simple_ok else 'FAIL'}")
    
    overall_ok = complex_ok and simple_ok
    print(f"\nOverall detailed verification: {'PASSED' if overall_ok else 'FAILED'}")
    
    return overall_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)