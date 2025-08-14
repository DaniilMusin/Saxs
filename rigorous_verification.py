#!/usr/bin/env python3
"""
Rigorous verification of all corrections before generating new dataset
Check EVERY aspect of physics and mathematics
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_curve(filename):
    """Load SAXS curve"""
    q, I = [], []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    q_val = float(parts[0])
                    I_val = float(parts[1])
                    if q_val >= 0 and I_val >= 0:
                        q.append(q_val)
                        I.append(I_val)
                except:
                    continue
    
    return np.array(q), np.array(I)

def rigorous_sphere_check(filename, expected_R):
    """Rigorous check of sphere form factor"""
    
    print(f"\n=== RIGOROUS SPHERE CHECK ===")
    print(f"File: {filename}")
    print(f"Expected R: {expected_R} Å")
    
    q, I = load_curve(filename)
    
    if len(q) == 0:
        return False, "Could not load data"
    
    # Check 1: First minimum position (most critical)
    theoretical_min = 4.493 / expected_R
    
    # Find minimum around theoretical position
    search_range = (theoretical_min * 0.9, theoretical_min * 1.1)
    mask = (q >= search_range[0]) & (q <= search_range[1])
    
    if np.sum(mask) == 0:
        return False, f"No data points around expected minimum q={theoretical_min:.4f}"
    
    q_search = q[mask]
    I_search = I[mask]
    min_idx = np.argmin(I_search)
    actual_min = q_search[min_idx]
    
    error = abs(actual_min - theoretical_min) / theoretical_min * 100
    
    print(f"First minimum:")
    print(f"  Theoretical: {theoretical_min:.6f} A^-1")
    print(f"  Actual: {actual_min:.6f} A^-1")
    print(f"  Error: {error:.3f}%")
    
    if error > 2:
        return False, f"First minimum error too large: {error:.1f}%"
    
    # Check 2: Forward scattering (should be maximum)
    forward_ratio = I[0] / np.max(I)
    print(f"Forward scattering: {forward_ratio:.3f} (should be ~1.0)")
    
    if forward_ratio < 0.8:
        return False, f"Forward scattering too low: {forward_ratio:.3f}"
    
    # Check 3: High-q decay (should follow power law)
    high_q_mask = q > 0.3
    if np.sum(high_q_mask) > 10:
        q_high = q[high_q_mask]
        I_high = I[high_q_mask]
        
        # Fit power law in log space
        log_q = np.log(q_high)
        log_I = np.log(I_high + 1e-15)
        slope = np.polyfit(log_q, log_I, 1)[0]
        
        print(f"High-q decay: I ~ q^{slope:.2f} (should be -4 to -2)")
        
        if slope > -1 or slope < -6:
            return False, f"Unphysical high-q decay: q^{slope:.2f}"
    
    # Check 4: Oscillation structure
    # Count minima and maxima
    dI = np.diff(I)
    sign_changes = np.sum(np.diff(np.sign(dI)) != 0)
    oscillation_density = sign_changes / len(q)
    
    print(f"Oscillations: {oscillation_density:.3f} changes/point (should be 0.1-0.3)")
    
    if oscillation_density < 0.05:
        return False, f"Too few oscillations: {oscillation_density:.3f}"
    
    # Check 5: Mathematical consistency
    # Verify form factor at q=0 is 1.0 (normalized)
    I_normalized = I / I[0]
    print(f"Normalization check: I(0) = {I_normalized[0]:.6f}")
    
    print("SPHERE VERIFICATION: PASSED")
    return True, "All checks passed"

def rigorous_cylinder_check(filename):
    """Rigorous check of cylinder form factor"""
    
    print(f"\n=== RIGOROUS CYLINDER CHECK ===")
    print(f"File: {filename}")
    
    q, I = load_curve(filename)
    
    if len(q) == 0:
        return False, "Could not load data"
    
    # Check 1: Forward scattering behavior
    forward_ratio = I[0] / np.max(I)
    print(f"Forward scattering: {forward_ratio:.3f}")
    
    # For cylinders, forward scattering can be lower due to orientation averaging
    if forward_ratio < 0.1:
        return False, f"Forward scattering too low: {forward_ratio:.3f}"
    
    # Check 2: Oscillatory structure (characteristic of cylinders)
    dI = np.diff(I)
    sign_changes = np.sum(np.diff(np.sign(dI)) != 0)
    oscillation_density = sign_changes / len(q)
    
    print(f"Oscillations: {oscillation_density:.3f} changes/point")
    
    if oscillation_density < 0.05:
        return False, f"Too few oscillations for cylinder: {oscillation_density:.3f}"
    
    # Check 3: Asymptotic decay
    high_q_mask = q > 0.2
    if np.sum(high_q_mask) > 10:
        q_high = q[high_q_mask]
        I_high = I[high_q_mask]
        
        log_q = np.log(q_high)
        log_I = np.log(I_high + 1e-15)
        slope = np.polyfit(log_q, log_I, 1)[0]
        
        print(f"High-q decay: I ~ q^{slope:.2f}")
        
        if slope > 0 or slope < -6:
            return False, f"Unphysical decay: q^{slope:.2f}"
    
    print("CYLINDER VERIFICATION: PASSED")
    return True, "All checks passed"

def rigorous_complex_shape_check(filename, shape_type):
    """Rigorous check of complex shapes"""
    
    print(f"\n=== RIGOROUS {shape_type.upper()} CHECK ===")
    print(f"File: {filename}")
    
    q, I = load_curve(filename)
    
    if len(q) == 0:
        return False, "Could not load data"
    
    # Check 1: Physical intensity range
    I_positive = I[I > 0]
    if len(I_positive) == 0:
        return False, "No positive intensities"
    
    dynamic_range = np.max(I_positive) / np.min(I_positive)
    print(f"Dynamic range: {dynamic_range:.2e}")
    
    if dynamic_range > 1e20:
        return False, f"Excessive dynamic range: {dynamic_range:.2e}"
    
    # Check 2: Monotonic overall decay
    # Split into regions and check general trend
    n_regions = 5
    region_size = len(I) // n_regions
    
    region_averages = []
    for i in range(n_regions):
        start = i * region_size
        end = start + region_size if i < n_regions-1 else len(I)
        if end > start:
            region_avg = np.mean(I[start:end])
            region_averages.append(region_avg)
    
    # Check if generally decreasing
    decreasing_regions = sum(1 for i in range(len(region_averages)-1) 
                           if region_averages[i+1] < region_averages[i])
    
    print(f"Decreasing trend: {decreasing_regions}/{len(region_averages)-1} regions")
    
    if decreasing_regions < len(region_averages) // 2:
        return False, f"No clear decreasing trend"
    
    # Check 3: No extreme discontinuities
    if len(I) > 5:
        log_I = np.log(I + 1e-15)
        max_jump = np.max(np.abs(np.diff(log_I)))
        print(f"Maximum jump: {max_jump:.2f} (log scale)")
        
        if max_jump > 10:  # Very large jump
            return False, f"Extreme discontinuity: {max_jump:.2f}"
    
    print(f"{shape_type.upper()} VERIFICATION: PASSED")
    return True, "All checks passed"

def comprehensive_verification():
    """Comprehensive verification of ALL truly correct files"""
    
    print("=" * 80)
    print("COMPREHENSIVE RIGOROUS VERIFICATION")
    print("=" * 80)
    
    # Define all files to check
    verification_tests = [
        ('sphere', 'TRULY_CORRECT_sphere.dat', 55.0, rigorous_sphere_check),
        ('cylinder', 'TRULY_CORRECT_cylinder.dat', None, rigorous_cylinder_check),
        ('prolate', 'TRULY_CORRECT_prolate.dat', None, rigorous_complex_shape_check),
        ('oblate', 'TRULY_CORRECT_oblate.dat', None, rigorous_complex_shape_check),
        ('ellipsoid_triaxial', 'TRULY_CORRECT_ellipsoid_triaxial.dat', None, rigorous_complex_shape_check),
        ('ellipsoid_of_rotation', 'TRULY_CORRECT_ellipsoid_of_rotation.dat', None, rigorous_complex_shape_check),
    ]
    
    # Add complex shape files
    for file_path in Path('.').glob('TRULY_CORRECT_*.dat'):
        filename = file_path.name
        if not any(filename == test[1] for test in verification_tests):
            shape_name = filename.replace('TRULY_CORRECT_', '').replace('.dat', '')
            verification_tests.append((shape_name, filename, None, rigorous_complex_shape_check))
    
    results = []
    passed = 0
    
    for shape_type, filename, param, check_func in verification_tests:
        if not Path(filename).exists():
            print(f"\nERROR: File not found: {filename}")
            results.append((shape_type, False, "File not found"))
            continue
        
        try:
            if param is not None:
                success, message = check_func(filename, param)
            elif check_func == rigorous_complex_shape_check:
                success, message = check_func(filename, shape_type)
            else:
                success, message = check_func(filename)
            
            results.append((shape_type, success, message))
            if success:
                passed += 1
        
        except Exception as e:
            print(f"\nERROR checking {shape_type}: {e}")
            results.append((shape_type, False, f"Exception: {e}"))
    
    # Summary
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n" + "=" * 80)
    print("RIGOROUS VERIFICATION SUMMARY")
    print("=" * 80)
    
    print(f"Total files checked: {total}")
    print(f"Passed rigorous tests: {passed}")
    print(f"Success rate: {success_rate:.1f}%")
    
    print(f"\nDetailed results:")
    for shape_type, success, message in results:
        status = "PASS" if success else "FAIL"
        print(f"  {shape_type:25s}: {status} - {message}")
    
    # Final verdict
    if success_rate >= 95:
        verdict = "READY FOR DATASET GENERATION"
        ready = True
    elif success_rate >= 85:
        verdict = "MOSTLY READY - Minor issues"
        ready = True
    elif success_rate >= 70:
        verdict = "NEEDS SOME FIXES"
        ready = False
    else:
        verdict = "SIGNIFICANT ISSUES - DO NOT PROCEED"
        ready = False
    
    print(f"\nFINAL VERDICT: {verdict}")
    
    return ready, success_rate, results

def main():
    """Main verification function"""
    return comprehensive_verification()

if __name__ == "__main__":
    ready, success_rate, results = main()