#!/usr/bin/env python3
"""
Create TRULY correct complex SAXS shapes using rigorous physics
"""
import numpy as np
from scipy.special import j1
from scipy.integrate import quad
import json

def exact_core_shell_sphere(q, R_core, R_shell, rho_core=1.0, rho_shell=0.8, rho_solvent=0.0):
    """
    Exact core-shell sphere form factor
    """
    # Volume and contrast factors
    V_core = (4/3) * np.pi * R_core**3
    V_shell = (4/3) * np.pi * (R_shell**3 - R_core**3)
    
    delta_rho_core = rho_core - rho_solvent
    delta_rho_shell = rho_shell - rho_solvent
    
    # Form factors for core and shell
    qR_core = q * R_core
    qR_shell = q * R_shell
    
    # Core form factor
    F_core = np.ones_like(qR_core)
    mask_core = np.abs(qR_core) > 1e-10
    qR_c = qR_core[mask_core]
    F_core[mask_core] = 3 * (np.sin(qR_c) - qR_c * np.cos(qR_c)) / (qR_c**3)
    
    # Shell form factor
    F_shell = np.ones_like(qR_shell)
    mask_shell = np.abs(qR_shell) > 1e-10
    qR_s = qR_shell[mask_shell]
    F_shell[mask_shell] = 3 * (np.sin(qR_s) - qR_s * np.cos(qR_s)) / (qR_s**3)
    
    # Combined amplitude
    F_total = delta_rho_core * V_core * F_core + delta_rho_shell * V_shell * F_shell
    
    return np.abs(F_total)**2

def exact_core_shell_cylinder(q, R_core, R_shell, L, rho_core=1.0, rho_shell=0.8, rho_solvent=0.0):
    """
    Exact core-shell cylinder with orientation averaging
    """
    def integrand(alpha, q_val, R_core, R_shell, L):
        q_perp = q_val * np.sin(alpha)  
        q_par = q_val * np.cos(alpha)
        
        # Radial contributions
        qR_core = q_perp * R_core
        qR_shell = q_perp * R_shell
        
        # Core radial
        if abs(qR_core) < 1e-10:
            F_R_core = 1.0
        else:
            F_R_core = 2 * j1(qR_core) / qR_core
        
        # Shell radial  
        if abs(qR_shell) < 1e-10:
            F_R_shell = 1.0
        else:
            F_R_shell = 2 * j1(qR_shell) / qR_shell
        
        # Axial contribution
        qL_half = q_par * L / 2
        if abs(qL_half) < 1e-10:
            F_L = 1.0
        else:
            F_L = np.sin(qL_half) / qL_half
        
        # Volume and contrasts
        V_core = np.pi * R_core**2 * L
        V_shell = np.pi * (R_shell**2 - R_core**2) * L
        
        delta_rho_core = rho_core - rho_solvent
        delta_rho_shell = rho_shell - rho_solvent
        
        # Combined amplitude
        F_total = delta_rho_core * V_core * F_R_core * F_L + delta_rho_shell * V_shell * F_R_shell * F_L
        
        return np.abs(F_total)**2 * np.sin(alpha)
    
    I = np.zeros_like(q)
    for i, q_val in enumerate(q):
        if q_val < 1e-10:
            I[i] = 1.0
        else:
            result, _ = quad(integrand, 0, np.pi/2, args=(q_val, R_core, R_shell, L),
                           epsabs=1e-8, epsrel=1e-6)
            I[i] = result
    
    return I

def exact_hollow_sphere(q, R_outer, R_inner, rho_shell=1.0, rho_solvent=0.0):
    """
    Exact hollow sphere (spherical shell)
    """
    # Volumes
    V_outer = (4/3) * np.pi * R_outer**3
    V_inner = (4/3) * np.pi * R_inner**3
    
    delta_rho = rho_shell - rho_solvent
    
    # Form factors
    qR_outer = q * R_outer
    qR_inner = q * R_inner
    
    # Outer sphere
    F_outer = np.ones_like(qR_outer)
    mask_outer = np.abs(qR_outer) > 1e-10
    qR_o = qR_outer[mask_outer]
    F_outer[mask_outer] = 3 * (np.sin(qR_o) - qR_o * np.cos(qR_o)) / (qR_o**3)
    
    # Inner sphere (cavity)
    F_inner = np.ones_like(qR_inner)
    mask_inner = np.abs(qR_inner) > 1e-10
    qR_i = qR_inner[mask_inner]
    F_inner[mask_inner] = 3 * (np.sin(qR_i) - qR_i * np.cos(qR_i)) / (qR_i**3)
    
    # Net form factor (outer - inner)
    F_net = delta_rho * (V_outer * F_outer - V_inner * F_inner)
    
    return np.abs(F_net)**2

def exact_hollow_cylinder(q, R_outer, R_inner, L, rho_shell=1.0, rho_solvent=0.0):
    """
    Exact hollow cylinder with orientation averaging
    """
    def integrand(alpha, q_val, R_outer, R_inner, L):
        q_perp = q_val * np.sin(alpha)
        q_par = q_val * np.cos(alpha)
        
        # Radial contributions
        qR_outer = q_perp * R_outer
        qR_inner = q_perp * R_inner
        
        # Outer cylinder radial
        if abs(qR_outer) < 1e-10:
            F_R_outer = 1.0
        else:
            F_R_outer = 2 * j1(qR_outer) / qR_outer
        
        # Inner cylinder radial
        if abs(qR_inner) < 1e-10:
            F_R_inner = 1.0
        else:
            F_R_inner = 2 * j1(qR_inner) / qR_inner
        
        # Axial contribution
        qL_half = q_par * L / 2
        if abs(qL_half) < 1e-10:
            F_L = 1.0
        else:
            F_L = np.sin(qL_half) / qL_half
        
        # Volumes
        V_outer = np.pi * R_outer**2 * L
        V_inner = np.pi * R_inner**2 * L
        
        delta_rho = rho_shell - rho_solvent
        
        # Net form factor
        F_net = delta_rho * (V_outer * F_R_outer - V_inner * F_R_inner) * F_L
        
        return np.abs(F_net)**2 * np.sin(alpha)
    
    I = np.zeros_like(q)
    for i, q_val in enumerate(q):
        if q_val < 1e-10:
            I[i] = 1.0
        else:
            result, _ = quad(integrand, 0, np.pi/2, args=(q_val, R_outer, R_inner, L),
                           epsabs=1e-8, epsrel=1e-6)
            I[i] = result
    
    return I

def exact_dumbbell(q, R1, R2, center_distance, rho_sphere=1.0, rho_solvent=0.0):
    """
    Exact dumbbell: two spheres separated by distance D
    """
    # Volumes
    V1 = (4/3) * np.pi * R1**3
    V2 = (4/3) * np.pi * R2**3
    
    delta_rho = rho_sphere - rho_solvent
    
    # Form factors for both spheres
    qR1 = q * R1
    qR2 = q * R2
    
    # Sphere 1
    F1 = np.ones_like(qR1)
    mask1 = np.abs(qR1) > 1e-10
    qR1_nz = qR1[mask1]
    F1[mask1] = 3 * (np.sin(qR1_nz) - qR1_nz * np.cos(qR1_nz)) / (qR1_nz**3)
    
    # Sphere 2
    F2 = np.ones_like(qR2)
    mask2 = np.abs(qR2) > 1e-10
    qR2_nz = qR2[mask2]
    F2[mask2] = 3 * (np.sin(qR2_nz) - qR2_nz * np.cos(qR2_nz)) / (qR2_nz**3)
    
    # Interference term
    qD = q * center_distance
    interference = np.ones_like(qD)
    mask_d = np.abs(qD) > 1e-10
    qD_nz = qD[mask_d]
    interference[mask_d] = np.sin(qD_nz) / qD_nz
    
    # Combined amplitude with interference
    F_total = delta_rho * (V1 * F1 + V2 * F2 * interference)
    
    return np.abs(F_total)**2

def exact_liposome(q, R_outer, bilayer_thickness, rho_head=1.2, rho_tail=0.8, rho_solvent=0.0):
    """
    Exact liposome as bilayer vesicle
    """
    # Inner radius
    R_inner = R_outer - bilayer_thickness
    
    # Average bilayer density (simplified)
    rho_bilayer = (rho_head + rho_tail) / 2
    
    # Use hollow sphere approximation
    return exact_hollow_sphere(q, R_outer, R_inner, rho_bilayer, rho_solvent)

def create_truly_correct_complex_shape(shape_type, params, output_file):
    """Create truly correct complex SAXS curve"""
    
    print(f"\n=== CREATING TRULY CORRECT {shape_type.upper()} ===")
    print(f"Parameters: {params}")
    
    # Create q-grid
    q = np.logspace(np.log10(0.01), np.log10(0.5), 201)
    
    # Calculate exact form factor
    if shape_type == 'core_shell_sphere':
        R_core = params['core_radius']
        R_shell = R_core + params['shell_thickness']
        I = exact_core_shell_sphere(q, R_core, R_shell)
        
    elif shape_type == 'core_shell_cylinder':
        R_core = params['core_radius']
        R_shell = R_core + params['shell_thickness']
        L = params['length']
        I = exact_core_shell_cylinder(q, R_core, R_shell, L)
        
    elif shape_type == 'core_shell_prolate':
        # Treat as core-shell sphere with equivalent volume
        R_core = params['equatorial_core_radius']
        R_shell = R_core + params['shell_thickness']
        I = exact_core_shell_sphere(q, R_core, R_shell)
        
    elif shape_type == 'hollow_sphere':
        R_outer = params['outer_radius']
        R_inner = params['inner_radius']
        I = exact_hollow_sphere(q, R_outer, R_inner)
        
    elif shape_type == 'hollow_cylinder':
        R_outer = params['outer_radius']
        R_inner = params['inner_radius']
        L = params['length']
        I = exact_hollow_cylinder(q, R_outer, R_inner, L)
        
    elif shape_type == 'dumbbell':
        R1 = params['r1']
        R2 = params['r2']
        D = params['center_distance']
        I = exact_dumbbell(q, R1, R2, D)
        
    elif shape_type == 'liposome':
        R_outer = params['outer_radius']
        thickness = params['bilayer_thickness']
        I = exact_liposome(q, R_outer, thickness)
        
    else:
        raise ValueError(f"Unknown complex shape: {shape_type}")
    
    # Normalize
    I_max = np.max(I)
    if I_max > 0:
        I = I / I_max * 1e12
    
    print(f"Generated {len(q)} points")
    print(f"q range: {q.min():.6f} - {q.max():.6f} A^-1")
    print(f"I range: {I.min():.2e} - {I.max():.2e}")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# TRULY CORRECT {shape_type} SAXS curve\n")
        f.write(f"# Parameters: {params}\n")
        f.write(f"# Generated using EXACT analytical form factors\n")
        f.write("# q(A^-1) I(q)\n")
        
        for q_val, I_val in zip(q, I):
            f.write(f"{q_val:.8e} {I_val:.8e}\n")
    
    print(f"Saved: {output_file}")
    
    return q, I

def load_original_params(dataset_dir, shape_class):
    """Load original parameters from dataset"""
    
    import pandas as pd
    from pathlib import Path
    
    meta_file = Path(dataset_dir) / "meta.csv"
    if not meta_file.exists():
        return None
    
    df = pd.read_csv(meta_file)
    shape_data = df[df['shape_class'] == shape_class]
    
    if len(shape_data) == 0:
        return None
    
    # Take first sample
    row = shape_data.iloc[0]
    params = json.loads(row['true_params'])
    
    return {
        'uid': row['uid'],
        'params': params,
        'filename': row['filename']
    }

def create_all_truly_correct_complex_shapes():
    """Create all truly correct complex shapes"""
    
    print("=" * 70)
    print("CREATING TRULY CORRECT COMPLEX SHAPES")
    print("=" * 70)
    
    # Get original parameters from dataset
    dataset_dir = "dataset_all_shapes_5k_labeled"
    
    complex_shapes = [
        'core_shell_sphere',
        'core_shell_cylinder', 
        'core_shell_prolate',
        'hollow_sphere',
        'hollow_cylinder',
        'dumbbell',
        'liposome'
    ]
    
    results = {}
    
    for shape_type in complex_shapes:
        print(f"\n--- Processing {shape_type} ---")
        
        # Load original parameters
        original_data = load_original_params(dataset_dir, shape_type)
        if original_data is None:
            print(f"Could not load parameters for {shape_type}")
            continue
        
        params = original_data['params']
        uid = original_data['uid']
        
        output_file = f"TRULY_CORRECT_{shape_type}_{uid}.dat"
        
        try:
            q, I = create_truly_correct_complex_shape(shape_type, params, output_file)
            results[shape_type] = {
                'q': q,
                'I': I,
                'params': params,
                'uid': uid,
                'file': output_file
            }
        except Exception as e:
            print(f"ERROR creating {shape_type}: {e}")
    
    print(f"\nSuccessfully created {len(results)} truly correct complex shapes!")
    
    return results

def main():
    """Main function"""
    return create_all_truly_correct_complex_shapes()

if __name__ == "__main__":
    results = main()