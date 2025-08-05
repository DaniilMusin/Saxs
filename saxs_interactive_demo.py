#!/usr/bin/env python3
"""
demo_interactive.py - Demonstrates interactive use of ATSAS tools
================================================================

This script shows how to properly interact with bodies and mixtures
in their interactive modes, which is necessary since they don't
support all parameters via command-line flags.
"""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def demo_bodies_interactive():
    """Demonstrate interactive use of bodies for different shapes"""
    
    print("=== Demonstrating interactive bodies usage ===\n")
    
    # Example 1: Generate sphere
    print("1. Generating sphere (radius=100 Å)...")
    
    with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as tmp:
        sphere_file = tmp.name
    
    # Interactive input for bodies
    sphere_input = """p
7
100
0
1.0
0.005
0.5
200
{}
""".format(sphere_file)
    
    result = subprocess.run(
        ['bodies'],
        input=sphere_input,
        text=True,
        capture_output=True
    )
    
    if result.returncode == 0:
        print(f"   ✓ Sphere curve saved to: {sphere_file}")
    else:
        print(f"   ✗ Error: {result.stderr}")
    
    # Example 2: Generate cylinder
    print("\n2. Generating cylinder (R=50 Å, L=100 Å)...")
    
    with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as tmp:
        cylinder_file = tmp.name
    
    cylinder_input = """p
3
50
100
1.0
0.005
0.5
200
{}
""".format(cylinder_file)
    
    result = subprocess.run(
        ['bodies'],
        input=cylinder_input,
        text=True,
        capture_output=True
    )
    
    if result.returncode == 0:
        print(f"   ✓ Cylinder curve saved to: {cylinder_file}")
    else:
        print(f"   ✗ Error: {result.stderr}")
    
    # Example 3: Generate ellipsoid (oblate)
    print("\n3. Generating oblate ellipsoid (a=b=100 Å, c=50 Å)...")
    
    with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as tmp:
        ellipsoid_file = tmp.name
    
    ellipsoid_input = """p
1
100
100
50
1.0
0.005
0.5
200
{}
""".format(ellipsoid_file)
    
    result = subprocess.run(
        ['bodies'],
        input=ellipsoid_input,
        text=True,
        capture_output=True
    )
    
    if result.returncode == 0:
        print(f"   ✓ Ellipsoid curve saved to: {ellipsoid_file}")
    else:
        print(f"   ✗ Error: {result.stderr}")
    
    # Plot the curves
    print("\n4. Plotting generated curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for file, label in [(sphere_file, 'Sphere'), 
                       (cylinder_file, 'Cylinder'), 
                       (ellipsoid_file, 'Oblate ellipsoid')]:
        try:
            data = np.loadtxt(file)
            q, I = data[:, 0], data[:, 1]
            
            # Log plot
            ax1.loglog(q, I, 'o-', label=label, alpha=0.7, markersize=3)
            
            # Guinier plot
            guinier_range = q < 0.1  # Typical Guinier region
            ax2.plot(q[guinier_range]**2, np.log(I[guinier_range]), 
                    'o-', label=label, alpha=0.7, markersize=3)
            
        except Exception as e:
            print(f"   Could not plot {label}: {e}")
    
    ax1.set_xlabel('q (Å⁻¹)')
    ax1.set_ylabel('I(q)')
    ax1.set_title('SAXS Curves (log-log)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('q² (Å⁻²)')
    ax2.set_ylabel('ln I(q)')
    ax2.set_title('Guinier Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bodies_demo.png', dpi=150)
    print(f"   ✓ Plot saved to: bodies_demo.png")
    
    return sphere_file, cylinder_file, ellipsoid_file


def demo_mixtures_polydispersity():
    """Demonstrate using mixtures for polydispersity"""
    
    print("\n\n=== Demonstrating mixtures for polydispersity ===\n")
    
    # First generate a monodisperse sphere
    print("1. Generating monodisperse sphere (R=100 Å)...")
    
    with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as tmp:
        mono_file = tmp.name
    
    mono_input = """p
7
100
0
1.0
0.005
0.5
200
{}
""".format(mono_file)
    
    subprocess.run(['bodies'], input=mono_input, text=True, capture_output=True)
    
    # Try mixtures with command line (may fail depending on version)
    print("\n2. Applying 20% polydispersity using mixtures...")
    
    with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as tmp:
        poly_file = tmp.name
    
    # First try command-line mode
    cmd = [
        'mixtures',
        '--comp1', mono_file,
        '--dens1', '1.0',
        '--polydisperse', 'gaussian,0.2',
        '--output', poly_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("   Command-line mode failed, trying interactive mode...")
        
        # Interactive mode for mixtures (syntax varies by version)
        mixtures_input = f"""1
{mono_file}
1.0
y
gaussian
0.2
{poly_file}
"""
        
        result = subprocess.run(
            ['mixtures'],
            input=mixtures_input,
            text=True,
            capture_output=True
        )
    
    if result.returncode == 0:
        print(f"   ✓ Polydisperse curve saved to: {poly_file}")
        
        # Compare monodisperse vs polydisperse
        plt.figure(figsize=(10, 6))
        
        mono_data = np.loadtxt(mono_file)
        poly_data = np.loadtxt(poly_file)
        
        plt.loglog(mono_data[:, 0], mono_data[:, 1], 'o-', 
                  label='Monodisperse', alpha=0.7, markersize=3)
        plt.loglog(poly_data[:, 0], poly_data[:, 1], 's-', 
                  label='20% Polydispersity', alpha=0.7, markersize=3)
        
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('I(q)')
        plt.title('Effect of Polydispersity on SAXS Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('polydispersity_demo.png', dpi=150)
        print(f"   ✓ Comparison plot saved to: polydispersity_demo.png")
    else:
        print(f"   ✗ Error applying polydispersity: {result.stderr}")


def demo_core_shell_structure():
    """Demonstrate creating core-shell structure with mixtures"""
    
    print("\n\n=== Demonstrating core-shell structure ===\n")
    
    # Generate core
    print("1. Generating core (sphere R=80 Å)...")
    
    with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as tmp:
        core_file = tmp.name
    
    core_input = """p
7
80
0
1.0
0.005
0.5
200
{}
""".format(core_file)
    
    subprocess.run(['bodies'], input=core_input, text=True, capture_output=True)
    
    # Generate shell (larger sphere)
    print("2. Generating shell (sphere R=100 Å)...")
    
    with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as tmp:
        shell_file = tmp.name
    
    shell_input = """p
7
100
0
1.0
0.005
0.5
200
{}
""".format(shell_file)
    
    subprocess.run(['bodies'], input=shell_input, text=True, capture_output=True)
    
    # Combine with mixtures
    print("3. Creating core-shell structure...")
    
    with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as tmp:
        coreshell_file = tmp.name
    
    # Core has positive density, shell has negative (to subtract core volume)
    cmd = [
        'mixtures',
        '--comp1', shell_file,
        '--dens1', '1.0',
        '--comp2', core_file,
        '--dens2', '-0.5',  # Negative density for core
        '--output', coreshell_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✓ Core-shell curve saved to: {coreshell_file}")
    else:
        print(f"   ✗ Error creating core-shell: {result.stderr}")
        # Could implement interactive fallback here


def main():
    """Run all demonstrations"""
    
    print("ATSAS Interactive Mode Demonstration")
    print("====================================\n")
    
    print("This script demonstrates the correct way to use ATSAS tools")
    print("in their interactive modes, as required for full functionality.\n")
    
    # Check ATSAS availability
    try:
        subprocess.run(['bodies'], capture_output=True, check=False)
    except FileNotFoundError:
        print("ERROR: 'bodies' not found. Please ensure ATSAS is installed and in PATH.")
        return
    
    # Run demonstrations
    demo_bodies_interactive()
    demo_mixtures_polydispersity()
    demo_core_shell_structure()
    
    print("\n\nDemonstration complete!")
    print("Check the generated .png files for visualizations.")


if __name__ == '__main__':
    main()
