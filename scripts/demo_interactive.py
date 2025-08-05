#!/usr/bin/env python3
"""
Interactive demonstration of ATSAS tools usage for SAXS dataset generation.

This script demonstrates how to use ATSAS tools in interactive mode,
which is the preferred method for the SAXS dataset generator.

Usage:
    python demo_interactive.py

This will show examples of:
- Using bodies for basic shapes
- Using mixtures for polydispersity
- Creating core-shell structures
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path

def check_atsas_availability():
    """Check if ATSAS tools are available."""
    tools = ['bodies', 'mixtures', 'imsim', 'im2dat']
    missing = []
    
    for tool in tools:
        try:
            result = subprocess.run([tool, '--help'], 
                                  capture_output=True, 
                                  timeout=5)
            if result.returncode != 0:
                missing.append(tool)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append(tool)
    
    return missing

def demo_bodies_sphere():
    """Demonstrate bodies interactive mode for sphere."""
    print("\n=== Demonstrating bodies for sphere ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "sphere_demo.int")
        
        # Prepare input for bodies interactive mode
        input_text = """3
1
100.0
50
2
100
0.1
1.0
1
0.001
0.5
100
1
{}
""".format(output_file)
        
        print("Bodies input commands:")
        print(input_text)
        
        try:
            process = subprocess.run(['bodies'], 
                                   input=input_text, 
                                   text=True, 
                                   capture_output=True,
                                   timeout=30)
            
            if process.returncode == 0 and os.path.exists(output_file):
                print("✓ Sphere generation successful!")
                print(f"Output file: {output_file}")
                
                # Show first few lines
                with open(output_file) as f:
                    lines = f.readlines()[:5]
                    print("First few lines:")
                    for line in lines:
                        print(f"  {line.strip()}")
            else:
                print("✗ Sphere generation failed")
                print("STDOUT:", process.stdout)
                print("STDERR:", process.stderr)
                
        except subprocess.TimeoutExpired:
            print("✗ Timeout - bodies took too long")
        except Exception as e:
            print(f"✗ Error: {e}")

def demo_mixtures_polydispersity():
    """Demonstrate mixtures for polydispersity."""
    print("\n=== Demonstrating mixtures for polydispersity ===")
    
    # First need to create a base curve with bodies
    with tempfile.TemporaryDirectory() as tmpdir:
        base_file = os.path.join(tmpdir, "base_sphere.int")
        output_file = os.path.join(tmpdir, "poly_sphere.int")
        
        # Generate base sphere
        input_text = """3
1
100.0
50
2
100
0.1
1.0
1
0.001
0.5
100
1
{}
""".format(base_file)
        
        try:
            subprocess.run(['bodies'], 
                          input=input_text, 
                          text=True, 
                          capture_output=True,
                          timeout=30)
            
            if os.path.exists(base_file):
                # Now apply polydispersity with mixtures
                mix_input = """1
{}
1
1
0.2
100
1
{}
""".format(base_file, output_file)
                
                print("Mixtures input commands:")
                print(mix_input)
                
                process = subprocess.run(['mixtures'], 
                                       input=mix_input, 
                                       text=True, 
                                       capture_output=True,
                                       timeout=30)
                
                if process.returncode == 0 and os.path.exists(output_file):
                    print("✓ Polydisperse sphere generation successful!")
                else:
                    print("✗ Mixtures failed")
                    print("STDOUT:", process.stdout)
                    print("STDERR:", process.stderr)
            
        except Exception as e:
            print(f"✗ Error in polydispersity demo: {e}")

def demo_imsim():
    """Demonstrate detector simulation with imsim."""
    print("\n=== Demonstrating imsim detector simulation ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # First create a simple I(q) curve
        int_file = os.path.join(tmpdir, "test_curve.int")
        edf_file = os.path.join(tmpdir, "detector_image.edf")
        
        # Generate simple curve first
        input_text = """3
1
100.0
50
2
100
0.1
1.0
1
0.001
0.5
100
1
{}
""".format(int_file)
        
        try:
            subprocess.run(['bodies'], 
                          input=input_text, 
                          text=True, 
                          capture_output=True,
                          timeout=30)
            
            if os.path.exists(int_file):
                # Now simulate detector image
                imsim_input = """{}
{}
1024
1024
0.172
1000
1.54
100000
1.0
""".format(int_file, edf_file)
                
                print("Imsim input commands:")
                print(imsim_input)
                
                process = subprocess.run(['imsim'], 
                                       input=imsim_input, 
                                       text=True, 
                                       capture_output=True,
                                       timeout=60)
                
                if process.returncode == 0 and os.path.exists(edf_file):
                    print("✓ Detector simulation successful!")
                    print(f"Generated EDF file: {edf_file}")
                else:
                    print("✗ Imsim failed")
                    print("STDOUT:", process.stdout)
                    print("STDERR:", process.stderr)
            
        except Exception as e:
            print(f"✗ Error in imsim demo: {e}")

def main():
    """Main demonstration function."""
    print("SAXS Dataset Generator - ATSAS Interactive Mode Demo")
    print("=" * 50)
    
    # Check ATSAS availability
    missing_tools = check_atsas_availability()
    
    if missing_tools:
        print(f"✗ Missing ATSAS tools: {', '.join(missing_tools)}")
        print("Please ensure ATSAS is properly installed and in PATH")
        return 1
    
    print("✓ All required ATSAS tools found")
    
    # Run demonstrations
    demo_bodies_sphere()
    demo_mixtures_polydispersity() 
    demo_imsim()
    
    print("\n" + "=" * 50)
    print("Demo complete! Check the main generator script for full implementation.")
    print("Use: python scripts/generate.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())