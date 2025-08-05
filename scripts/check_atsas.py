#!/usr/bin/env python3
"""
ATSAS Installation and Configuration Checker

This script verifies that ATSAS is properly installed and configured
for use with the SAXS dataset generator.

Usage:
    python check_atsas.py [--verbose]
"""

import subprocess
import sys
import os
import tempfile
import argparse
from pathlib import Path

# Required ATSAS tools
REQUIRED_TOOLS = {
    'bodies': 'Generate scattering curves from geometric shapes',
    'mixtures': 'Apply polydispersity and mix components (ATSAS 4.x)',
    'mixture': 'Apply polydispersity and mix components (ATSAS 3.x)', 
    'imsim': 'Simulate detector images',
    'im2dat': 'Convert 2D images to 1D curves',
    'autorg': 'Guinier analysis for quality control'
}

def check_tool_availability(tool_name, verbose=False):
    """Check if a specific ATSAS tool is available."""
    try:
        result = subprocess.run([tool_name, '--help'], 
                              capture_output=True, 
                              timeout=10,
                              text=True)
        
        if result.returncode == 0:
            if verbose:
                print(f"  ✓ {tool_name}: Available")
            return True, "Available"
        else:
            if verbose:
                print(f"  ✗ {tool_name}: Returns error code {result.returncode}")
            return False, f"Error code {result.returncode}"
            
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  ✗ {tool_name}: Timeout")
        return False, "Timeout"
    except FileNotFoundError:
        if verbose:
            print(f"  ✗ {tool_name}: Not found in PATH")
        return False, "Not found"
    except Exception as e:
        if verbose:
            print(f"  ✗ {tool_name}: {str(e)}")
        return False, str(e)

def detect_atsas_version():
    """Detect ATSAS version and available mixing tools."""
    has_mixtures = check_tool_availability('mixtures')[0]
    has_mixture = check_tool_availability('mixture')[0]
    
    if has_mixtures:
        return "4.x", "mixtures"
    elif has_mixture:
        return "3.x", "mixture"
    else:
        return "unknown", None

def test_bodies_interactive():
    """Test bodies in interactive mode with a simple sphere."""
    print("\nTesting bodies interactive mode...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test_sphere.int")
        
        # Simple sphere: radius=50, q-range 0.01-0.5
        input_text = """3
1
50.0
50
2
100
0.01
0.5
50
1
{}
""".format(output_file)
        
        try:
            process = subprocess.run(['bodies'], 
                                   input=input_text, 
                                   text=True, 
                                   capture_output=True,
                                   timeout=30)
            
            if process.returncode == 0 and os.path.exists(output_file):
                # Check file has reasonable content
                with open(output_file) as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                if len(lines) >= 10:  # Should have at least 10 data points
                    print("  ✓ bodies interactive mode working")
                    return True
                else:
                    print(f"  ✗ bodies output too short ({len(lines)} lines)")
                    return False
            else:
                print("  ✗ bodies failed or no output file")
                print(f"    Return code: {process.returncode}")
                if process.stderr:
                    print(f"    Error: {process.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            print("  ✗ bodies timeout")
            return False
        except Exception as e:
            print(f"  ✗ bodies test error: {e}")
            return False

def test_mixing_tool(mix_tool):
    """Test the mixing tool (mixtures or mixture)."""
    print(f"\nTesting {mix_tool}...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_file = os.path.join(tmpdir, "base.int")
        output_file = os.path.join(tmpdir, "mixed.int")
        
        # First create base curve
        input_text = """3
1
50.0
30
2
100
0.01
0.5
30
1
{}
""".format(base_file)
        
        try:
            # Generate base curve
            subprocess.run(['bodies'], 
                          input=input_text, 
                          text=True, 
                          capture_output=True,
                          timeout=30)
            
            if not os.path.exists(base_file):
                print(f"  ✗ Failed to create base file for {mix_tool} test")
                return False
            
            # Test mixing with 10% polydispersity
            mix_input = """1
{}
1
1
0.1
50
1
{}
""".format(base_file, output_file)
            
            process = subprocess.run([mix_tool], 
                                   input=mix_input, 
                                   text=True, 
                                   capture_output=True,
                                   timeout=30)
            
            if process.returncode == 0 and os.path.exists(output_file):
                print(f"  ✓ {mix_tool} working")
                return True
            else:
                print(f"  ✗ {mix_tool} failed")
                if process.stderr:
                    print(f"    Error: {process.stderr[:200]}")
                return False
                
        except Exception as e:
            print(f"  ✗ {mix_tool} test error: {e}")
            return False

def test_imsim():
    """Test imsim detector simulation."""
    print("\nTesting imsim...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        int_file = os.path.join(tmpdir, "test.int")
        edf_file = os.path.join(tmpdir, "test.edf")
        
        # Create simple I(q) curve first
        input_text = """3
1
50.0
20
2
100
0.01
0.3
20
1
{}
""".format(int_file)
        
        try:
            subprocess.run(['bodies'], 
                          input=input_text, 
                          text=True, 
                          capture_output=True,
                          timeout=30)
            
            if not os.path.exists(int_file):
                print("  ✗ Failed to create input file for imsim test")
                return False
            
            # Simple detector simulation
            imsim_input = """{}
{}
512
512
0.172
1000
1.54
10000
1.0
""".format(int_file, edf_file)
            
            process = subprocess.run(['imsim'], 
                                   input=imsim_input, 
                                   text=True, 
                                   capture_output=True,
                                   timeout=60)
            
            if process.returncode == 0 and os.path.exists(edf_file):
                print("  ✓ imsim working")
                return True
            else:
                print("  ✗ imsim failed")
                if process.stderr:
                    print(f"    Error: {process.stderr[:200]}")
                return False
                
        except Exception as e:
            print(f"  ✗ imsim test error: {e}")
            return False

def check_license():
    """Check ATSAS license status."""
    print("\nChecking ATSAS license...")
    
    # Try to run two instances of bodies simultaneously
    try:
        processes = []
        for i in range(3):  # Try 3 processes to test license limit
            with tempfile.NamedTemporaryFile(suffix='.int', delete=False) as f:
                temp_file = f.name
            
            input_text = f"""3
1
50.0
10
2
100
0.01
0.1
10
1
{temp_file}
"""
            
            proc = subprocess.Popen(['bodies'], 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
            proc.stdin.write(input_text)
            proc.stdin.close()
            processes.append((proc, temp_file))
        
        # Wait for all to complete
        results = []
        for proc, temp_file in processes:
            proc.wait()
            results.append(proc.returncode == 0)
            try:
                os.unlink(temp_file)
            except:
                pass
        
        successful = sum(results)
        
        if successful >= 2:
            print(f"  ✓ License allows at least {successful} parallel processes")
        elif successful == 1:
            print("  ! License limited to 1 parallel process")
        else:
            print("  ✗ License issues detected")
            
    except Exception as e:
        print(f"  ? Could not test license: {e}")

def main():
    """Main checking function."""
    parser = argparse.ArgumentParser(description='Check ATSAS installation for SAXS generator')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    args = parser.parse_args()
    
    print("ATSAS Installation Checker")
    print("=" * 40)
    
    # Check tool availability
    print("\nChecking ATSAS tools availability:")
    available_tools = {}
    missing_tools = []
    
    for tool, description in REQUIRED_TOOLS.items():
        is_available, status = check_tool_availability(tool, args.verbose)
        available_tools[tool] = is_available
        if not is_available:
            missing_tools.append(tool)
    
    # Detect version
    version, mix_tool = detect_atsas_version()
    print(f"\nATSAS version detected: {version}")
    if mix_tool:
        print(f"Mixing tool available: {mix_tool}")
    
    # Essential tools check
    essential = ['bodies', 'imsim', 'im2dat']
    missing_essential = [t for t in essential if not available_tools.get(t, False)]
    
    if missing_essential:
        print(f"\n✗ CRITICAL: Missing essential tools: {', '.join(missing_essential)}")
        print("The SAXS generator cannot work without these tools.")
        return 1
    
    if not available_tools.get('mixtures', False) and not available_tools.get('mixture', False):
        print("\n✗ CRITICAL: No mixing tool available (mixtures or mixture)")
        print("Polydispersity features will not work.")
        return 1
    
    # Functional tests
    if not test_bodies_interactive():
        print("\n✗ CRITICAL: bodies interactive mode not working")
        return 1
    
    if mix_tool and not test_mixing_tool(mix_tool):
        print(f"\n✗ CRITICAL: {mix_tool} not working")
        return 1
    
    if not test_imsim():
        print("\n! WARNING: imsim not working properly")
        print("  Detector simulation may fail")
    
    # License check
    check_license()
    
    print("\n" + "=" * 40)
    print("✓ ATSAS appears to be properly configured!")
    print("\nYou can now use:")
    print("  python scripts/generate.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())