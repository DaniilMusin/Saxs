#!/usr/bin/env python3
"""
check_atsas.py - Check ATSAS installation and capabilities
=========================================================

This script verifies ATSAS installation, checks versions,
and tests available functionality to help configure the
dataset generator appropriately.
"""

import subprocess
import sys
import re
from pathlib import Path


def check_tool(tool_name: str) -> dict:
    """Check if a tool is available and get version info"""
    
    result = {'available': False, 'version': None, 'path': None}
    
    # Try to run the tool
    for cmd in [tool_name, f"{tool_name}.exe"]:
        try:
            # First try --version
            proc = subprocess.run([cmd, '--version'], 
                                capture_output=True, text=True)
            if proc.returncode == 0:
                result['available'] = True
                result['version'] = proc.stdout.strip()
                result['path'] = subprocess.run(['which', cmd], 
                                              capture_output=True, 
                                              text=True).stdout.strip()
                break
                
            # Try --help if --version doesn't work
            proc = subprocess.run([cmd, '--help'], 
                                capture_output=True, text=True)
            if proc.returncode == 0 or 'usage' in proc.stdout.lower():
                result['available'] = True
                # Try to extract version from help text
                version_match = re.search(r'version\s+(\d+\.\d+)', 
                                        proc.stdout, re.IGNORECASE)
                if version_match:
                    result['version'] = version_match.group(1)
                result['path'] = subprocess.run(['which', cmd], 
                                              capture_output=True, 
                                              text=True).stdout.strip()
                break
                
            # Try running without arguments
            proc = subprocess.run([cmd], capture_output=True, text=True)
            if 'ATSAS' in proc.stdout or 'ATSAS' in proc.stderr:
                result['available'] = True
                result['path'] = subprocess.run(['which', cmd], 
                                              capture_output=True, 
                                              text=True).stdout.strip()
                break
                
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error checking {cmd}: {e}")
    
    return result


def test_bodies_modes():
    """Test available modes in bodies"""
    
    print("\n=== Testing bodies functionality ===")
    
    # Test predict mode
    test_input = """p
7
100
0
1.0
0.005
0.1
10
test_bodies.int
"""
    
    try:
        proc = subprocess.run(['bodies'], input=test_input, 
                            text=True, capture_output=True)
        if proc.returncode == 0:
            print("✓ bodies predict mode: WORKING")
            # Clean up test file
            Path('test_bodies.int').unlink(missing_ok=True)
        else:
            print("✗ bodies predict mode: FAILED")
            print(f"  Error: {proc.stderr}")
    except Exception as e:
        print(f"✗ bodies predict mode: ERROR - {e}")


def test_mixtures_cli():
    """Test if mixtures supports CLI mode"""
    
    print("\n=== Testing mixtures functionality ===")
    
    # Try different command names
    mixture_cmds = ['mixtures', 'mixture']
    
    for cmd in mixture_cmds:
        result = check_tool(cmd)
        if result['available']:
            print(f"✓ {cmd} found at: {result['path']}")
            
            # Test CLI mode
            test_cmd = [cmd, '--help']
            proc = subprocess.run(test_cmd, capture_output=True, text=True)
            
            if '--comp' in proc.stdout or '--polydisperse' in proc.stdout:
                print(f"  ✓ CLI mode supported")
            else:
                print(f"  ✗ CLI mode not detected (interactive only)")
            
            break
    else:
        print("✗ No mixture/mixtures tool found")


def check_license_info():
    """Check for ATSAS license information"""
    
    print("\n=== Checking license information ===")
    
    # Common license file locations
    license_paths = [
        Path.home() / '.atsas' / 'license.dat',
        Path('/etc/atsas/license.dat'),
        Path(sys.prefix) / 'share' / 'atsas' / 'license.dat'
    ]
    
    found_license = False
    for path in license_paths:
        if path.exists():
            print(f"✓ License file found: {path}")
            found_license = True
            
            # Try to read parallel limit
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    if 'parallel' in content.lower():
                        parallel_match = re.search(r'parallel[:\s]+(\d+)', 
                                                 content, re.IGNORECASE)
                        if parallel_match:
                            print(f"  Parallel limit: {parallel_match.group(1)}")
            except:
                pass
            break
    
    if not found_license:
        print("✗ No license file found in standard locations")
        print("  ATSAS may still work with environment variables")


def main():
    """Run all checks"""
    
    print("ATSAS Installation Check")
    print("========================\n")
    
    # Check required tools
    required_tools = ['bodies', 'mixtures', 'imsim', 'im2dat', 'autorg']
    optional_tools = ['crysol', 'datcmp', 'datgnom']
    
    print("=== Required tools ===")
    all_found = True
    for tool in required_tools:
        if tool == 'mixtures':
            # Special case: might be 'mixture' in older versions
            result = check_tool('mixtures')
            if not result['available']:
                result = check_tool('mixture')
                if result['available']:
                    tool = 'mixture'
        else:
            result = check_tool(tool)
        
        if result['available']:
            version_str = f" (version: {result['version']})" if result['version'] else ""
            print(f"✓ {tool:<12} {version_str}")
        else:
            print(f"✗ {tool:<12} NOT FOUND")
            all_found = False
    
    print("\n=== Optional tools ===")
    for tool in optional_tools:
        result = check_tool(tool)
        if result['available']:
            version_str = f" (version: {result['version']})" if result['version'] else ""
            print(f"✓ {tool:<12} {version_str}")
        else:
            print(f"- {tool:<12} not found")
    
    # Test functionality
    if all_found:
        test_bodies_modes()
        test_mixtures_cli()
    
    # Check license
    check_license_info()
    
    # Summary
    print("\n=== Summary ===")
    if all_found:
        print("✓ All required ATSAS tools are available")
        print("✓ Ready to generate SAXS dataset")
        print("\nRecommended command:")
        print("  python scripts/generate.py --n 1000 --jobs 4 --atsas-parallel 2")
    else:
        print("✗ Some required tools are missing")
        print("✗ Please install ATSAS and ensure tools are in PATH")
        print("\nDownload ATSAS from:")
        print("  https://www.embl-hamburg.de/biosaxs/software.html")
        
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
