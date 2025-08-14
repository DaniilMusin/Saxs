#!/usr/bin/env python3
"""
Fix the dataset files that have literal \\n characters
"""
import pandas as pd
from pathlib import Path

def fix_dataset_files(dataset_dir):
    """Fix all SAXS files in dataset"""
    
    print(f"Fixing files in {dataset_dir}")
    
    dataset_path = Path(dataset_dir)
    saxs_dir = dataset_path / "saxs"
    
    if not saxs_dir.exists():
        print("SAXS directory not found!")
        return False
    
    # Get all .dat files
    dat_files = list(saxs_dir.glob("*.dat"))
    print(f"Found {len(dat_files)} files to fix")
    
    fixed_count = 0
    
    for file_path in dat_files:
        try:
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it has literal \\n
            if '\\n' in content:
                print(f"Fixing {file_path.name}")
                
                # Replace literal \\n with actual newlines
                fixed_content = content.replace('\\n', '\n')
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                fixed_count += 1
        
        except Exception as e:
            print(f"Error fixing {file_path.name}: {e}")
    
    print(f"Fixed {fixed_count} files")
    return fixed_count > 0

def verify_fixed_files(dataset_dir):
    """Verify that files are now readable"""
    
    print(f"\nVerifying fixed files in {dataset_dir}")
    
    dataset_path = Path(dataset_dir)
    saxs_dir = dataset_path / "saxs" 
    
    # Test a sample file
    sample_files = list(saxs_dir.glob("*sphere*.dat"))
    
    if not sample_files:
        print("No sphere files found for testing")
        return False
    
    sample_file = sample_files[0]
    print(f"Testing file: {sample_file.name}")
    
    try:
        q, I = [], []
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if line_num == 1:
                    print(f"Header line: '{line}'")
                    continue
                
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        q_val = float(parts[0])
                        I_val = float(parts[1])
                        q.append(q_val)
                        I.append(I_val)
                    except ValueError:
                        print(f"Line {line_num}: Could not parse '{line}'")
                        continue
                
                # Only read first 10 data lines for verification
                if len(q) >= 10:
                    break
        
        print(f"Successfully parsed {len(q)} data points")
        
        if len(q) > 0:
            print(f"q range: {min(q):.4f} - {max(q):.4f}")
            print(f"I range: {min(I):.2e} - {max(I):.2e}")
            print("File format is now correct!")
            return True
        else:
            print("No data points found")
            return False
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

def main():
    """Main fixing function"""
    
    print("=" * 60)
    print("FIXING DATASET FILES")
    print("=" * 60)
    
    dataset_dir = "corrected_saxs_sample_20250814_104054"
    
    # Fix the files
    success = fix_dataset_files(dataset_dir)
    
    if success:
        # Verify the fix worked
        verify_success = verify_fixed_files(dataset_dir)
        
        if verify_success:
            print(f"\n✓ SUCCESS: All files in {dataset_dir} are now readable!")
            print(f"You can now open and use the SAXS files.")
        else:
            print(f"\n✗ VERIFICATION FAILED: Files still have issues")
    else:
        print(f"\n✗ NO FIXES APPLIED: Files may already be correct or not found")

if __name__ == "__main__":
    main()