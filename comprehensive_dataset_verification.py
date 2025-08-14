#!/usr/bin/env python3
"""
Comprehensive verification of all 85,000 curves in the final dataset
Checks mathematical correctness, physical consistency, and data quality
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

class DatasetVerifier:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.meta_file = self.dataset_dir / "meta.csv"
        self.saxs_dir = self.dataset_dir / "saxs"
        self.errors = []
        self.warnings = []
        self.stats = {}
        
    def load_metadata(self):
        """Load and validate metadata"""
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_file}")
        
        self.df = pd.read_csv(self.meta_file)
        print(f"Loaded metadata: {len(self.df)} entries")
        
        # Check metadata structure
        required_cols = ['uid', 'filename', 'shape_class', 'true_params', 'corrected']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            self.errors.append(f"Missing metadata columns: {missing_cols}")
        
        return len(self.df)
    
    def exact_sphere_form_factor(self, q, R):
        """Reference exact sphere form factor for validation"""
        qR = q * R
        F = np.ones_like(qR)
        mask = np.abs(qR) > 1e-10
        qR_nz = qR[mask]
        F[mask] = 3 * (np.sin(qR_nz) - qR_nz * np.cos(qR_nz)) / (qR_nz**3)
        return np.abs(F)**2
    
    def find_first_minimum(self, q, I, shape_class, R):
        """Find first minimum in scattering curve"""
        if shape_class != 'sphere':
            return None, None  # Only check spheres for exact validation
            
        # Theoretical first minimum for sphere: q_min = 4.493 / R
        q_theoretical = 4.493 / R
        
        # Find minimum in the curve
        # Look for minimum in the range where we expect it
        q_search_min = max(0.02, q_theoretical * 0.5)
        q_search_max = min(0.4, q_theoretical * 1.5)
        
        mask = (q >= q_search_min) & (q <= q_search_max)
        if not np.any(mask):
            return None, None
        
        q_region = q[mask]
        I_region = I[mask]
        
        min_idx = np.argmin(I_region)
        q_observed = q_region[min_idx]
        
        return q_theoretical, q_observed
    
    def verify_single_file(self, row):
        """Verify a single SAXS file"""
        filename = row['filename']
        shape_class = row['shape_class']
        uid = row['uid']
        
        saxs_file = self.saxs_dir / filename
        
        if not saxs_file.exists():
            return {'error': f"File not found: {filename}"}
        
        try:
            # Load SAXS data
            q, I = [], []
            with open(saxs_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line_num == 1:  # Header
                        continue
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            q.append(float(parts[0]))
                            I.append(float(parts[1]))
                        except ValueError:
                            return {'error': f"Invalid data in line {line_num}: {line}"}
            
            if len(q) < 10:
                return {'error': f"Insufficient data points: {len(q)}"}
            
            q = np.array(q)
            I = np.array(I)
            
            # Basic validation checks
            checks = {
                'file_exists': True,
                'data_points': len(q),
                'q_range_ok': 0.009 <= q.min() <= 0.02 and 0.4 <= q.max() <= 0.6,
                'q_monotonic': np.all(np.diff(q) > 0),
                'I_positive': np.all(I > 0),
                'I_finite': np.all(np.isfinite(I)),
                'I_reasonable_range': 1e-20 <= I.min() and I.max() <= 1e20,
                'forward_scattering_ok': I[0] > 0.1,  # Forward scattering should be significant
                'decay_ok': I[-1] < I[0],  # Should decay with q
                'no_jumps': np.max(np.abs(np.diff(np.log(I)))) < 10  # No extreme jumps
            }
            
            # Shape-specific validation
            if shape_class == 'sphere':
                try:
                    params_str = row['true_params']
                    params = json.loads(params_str.replace('""', '"'))
                    R = params['R']
                    
                    # Check first minimum position
                    q_theo, q_obs = self.find_first_minimum(q, I, shape_class, R)
                    if q_theo is not None and q_obs is not None:
                        error_pct = abs(q_obs - q_theo) / q_theo * 100
                        checks['sphere_minimum_error'] = error_pct
                        checks['sphere_minimum_ok'] = error_pct < 5.0  # Less than 5% error
                    else:
                        checks['sphere_minimum_found'] = False
                        
                except Exception as e:
                    checks['sphere_params_error'] = str(e)
            
            # Quality score
            quality_checks = ['q_range_ok', 'q_monotonic', 'I_positive', 'I_finite', 
                            'I_reasonable_range', 'forward_scattering_ok', 'decay_ok', 'no_jumps']
            passed_checks = sum(checks.get(check, False) for check in quality_checks)
            checks['quality_score'] = passed_checks / len(quality_checks)
            
            return checks
            
        except Exception as e:
            return {'error': f"Failed to process {filename}: {str(e)}"}
    
    def verify_dataset(self, sample_size=None):
        """Verify the entire dataset or a sample"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DATASET VERIFICATION")
        print("="*80)
        
        total_files = len(self.df)
        
        if sample_size and sample_size < total_files:
            print(f"Verifying random sample of {sample_size} files out of {total_files}")
            sample_df = self.df.sample(n=sample_size, random_state=42)
        else:
            print(f"Verifying all {total_files} files")
            sample_df = self.df
            sample_size = total_files
        
        # Initialize statistics
        self.stats = {
            'total_checked': 0,
            'files_ok': 0,
            'files_with_errors': 0,
            'files_with_warnings': 0,
            'shape_stats': {},
            'quality_distribution': []
        }
        
        # Process files
        for idx, row in sample_df.iterrows():
            if self.stats['total_checked'] % 1000 == 0 and self.stats['total_checked'] > 0:
                progress = self.stats['total_checked'] / sample_size * 100
                print(f"Progress: {self.stats['total_checked']}/{sample_size} ({progress:.1f}%)")
            
            result = self.verify_single_file(row)
            self.stats['total_checked'] += 1
            
            shape_class = row['shape_class']
            if shape_class not in self.stats['shape_stats']:
                self.stats['shape_stats'][shape_class] = {
                    'total': 0, 'ok': 0, 'errors': 0, 'warnings': 0,
                    'quality_scores': [], 'sphere_errors': []
                }
            
            shape_stats = self.stats['shape_stats'][shape_class]
            shape_stats['total'] += 1
            
            if 'error' in result:
                self.stats['files_with_errors'] += 1
                shape_stats['errors'] += 1
                self.errors.append(f"{row['filename']}: {result['error']}")
            else:
                quality_score = result.get('quality_score', 0)
                self.stats['quality_distribution'].append(quality_score)
                shape_stats['quality_scores'].append(quality_score)
                
                if quality_score >= 0.8:
                    self.stats['files_ok'] += 1
                    shape_stats['ok'] += 1
                else:
                    self.stats['files_with_warnings'] += 1
                    shape_stats['warnings'] += 1
                    self.warnings.append(f"{row['filename']}: Low quality score {quality_score:.2f}")
                
                # Track sphere accuracy
                if shape_class == 'sphere' and 'sphere_minimum_error' in result:
                    shape_stats['sphere_errors'].append(result['sphere_minimum_error'])
        
        self.print_results()
        return self.generate_report()
    
    def print_results(self):
        """Print verification results"""
        print("\n" + "="*80)
        print("VERIFICATION RESULTS")
        print("="*80)
        
        total = self.stats['total_checked']
        ok = self.stats['files_ok']
        warnings = self.stats['files_with_warnings']
        errors = self.stats['files_with_errors']
        
        print(f"Total files checked: {total}")
        print(f"Files OK: {ok} ({ok/total*100:.1f}%)")
        print(f"Files with warnings: {warnings} ({warnings/total*100:.1f}%)")
        print(f"Files with errors: {errors} ({errors/total*100:.1f}%)")
        
        if self.stats['quality_distribution']:
            avg_quality = np.mean(self.stats['quality_distribution'])
            print(f"Average quality score: {avg_quality:.3f}")
        
        print(f"\nShape-specific results:")
        for shape, stats in self.stats['shape_stats'].items():
            total_shape = stats['total']
            ok_shape = stats['ok']
            success_rate = ok_shape / total_shape * 100 if total_shape > 0 else 0
            
            print(f"  {shape}: {ok_shape}/{total_shape} OK ({success_rate:.1f}%)")
            
            if shape == 'sphere' and stats['sphere_errors']:
                avg_error = np.mean(stats['sphere_errors'])
                max_error = np.max(stats['sphere_errors'])
                print(f"    Sphere accuracy: avg {avg_error:.2f}% error, max {max_error:.2f}%")
        
        # Show some errors if any
        if self.errors:
            print(f"\nFirst 5 errors:")
            for error in self.errors[:5]:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\nFirst 3 warnings:")
            for warning in self.warnings[:3]:
                print(f"  - {warning}")
    
    def generate_report(self):
        """Generate final report"""
        total = self.stats['total_checked']
        ok = self.stats['files_ok']
        success_rate = ok / total * 100 if total > 0 else 0
        
        if success_rate >= 95:
            verdict = "EXCELLENT"
        elif success_rate >= 90:
            verdict = "GOOD"
        elif success_rate >= 80:
            verdict = "ACCEPTABLE"
        else:
            verdict = "NEEDS IMPROVEMENT"
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_dir': str(self.dataset_dir),
            'total_files': total,
            'success_rate': success_rate,
            'verdict': verdict,
            'stats': self.stats,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
        
        print(f"\n" + "="*80)
        print(f"FINAL VERDICT: {verdict}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Dataset Quality: {'PASSED' if success_rate >= 90 else 'NEEDS WORK'}")
        print("="*80)
        
        return report

def main():
    """Main verification function"""
    dataset_dir = "corrected_dataset_full_20250814_110527"
    
    if not Path(dataset_dir).exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return False
    
    verifier = DatasetVerifier(dataset_dir)
    
    # Load metadata
    total_files = verifier.load_metadata()
    
    # For full verification, we'll check a representative sample first
    # then do full verification if sample looks good
    print("\nStarting with sample verification (1000 files)...")
    sample_report = verifier.verify_dataset(sample_size=1000)
    
    if sample_report['success_rate'] >= 85:
        print(f"\nSample verification passed ({sample_report['success_rate']:.1f}%)")
        print("Proceeding with full verification...")
        
        # Reset for full verification
        verifier.errors = []
        verifier.warnings = []
        verifier.stats = {}
        
        full_report = verifier.verify_dataset()
        return full_report['success_rate'] >= 90
    else:
        print(f"\nSample verification failed ({sample_report['success_rate']:.1f}%)")
        print("Issues detected. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n*** DATASET VERIFICATION: PASSED ***")
    else:
        print("\n*** DATASET VERIFICATION: FAILED ***")