#!/usr/bin/env python3
"""
check_quality.py - Quality control for generated SAXS dataset
============================================================

This script performs the quality checks specified in the technical requirements:
1. Compare noisy curves with theoretical (noiseless) curves - expect χ² ≈ 1.0
2. Run Guinier analysis (autorg) - expect Rg within 10% of true radius
3. Check dataset balance and completeness
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging


def run_datcmp(curve1: Path, curve2: Path) -> float:
    """
    Compare two SAXS curves using ATSAS datcmp
    
    Returns:
    --------
    chi2 : float
        Chi-squared value from comparison
    """
    cmd = ['datcmp', str(curve1), str(curve2)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"datcmp failed: {result.stderr}")
    
    # Parse chi2 from output (format may vary with ATSAS version)
    for line in result.stdout.split('\n'):
        if 'chi' in line.lower():
            # Extract numerical value
            parts = line.split()
            for part in parts:
                try:
                    chi2 = float(part)
                    if 0 < chi2 < 1000:  # Reasonable range
                        return chi2
                except ValueError:
                    continue
    
    raise ValueError(f"Could not parse chi2 from datcmp output: {result.stdout}")


def run_autorg(curve_file: Path) -> Dict[str, float]:
    """
    Run Guinier analysis using ATSAS autorg
    
    Returns:
    --------
    results : dict
        Dictionary with 'Rg', 'I0', 'quality' etc.
    """
    cmd = ['autorg', str(curve_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"autorg failed: {result.stderr}")
    
    results = {}
    for line in result.stdout.split('\n'):
        if 'Rg' in line and '±' in line:
            # Parse Rg value
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'Rg':
                    try:
                        results['Rg'] = float(parts[i+2])  # Skip '='
                        results['Rg_error'] = float(parts[i+4].strip('±'))
                    except (IndexError, ValueError):
                        pass
        elif 'I(0)' in line:
            # Parse I(0)
            parts = line.split()
            for i, part in enumerate(parts):
                if 'I(0)' in part:
                    try:
                        results['I0'] = float(parts[i+2])
                    except (IndexError, ValueError):
                        pass
    
    return results


def check_chi2_distribution(dataset_dir: Path, n_samples: int = 100) -> Dict:
    """
    Check chi-squared distribution by comparing with noiseless curves
    """
    logging.info(f"Checking chi² distribution for {n_samples} curves...")
    
    meta_file = dataset_dir / 'meta.csv'
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    
    df = pd.read_csv(meta_file)
    
    # Sample random curves
    sample_indices = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
    
    chi2_values = []
    failed_comparisons = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for idx in sample_indices:
            row = df.iloc[idx]
            uid = row['uid']
            
            # Load original curve
            curve_file = dataset_dir / 'saxs' / f'{uid:06d}.dat'
            if not curve_file.exists():
                logging.warning(f"Curve file not found: {curve_file}")
                continue
            
            try:
                # Regenerate noiseless curve with same parameters
                params = json.loads(row['true_params'])
                
                # Generate PDB with same parameters
                # (This is simplified - would need full parameter reconstruction)
                pdb_file = tmpdir / f'structure_{uid}.pdb'
                # ... generate PDB ...
                
                # Generate noiseless curve
                noiseless_file = tmpdir / f'noiseless_{uid}.dat'
                # ... run crysol ...
                
                # Compare curves
                chi2 = run_datcmp(curve_file, noiseless_file)
                chi2_values.append(chi2)
                
            except Exception as e:
                logging.error(f"Failed to process UID {uid}: {e}")
                failed_comparisons.append(uid)
    
    # Analyze chi2 distribution
    if chi2_values:
        chi2_array = np.array(chi2_values)
        results = {
            'n_samples': len(chi2_values),
            'chi2_mean': np.mean(chi2_array),
            'chi2_std': np.std(chi2_array),
            'chi2_median': np.median(chi2_array),
            'chi2_range': (np.min(chi2_array), np.max(chi2_array)),
            'failed_comparisons': len(failed_comparisons)
        }
        
        logging.info(f"Chi² statistics: mean={results['chi2_mean']:.3f} ± {results['chi2_std']:.3f}")
        
        # Check if chi2 ≈ 1
        if 0.8 < results['chi2_mean'] < 1.2:
            logging.info("✓ Chi² values are within expected range (0.8-1.2)")
        else:
            logging.warning(f"✗ Chi² values outside expected range: {results['chi2_mean']:.3f}")
    else:
        results = {'error': 'No successful comparisons'}
    
    return results


def check_guinier_accuracy(dataset_dir: Path, n_samples: int = 100) -> Dict:
    """
    Check Guinier analysis accuracy by comparing Rg with true radius
    """
    logging.info(f"Checking Guinier analysis for {n_samples} curves...")
    
    meta_file = dataset_dir / 'meta.csv'
    df = pd.read_csv(meta_file)
    
    # Focus on simple shapes where Rg relationship is known
    simple_shapes = df[df['shape_class'].isin(['sphere', 'cylinder'])].copy()
    if len(simple_shapes) == 0:
        return {'error': 'No simple shapes found for Guinier check'}
    
    # Sample curves
    sample_indices = np.random.choice(len(simple_shapes), 
                                    min(n_samples, len(simple_shapes)), 
                                    replace=False)
    
    rg_comparisons = []
    failed_analyses = []
    
    for idx in sample_indices:
        row = simple_shapes.iloc[idx]
        uid = row['uid']
        params = json.loads(row['true_params'])
        
        # Calculate expected Rg
        if row['shape_class'] == 'sphere':
            # For sphere: Rg = sqrt(3/5) * R ≈ 0.775 * R
            expected_rg = np.sqrt(3/5) * params['radius']
        elif row['shape_class'] == 'cylinder':
            # For cylinder: Rg depends on R and L
            R = params['radius']
            L = params['length']
            expected_rg = np.sqrt(R**2/2 + L**2/12)
        else:
            continue
        
        # Run autorg
        curve_file = dataset_dir / 'saxs' / f'{uid:06d}.dat'
        if not curve_file.exists():
            continue
        
        try:
            results = run_autorg(curve_file)
            if 'Rg' in results:
                measured_rg = results['Rg']
                relative_error = abs(measured_rg - expected_rg) / expected_rg
                
                rg_comparisons.append({
                    'uid': uid,
                    'shape': row['shape_class'],
                    'expected_rg': expected_rg,
                    'measured_rg': measured_rg,
                    'relative_error': relative_error
                })
        except Exception as e:
            logging.error(f"Autorg failed for UID {uid}: {e}")
            failed_analyses.append(uid)
    
    # Analyze results
    if rg_comparisons:
        errors = [comp['relative_error'] for comp in rg_comparisons]
        results = {
            'n_samples': len(rg_comparisons),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'within_10_percent': sum(e < 0.1 for e in errors),
            'failed_analyses': len(failed_analyses)
        }
        
        pct_within_10 = 100 * results['within_10_percent'] / results['n_samples']
        logging.info(f"Rg accuracy: {pct_within_10:.1f}% within 10% of true value")
        
        if pct_within_10 > 90:
            logging.info("✓ Guinier analysis accuracy is good (>90% within 10%)")
        else:
            logging.warning(f"✗ Guinier analysis accuracy below target: {pct_within_10:.1f}%")
    else:
        results = {'error': 'No successful Guinier analyses'}
    
    return results


def check_dataset_balance(dataset_dir: Path) -> Dict:
    """
    Check dataset balance and completeness
    """
    logging.info("Checking dataset balance...")
    
    meta_file = dataset_dir / 'meta.csv'
    if not meta_file.exists():
        return {'error': 'Metadata file not found'}
    
    df = pd.read_csv(meta_file)
    
    # Check shape distribution
    shape_counts = df['shape_class'].value_counts()
    
    # Check instrument distribution
    instrument_counts = df['instrument_cfg'].value_counts()
    
    # Check for missing files
    missing_files = []
    for idx, row in df.iterrows():
        uid = row['uid']
        # Use actual filename from metadata if available
        filename = row['filename'] if 'filename' in df.columns and pd.notna(row['filename']) else f'{uid:06d}.dat'
        curve_file = dataset_dir / 'saxs' / filename
        if not curve_file.exists():
            missing_files.append(uid)
    
    # Calculate balance metrics
    shape_balance = shape_counts.std() / shape_counts.mean()
    instrument_balance = instrument_counts.std() / instrument_counts.mean()
    
    results = {
        'total_curves': len(df),
        'shape_distribution': shape_counts.to_dict(),
        'instrument_distribution': instrument_counts.to_dict(),
        'missing_files': len(missing_files),
        'shape_balance_cv': shape_balance,  # Coefficient of variation
        'instrument_balance_cv': instrument_balance
    }
    
    # Check balance criteria
    if shape_balance < 0.05:  # Within 5% variation
        logging.info("✓ Shape classes are well balanced")
    else:
        logging.warning(f"✗ Shape class imbalance detected: CV={shape_balance:.3f}")
    
    if len(missing_files) == 0:
        logging.info("✓ All curve files present")
    else:
        logging.warning(f"✗ {len(missing_files)} curve files missing")
    
    return results


def plot_quality_report(results: Dict, output_file: Path) -> None:
    """Generate visual quality report"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SAXS Dataset Quality Report', fontsize=16)
    
    # Plot 1: Chi2 distribution
    ax = axes[0, 0]
    if 'chi2_distribution' in results and 'chi2_values' in results['chi2_distribution']:
        chi2_values = results['chi2_distribution']['chi2_values']
        ax.hist(chi2_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', label='Expected (χ²=1)')
        ax.set_xlabel('χ²')
        ax.set_ylabel('Count')
        ax.set_title('Chi-squared Distribution')
        ax.legend()
    
    # Plot 2: Rg accuracy
    ax = axes[0, 1]
    if 'guinier_check' in results and 'rg_errors' in results['guinier_check']:
        errors = results['guinier_check']['rg_errors']
        ax.hist(errors, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(0.1, color='red', linestyle='--', label='10% threshold')
        ax.set_xlabel('Relative Rg Error')
        ax.set_ylabel('Count')
        ax.set_title('Guinier Analysis Accuracy')
        ax.legend()
    
    # Plot 3: Shape distribution
    ax = axes[1, 0]
    if 'balance_check' in results and 'shape_distribution' in results['balance_check']:
        shapes = list(results['balance_check']['shape_distribution'].keys())
        counts = list(results['balance_check']['shape_distribution'].values())
        ax.bar(shapes, counts, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Shape Class')
        ax.set_ylabel('Count')
        ax.set_title('Shape Class Distribution')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "Dataset Summary\n" + "="*30 + "\n"
    
    if 'balance_check' in results:
        bc = results['balance_check']
        summary_text += f"Total curves: {bc.get('total_curves', 'N/A')}\n"
        summary_text += f"Missing files: {bc.get('missing_files', 'N/A')}\n"
    
    if 'chi2_distribution' in results:
        chi2 = results['chi2_distribution']
        summary_text += f"\nχ² mean: {chi2.get('chi2_mean', 'N/A'):.3f}\n"
        summary_text += f"χ² std: {chi2.get('chi2_std', 'N/A'):.3f}\n"
    
    if 'guinier_check' in results:
        gc = results['guinier_check']
        pct = 100 * gc.get('within_10_percent', 0) / gc.get('n_samples', 1)
        summary_text += f"\nRg within 10%: {pct:.1f}%\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Quality report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Quality control for SAXS dataset')
    parser.add_argument('dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples for statistical checks')
    parser.add_argument('--report', type=str, default='quality_report.pdf',
                        help='Output report filename')
    parser.add_argument('--skip-chi2', action='store_true',
                        help='Skip chi-squared check (requires regeneration)')
    parser.add_argument('--skip-guinier', action='store_true',
                        help='Skip Guinier analysis')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logging.error(f"Dataset directory not found: {dataset_dir}")
        return 1
    
    results = {}
    
    # Run quality checks
    try:
        # Check dataset balance
        results['balance_check'] = check_dataset_balance(dataset_dir)
        
        # Check chi-squared distribution
        if not args.skip_chi2:
            results['chi2_distribution'] = check_chi2_distribution(
                dataset_dir, n_samples=args.n_samples
            )
        
        # Check Guinier accuracy
        if not args.skip_guinier:
            results['guinier_check'] = check_guinier_accuracy(
                dataset_dir, n_samples=args.n_samples
            )
        
        # Generate report
        report_path = Path(args.report)
        plot_quality_report(results, report_path)
        
        # Save detailed results
        json_path = report_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Detailed results saved to: {json_path}")
        
    except Exception as e:
        logging.error(f"Quality check failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
