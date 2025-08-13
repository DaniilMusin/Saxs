#!/usr/bin/env python3
"""
example_usage.py - Example of using the SAXS generator programmatically
======================================================================

This example shows how to integrate the SAXS generator into your own code
for custom workflows or integration with machine learning pipelines.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add scripts directory to path (if running from repo root)
sys.path.append('scripts')

# The public APIs used by examples are not available in current codebase.
# Keep imports optional to avoid runtime errors when running examples.
try:
    from generate import SHAPE_SPECS, ATSASWrapper, ParameterSampler
except Exception:  # noqa: BLE001
    SHAPE_SPECS = None
    ATSASWrapper = None
    ParameterSampler = None


def generate_custom_dataset():
    """Example: Generate a custom dataset with specific requirements"""
    
    print("Generating custom SAXS dataset...")
    
    # Initialize components  
    logger = logging.getLogger('custom')
    rng = np.random.default_rng(seed=42)
    if ParameterSampler is None or ATSASWrapper is None:
        print("Required APIs are not available in this build; skipping actual ATSAS generation.")
        return pd.DataFrame()
    atsas = ATSASWrapper(logger)
    sampler = ParameterSampler(rng)
    
    # Custom configuration - only spheres with specific size range
    custom_shapes = {
        'small_sphere': {
            'generator': 'bodies',
            'body_type': 7,  # Hollow sphere with r_inner=0
            'params': {
                'radius': {'type': 'linear', 'range': (50, 100)}
            },
            'polydispersity': False
        },
        'large_sphere': {
            'generator': 'bodies',
            'body_type': 7,
            'params': {
                'radius': {'type': 'linear', 'range': (200, 300)}
            },
            'polydispersity': True
        }
    }
    
    # Generate 10 curves of each type
    output_dir = Path('custom_dataset')
    output_dir.mkdir(exist_ok=True)
    
    metadata = []
    
    for shape_name, shape_spec in custom_shapes.items():
        for i in range(10):
            # Sample parameters
            params = sampler.sample(shape_name, shape_spec)
            
            # Generate theoretical scattering curve directly
            int_file = output_dir / f'{shape_name}_{i}.int'
            atsas.bodies_predict(
                body_type=shape_spec['body_type'],
                params=params,
                output_int=str(int_file),
                q_range=(0.005, 0.3),
                n_points=200
            )
            
            # Apply polydispersity if needed
            if params.get('polydispersity', 0) > 0:
                poly_file = output_dir / f'{shape_name}_{i}_poly.int'
                atsas.mixtures(
                    components=[{'file': str(int_file), 'density': 1.0}],
                    output_int=str(poly_file),
                    polydispersity={
                        'type': 'gaussian',
                        'width': params['polydispersity']
                    }
                )
                int_file = poly_file
            
            metadata.append({
                'shape': shape_name,
                'file': str(int_file),
                'params': params
            })
            
            print(f"Generated {shape_name} #{i+1}")
    
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f"Custom dataset saved to {output_dir}")
    return df


def analyze_shape_differences():
    """Example: Analyze scattering differences between shapes"""
    
    print("\nAnalyzing shape differences...")
    
    # Load some example curves (assuming they exist)
    shapes_to_compare = ['sphere', 'cylinder', 'ellipsoid']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for shape in shapes_to_compare:
        # This is a mock example - in reality, load actual .dat files
        q = np.logspace(-2, -0.5, 100)
        
        if shape == 'sphere':
            # Sphere form factor (simplified)
            I = (3 * (np.sin(q*100) - q*100*np.cos(q*100)) / (q*100)**3)**2
        elif shape == 'cylinder':
            # Cylinder (simplified approximation)
            I = 1 / (1 + (q*50)**2)
        else:
            # Ellipsoid (simplified)
            I = np.exp(-q**2 * 50**2 / 3)
        
        # Add noise
        I *= np.exp(np.random.normal(0, 0.05, len(q)))
        
        # Linear plot
        ax1.semilogy(q, I, 'o-', label=shape, alpha=0.7)
        
        # Guinier plot
        ax2.plot(q**2, np.log(I), 'o-', label=shape, alpha=0.7)
        ax2.set_xlim(0, 0.01)
    
    ax1.set_xlabel('q (Å⁻¹)')
    ax1.set_ylabel('I(q)')
    ax1.set_title('SAXS Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('q² (Å⁻²)')
    ax2.set_ylabel('ln I(q)')
    ax2.set_title('Guinier Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('shape_comparison.png', dpi=150)
    plt.show()
    
    print("Shape comparison saved to shape_comparison.png")


def batch_process_with_progress():
    """Example: Process curves with progress tracking"""
    
    from tqdm import tqdm
    import concurrent.futures
    
    print("\nBatch processing example...")
    
    def process_single_curve(params):
        """Process one curve (mock function)"""
        shape, size, seed = params
        # Simulate processing time
        import time
        time.sleep(0.1)
        
        # Return mock result
        return {
            'shape': shape,
            'size': size,
            'Rg': size * 0.775 if shape == 'sphere' else size * 0.8,
            'quality': np.random.choice(['good', 'ok', 'poor'], p=[0.7, 0.2, 0.1])
        }
    
    # Generate parameter list
    params_list = []
    for shape in ['sphere', 'cylinder', 'ellipsoid']:
        for size in [50, 100, 200, 300]:
            params_list.append((shape, size, np.random.randint(1000)))
    
    # Process in parallel with progress bar
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_curve, p): p 
                  for p in params_list}
        
        with tqdm(total=len(params_list), desc="Processing curves") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    # Analyze results
    df = pd.DataFrame(results)
    print("\nProcessing summary:")
    print(df.groupby('shape')['quality'].value_counts())


def integrate_with_ml_pipeline():
    """Example: Integration with machine learning pipeline"""
    
    print("\nML pipeline integration example...")
    
    class SAXSDataLoader:
        """Custom data loader for ML frameworks"""
        
        def __init__(self, dataset_dir: Path):
            self.dataset_dir = Path(dataset_dir)
            self.metadata = pd.read_csv(self.dataset_dir / 'meta.csv')
            
        def load_curve(self, uid: int) -> np.ndarray:
            """Load a single curve"""
            file_path = self.dataset_dir / 'saxs' / f'{uid:06d}.dat'
            data = np.loadtxt(file_path)
            return data[:, 1]  # Return I(q) column
        
        def get_batch(self, batch_size: int = 32):
            """Get a random batch of curves"""
            indices = np.random.choice(len(self.metadata), batch_size)
            
            curves = []
            labels = []
            
            for idx in indices:
                row = self.metadata.iloc[idx]
                curve = self.load_curve(row['uid'])
                label = row['shape_class']
                
                curves.append(curve)
                labels.append(label)
            
            return np.array(curves), np.array(labels)
    
    # Example usage (mock, since we don't have actual data)
    print("DataLoader initialized (mock example)")
    print("In practice, use with PyTorch/TensorFlow:")
    print()
    print("  loader = SAXSDataLoader('dataset/')")
    print("  X_batch, y_batch = loader.get_batch(32)")
    print("  model.train(X_batch, y_batch)")


def main():
    """Run all examples"""
    
    print("SAXS Generator - Usage Examples")
    print("==============================\n")
    
    # Example 1: Generate custom dataset
    # Note: This will actually generate files if ATSAS is installed
    # custom_df = generate_custom_dataset()
    
    # Example 2: Analyze shape differences (uses mock data)
    analyze_shape_differences()
    
    # Example 3: Batch processing with progress
    batch_process_with_progress()
    
    # Example 4: ML integration
    integrate_with_ml_pipeline()
    
    print("\nExamples completed!")
    print("See the code for more details on each example.")
    print("\nNote: To actually generate data, uncomment the generate_custom_dataset() call")
    print("and ensure ATSAS is installed and configured.")


if __name__ == '__main__':
    main()
