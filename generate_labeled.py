#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced generator with meaningful file names including shape type and parameters.
"""
import sys
import json
import random
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from param_sampler import available_shapes, draw_shape

def create_mock_saxs_data(q_range=(0.01, 0.5), n_points=100):
    """Create mock SAXS intensity data"""
    q = np.logspace(np.log10(q_range[0]), np.log10(q_range[1]), n_points)
    # Mock Guinier-like scattering
    I = np.exp(-q**2 * random.uniform(100, 1000)) * random.uniform(1e6, 1e8)
    # Add some noise
    noise = I * 0.01 * np.random.random(len(I))
    I_noise = I + noise
    errors = np.sqrt(I_noise) + I_noise * 0.01
    return q, I_noise, errors

def format_params_for_filename(shape_class, params):
    """Format parameters for filename"""
    if shape_class == "sphere":
        return f"R{params['radius']:.0f}"
    elif shape_class == "hollow_sphere":
        return f"R{params['outer_radius']:.0f}_r{params['inner_radius']:.0f}"
    elif shape_class == "cylinder":
        return f"R{params['radius']:.0f}_L{params['length']:.0f}"
    elif shape_class == "hollow_cylinder":
        return f"R{params['outer_radius']:.0f}_r{params['inner_radius']:.0f}_L{params['length']:.0f}"
    elif shape_class == "elliptical_cylinder":
        return f"A{params['a']:.0f}_C{params['c']:.0f}_L{params['length']:.0f}"
    elif shape_class in ("oblate", "prolate"):
        return f"A{params['a']:.0f}_C{params['c']:.0f}"
    elif shape_class == "parallelepiped":
        return f"A{params['a']:.0f}_B{params['b']:.0f}_C{params['c']:.0f}"
    elif shape_class == "ellipsoid_triaxial":
        return f"A{params['a']:.0f}_B{params['b']:.0f}_C{params['c']:.0f}"
    elif shape_class == "ellipsoid_of_rotation":
        return f"A{params['a']:.0f}_ratio{params['ratio']:.2f}"
    elif shape_class == "dumbbell":
        return f"R1_{params['r1']:.0f}_R2_{params['r2']:.0f}_D{params['center_distance']:.0f}"
    elif shape_class == "liposome":
        return f"R{params['outer_radius']:.0f}_t{params['bilayer_thickness']:.0f}"
    elif shape_class == "membrane_protein":
        return f"Rm{params['rmemb']:.0f}_Rt{params['rtail']:.0f}_Rh{params['rhead']:.0f}"
    elif shape_class == "core_shell_sphere":
        return f"Rc{params['core_radius']:.0f}_t{params['shell_thickness']:.0f}"
    elif shape_class in ("core_shell_oblate", "core_shell_prolate"):
        return f"Rc{params['equatorial_core_radius']:.0f}_t{params['shell_thickness']:.0f}_ar{params['axial_ratio']:.2f}"
    elif shape_class == "core_shell_cylinder":
        return f"Rc{params['core_radius']:.0f}_t{params['shell_thickness']:.0f}_L{params['length']:.0f}"
    else:
        return "params"

def main():
    print("Starting labeled generation of 5k samples per shape type...")
    print("Files will include shape type and parameters in filename")
    
    out_dir = Path("dataset_all_shapes_5k_labeled")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "saxs").mkdir(exist_ok=True)
    
    shapes = available_shapes()
    print(f"Will generate 5000 samples for each of {len(shapes)} shape types")
    print(f"Output directory: {out_dir.absolute()}")
    
    meta_rows = []
    rng = random.Random(42)
    
    uid = 0
    for shape_idx, shape in enumerate(shapes):
        print(f"Generating {shape} ({shape_idx+1}/{len(shapes)})...")
        
        for i in range(5000):
            sample = draw_shape(rng, shape=shape)
            
            # Create mock SAXS data
            q, I, err = create_mock_saxs_data()
            
            # Create descriptive filename
            param_str = format_params_for_filename(shape, sample.params)
            filename = f"{uid:06d}__{shape}__{param_str}__nanoinx.dat"
            
            # Write .dat file
            dat_file = out_dir / "saxs" / filename
            with open(dat_file, 'w') as f:
                f.write(f'Sample description: {shape} #{i+1} - {param_str}\n')
                f.write('Sample:   c= 1.000 mg/ml  Code: \n')
                for qi, Ii, erri in zip(q, I, err):
                    f.write(f'{qi:.6e}   {Ii:.6e}   {erri:.6e}\n')
            
            # Add to metadata
            meta_rows.append({
                "uid": uid,
                "filename": filename,
                "shape_class": sample.shape_class,
                "generator": "pythonDAM+demo_saxs_labeled",
                "true_params": json.dumps(sample.params, ensure_ascii=False),
                "param_summary": param_str,
                "instrument_cfg": "nanoinx",
                "seed": 42 + uid
            })
            
            uid += 1
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/5000 samples generated")
    
    # Write metadata
    df = pd.DataFrame(meta_rows)
    meta_file = out_dir / "meta.csv"
    df.to_csv(meta_file, index=False)
    
    print("Generation complete!")
    print(f"Total samples generated: {len(meta_rows)}")
    print(f"Metadata saved to: {meta_file}")
    print(f"SAXS curves saved to: {out_dir / 'saxs'}")
    
    # Statistics
    shape_counts = df['shape_class'].value_counts()
    print("\nShape distribution:")
    for shape, count in shape_counts.items():
        print(f"  {shape}: {count} samples")
    
    # Example filenames
    print("\nExample filenames:")
    for shape in shapes[:5]:
        examples = df[df['shape_class'] == shape]['filename'].head(2)
        for filename in examples:
            print(f"  {filename}")

if __name__ == "__main__":
    main()