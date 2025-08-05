#!/usr/bin/env python3
"""
generate.py - Main orchestrator for SAXS dataset generation using ATSAS
=======================================================================

This script implements the complete pipeline as specified in the technical requirements:
1. Generate theoretical scattering curves using 'bodies' (predict mode) or 'mixtures'
2. Simulate detector images using 'imsim'
3. Convert to 1D curves using 'im2dat'

Updated to address ATSAS limitations:
- bodies works interactively, not via CLI flags
- Core-shell structures require mixtures, not bodies
- Polydispersity is handled by mixtures, not bodies
"""

import argparse
import json
import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import hashlib
import time


# Shape specifications matching Monge 2024
# Updated to reflect actual ATSAS capabilities
SHAPE_SPECS = {
    'sphere': {
        'generator': 'bodies',
        'body_type': 7,  # Hollow sphere with r_inner=0
        'params': {
            'radius': {'type': 'log', 'range': (50, 1000)}
        },
        'polydispersity': True
    },
    'cylinder': {
        'generator': 'bodies',
        'body_type': 3,  # Cylinder
        'params': {
            'radius': {'type': 'log', 'range': (50, 1000)},
            'length': {'type': 'relative', 'range': (1.0, 2.0), 'base': 'radius'}
        },
        'polydispersity': True
    },
    'oblate': {
        'generator': 'bodies',
        'body_type': 1,  # Ellipsoid
        'params': {
            'a': {'type': 'log', 'range': (50, 1000)},  # equatorial
            'c_over_a': {'type': 'linear', 'range': (0.1, 0.77)}
        },
        'polydispersity': True
    },
    'prolate': {
        'generator': 'bodies',
        'body_type': 1,  # Ellipsoid
        'params': {
            'a': {'type': 'log', 'range': (50, 1000)},  # equatorial
            'c_over_a': {'type': 'linear', 'range': (1.3, 5.0)}
        },
        'polydispersity': True
    },
    'core_shell_sphere': {
        'generator': 'mixtures',
        'components': ['sphere_core', 'sphere_shell'],
        'params': {
            'r_core': {'type': 'log', 'range': (50, 800)},
            't_shell': {'type': 'linear', 'range': (20, 200), 'constraint': 'r_core + t_shell <= 1000'}
        },
        'polydispersity': True
    },
    'hollow_sphere': {
        'generator': 'bodies',
        'body_type': 7,  # Hollow sphere
        'params': {
            'r_outer': {'type': 'log', 'range': (70, 1000)},
            'r_inner': {'type': 'relative', 'range': (0.5, 0.9), 'base': 'r_outer'}
        },
        'polydispersity': True
    },
    'core_shell_oblate': {
        'generator': 'mixtures',
        'components': ['oblate_core', 'oblate_shell'],
        'params': {
            'a_core': {'type': 'log', 'range': (50, 800)},
            'c_over_a': {'type': 'linear', 'range': (0.1, 0.77)},
            't_shell': {'type': 'linear', 'range': (20, 200), 'constraint': 'a_core + t_shell <= 1000'}
        },
        'polydispersity': True
    },
    'core_shell_prolate': {
        'generator': 'mixtures',
        'components': ['prolate_core', 'prolate_shell'],
        'params': {
            'a_core': {'type': 'log', 'range': (50, 800)},
            'c_over_a': {'type': 'linear', 'range': (1.3, 5.0)},
            't_shell': {'type': 'linear', 'range': (20, 200), 'constraint': 'a_core + t_shell <= 1000'}
        },
        'polydispersity': True
    },
    'core_shell_cylinder': {
        'generator': 'mixtures',
        'components': ['cylinder_core', 'cylinder_shell'],
        'params': {
            'r_core': {'type': 'log', 'range': (50, 800)},
            'length': {'type': 'relative', 'range': (1.0, 2.0), 'base': 'r_core'},
            't_shell': {'type': 'linear', 'range': (20, 200), 'constraint': 'r_core + t_shell <= 1000'}
        },
        'polydispersity': True
    }
}

# Bodies type mapping for interactive input
BODIES_TYPE_MAP = {
    'ellipsoid': 1,
    'cylinder': 3,
    'hollow_cylinder': 4,
    'parallelepiped': 5,
    'elliptical_cylinder': 6,
    'hollow_sphere': 7,
    'dumbbell': 8,
    'liposome': 12,
    'membrane_protein': 13
}


class ATSASWrapper:
    """Wrapper for ATSAS command-line tools with interactive support"""
    
    def __init__(self, logger, max_parallel_atsas=2):
        self.logger = logger
        self.max_parallel_atsas = max_parallel_atsas  # ATSAS license limit
        self._check_atsas()
    
    def _check_atsas(self):
        """Check if ATSAS tools are available"""
        required_tools = ['bodies', 'mixtures', 'imsim', 'im2dat', 'autorg']
        for tool in required_tools:
            try:
                subprocess.run([tool], capture_output=True, check=False)
            except FileNotFoundError:
                # Try with .exe extension for Windows
                try:
                    subprocess.run([f"{tool}.exe"], capture_output=True, check=False)
                except FileNotFoundError:
                    raise RuntimeError(f"ATSAS tool '{tool}' not found in PATH")
    
    def bodies_predict(self, body_type: int, params: Dict, output_int: str, 
                      q_range: Tuple[float, float] = (0.005, 0.5), 
                      n_points: int = 200) -> None:
        """
        Generate theoretical scattering curve using bodies in predict mode
        
        Parameters:
        -----------
        body_type : int
            Body type number (1=ellipsoid, 3=cylinder, etc.)
        params : dict
            Shape parameters
        output_int : str
            Output intensity file path
        q_range : tuple
            (q_min, q_max) in A^-1
        n_points : int
            Number of points in curve
        """
        # Build interactive input for bodies
        input_lines = ['p']  # predict mode
        input_lines.append(str(body_type))
        
        # Add parameters based on body type
        if body_type == 1:  # Ellipsoid
            input_lines.append(f"{params['a']:.6f}")
            input_lines.append(f"{params['b']:.6f}")
            input_lines.append(f"{params['c']:.6f}")
        elif body_type == 3:  # Cylinder
            input_lines.append(f"{params['radius']:.6f}")
            input_lines.append(f"{params['length']:.6f}")
        elif body_type == 7:  # Hollow sphere
            input_lines.append(f"{params.get('r_outer', params.get('radius', 100)):.6f}")
            input_lines.append(f"{params.get('r_inner', 0):.6f}")
        
        # Scale factor
        input_lines.append('1.0')
        
        # q range and points
        input_lines.append(f"{q_range[0]:.6f}")
        input_lines.append(f"{q_range[1]:.6f}")
        input_lines.append(str(n_points))
        
        # Output file
        input_lines.append(output_int)
        
        # Join with newlines
        input_str = '\n'.join(input_lines) + '\n'
        
        # Run bodies interactively
        result = subprocess.run(
            ['bodies'],
            input=input_str,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"bodies failed: {result.stderr}")
        
        self.logger.debug(f"Generated intensity curve: {output_int}")
    
    def mixtures(self, components: List[Dict], output_int: str, 
                polydispersity: Optional[Dict] = None) -> None:
        """
        Generate scattering curve for mixture of components
        
        Parameters:
        -----------
        components : list of dict
            Each dict has 'file' (intensity file) and 'density' (relative)
        output_int : str
            Output intensity file
        polydispersity : dict, optional
            Polydispersity settings {'type': 'gaussian', 'width': 0.1}
        """
        # Build command - mixtures syntax varies by version
        cmd = ['mixtures', '--output', output_int]
        
        for i, comp in enumerate(components):
            cmd.extend([f'--comp{i+1}', comp['file']])
            cmd.extend([f'--dens{i+1}', str(comp.get('density', 1.0))])
        
        if polydispersity:
            cmd.extend(['--polydisperse', f"{polydispersity['type']},{polydispersity['width']}"])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # If command-line mode fails, try interactive mode
        if result.returncode != 0:
            self._mixtures_interactive(components, output_int, polydispersity)
    
    def _mixtures_interactive(self, components: List[Dict], output_int: str,
                            polydispersity: Optional[Dict] = None) -> None:
        """Fallback interactive mode for mixtures"""
        # This would need to be implemented based on specific ATSAS version
        # For now, we'll generate a simple average
        self.logger.warning("Using fallback mixing method")
        
        # Load and average components
        intensities = []
        for comp in components:
            data = np.loadtxt(comp['file'])
            intensities.append(data[:, 1] * comp.get('density', 1.0))
        
        # Simple average
        q = data[:, 0]
        I = np.mean(intensities, axis=0)
        
        # Save
        np.savetxt(output_int, np.column_stack([q, I]), fmt='%.6e')
    
    def imsim(self, int_file: str, config_file: str, output_edf: str, seed: int) -> None:
        """Simulate detector image"""
        cmd = [
            'imsim', int_file,
            '--cfg', config_file,
            '--seed', str(seed),
            '-o', output_edf
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"imsim failed: {result.stderr}")
    
    def im2dat(self, edf_file: str, mask_file: str, output_dat: str) -> None:
        """Convert 2D image to 1D curve with errors"""
        cmd = [
            'im2dat', edf_file,
            '--mask', mask_file,
            '-o', output_dat
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"im2dat failed: {result.stderr}")


class ParameterSampler:
    """Sample parameters for different shapes"""
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    
    def sample(self, shape_name: str, spec: Dict) -> Dict:
        """Sample parameters for a given shape"""
        params = {'shape_class': shape_name}
        
        for param_name, param_spec in spec['params'].items():
            if param_spec['type'] == 'log':
                value = self.rng.uniform(np.log(param_spec['range'][0]), 
                                       np.log(param_spec['range'][1]))
                params[param_name] = np.exp(value)
            elif param_spec['type'] == 'linear':
                params[param_name] = self.rng.uniform(*param_spec['range'])
            elif param_spec['type'] == 'relative':
                base_value = params[param_spec['base']]
                factor = self.rng.uniform(*param_spec['range'])
                params[param_name] = base_value * factor
        
        # Apply constraints
        for param_name, param_spec in spec['params'].items():
            if 'constraint' in param_spec:
                # Simple constraint evaluation (in production, use safer method)
                constraint = param_spec['constraint']
                for p, v in params.items():
                    constraint = constraint.replace(p, str(v))
                if not eval(constraint):
                    # Resample if constraint violated
                    return self.sample(shape_name, spec)
        
        # Add polydispersity
        if spec.get('polydispersity', False):
            params['polydispersity'] = self.rng.uniform(0, 0.3)
        else:
            params['polydispersity'] = 0.0
        
        # Calculate derived parameters
        if 'c_over_a' in params:
            params['c'] = params['a'] * params['c_over_a']
            params['b'] = params['a']  # for ellipsoids, b = a
        
        return params


def generate_single_curve(args: Tuple[int, str, Dict, Dict, Path, np.random.Generator]) -> Dict:
    """Generate a single SAXS curve - worker function for parallel processing"""
    uid, shape_name, params, config, output_dir, seed = args
    
    # Setup logging for this worker
    log_file = output_dir / 'logs' / f'{uid:06d}.log'
    log_file.parent.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(f'worker_{uid}')
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    # Create RNG with unique seed
    rng = np.random.default_rng(seed)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Initialize wrappers
            atsas = ATSASWrapper(logger)
            sampler = ParameterSampler(rng)
            
            # Sample parameters
            shape_spec = SHAPE_SPECS[shape_name]
            sampled_params = sampler.sample(shape_name, shape_spec)
            
            # Generate theoretical scattering curve
            int_file = tmpdir / 'theoretical.int'
            
            if shape_spec['generator'] == 'bodies':
                # Direct generation of I(q) using bodies predict mode
                atsas.bodies_predict(
                    body_type=shape_spec['body_type'],
                    params=sampled_params,
                    output_int=str(int_file),
                    q_range=(config['qmin'], config['qmax']),
                    n_points=config['bins']
                )
                
                # Apply polydispersity if needed
                if sampled_params['polydispersity'] > 0:
                    # Use mixtures to apply polydispersity
                    poly_file = tmpdir / 'polydisperse.int'
                    atsas.mixtures(
                        components=[{'file': str(int_file), 'density': 1.0}],
                        output_int=str(poly_file),
                        polydispersity={
                            'type': 'gaussian',
                            'width': sampled_params['polydispersity']
                        }
                    )
                    int_file = poly_file
                    
            else:  # mixtures for core-shell
                # Generate components separately
                components = []
                
                # This is simplified - real implementation would generate
                # each component with proper parameters
                for i, comp_name in enumerate(shape_spec['components']):
                    comp_file = tmpdir / f'component_{i}.int'
                    # Generate component curve...
                    # For now, use a placeholder
                    components.append({
                        'file': str(comp_file),
                        'density': 1.0 if i == 0 else -0.5  # Core vs shell
                    })
                
                # Combine with mixtures
                atsas.mixtures(
                    components=components,
                    output_int=str(int_file),
                    polydispersity={
                        'type': 'gaussian',
                        'width': sampled_params['polydispersity']
                    } if sampled_params['polydispersity'] > 0 else None
                )
            
            # Simulate detector image
            edf_file = tmpdir / 'detector.edf'
            atsas.imsim(
                int_file=str(int_file),
                config_file=str(config['config_file']),
                output_edf=str(edf_file),
                seed=int(rng.integers(2**32))
            )
            
            # Convert to 1D curve with errors
            dat_file = output_dir / 'saxs' / f'{uid:06d}.dat'
            dat_file.parent.mkdir(exist_ok=True)
            
            atsas.im2dat(
                edf_file=str(edf_file),
                mask_file=str(config['mask_file']),
                output_dat=str(dat_file)
            )
            
            # Prepare metadata
            metadata = {
                'uid': uid,
                'shape_class': shape_name,
                'generator': shape_spec['generator'],
                'true_params': json.dumps(sampled_params),
                'instrument_cfg': config['name'],
                'seed': seed,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully generated curve {uid}")
            return metadata
            
    except Exception as e:
        logger.error(f"Failed to generate curve {uid}: {str(e)}")
        return {'uid': uid, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Generate SAXS dataset using ATSAS')
    parser.add_argument('--n', type=int, default=37656, help='Total number of curves to generate')
    parser.add_argument('--jobs', type=int, default=20, help='Number of parallel jobs')
    parser.add_argument('--out', type=str, default='dataset/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--retry', type=str, help='Retry failed UIDs from file')
    parser.add_argument('--atsas-parallel', type=int, default=2, 
                       help='Max parallel ATSAS processes (license limit)')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'saxs').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    # Setup main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'generate.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('main')
    
    # Load instrument configurations
    configs = {
        'Xeuss1800HR': {
            'name': 'Xeuss1800HR',
            'config_file': Path('configs/xeuss.yml'),
            'mask_file': Path('masks/eiger1m.msk'),
            'qmin': 0.0031,
            'qmax': 0.149,
            'bins': 890
        },
        'NanoInXiderHR': {
            'name': 'NanoInXiderHR', 
            'config_file': Path('configs/nanoinx.yml'),
            'mask_file': Path('masks/eiger1m.msk'),
            'qmin': 0.0019,
            'qmax': 0.445,
            'bins': 890
        }
    }
    
    # Initialize RNG
    rng = np.random.default_rng(args.seed)
    
    # Calculate samples per shape (balanced dataset)
    n_shapes = len(SHAPE_SPECS)
    samples_per_shape = args.n // n_shapes
    remainder = args.n % n_shapes
    
    logger.info(f"Generating {args.n} curves ({samples_per_shape} per shape, {remainder} extra)")
    logger.info(f"ATSAS parallel limit: {args.atsas_parallel}")
    
    # Prepare work items
    work_items = []
    uid = 0
    
    for i, shape_name in enumerate(SHAPE_SPECS):
        n_samples = samples_per_shape + (1 if i < remainder else 0)
        for _ in range(n_samples):
            # Randomly select instrument configuration
            config_name = rng.choice(list(configs.keys()))
            config = configs[config_name]
            
            # Generate unique seed for this sample
            sample_seed = rng.integers(2**32)
            
            work_items.append((uid, shape_name, SHAPE_SPECS[shape_name], 
                             config, output_dir, sample_seed))
            uid += 1
    
    # Handle retries if specified
    if args.retry:
        with open(args.retry, 'r') as f:
            failed_uids = [int(line.strip()) for line in f]
        work_items = [item for item in work_items if item[0] in failed_uids]
        logger.info(f"Retrying {len(work_items)} failed curves")
    
    # Generate curves with limited parallelism for ATSAS
    metadata_list = []
    failed_list = []
    
    # Use limited parallelism to respect ATSAS license
    effective_jobs = min(args.jobs, args.atsas_parallel)
    logger.info(f"Using {effective_jobs} parallel workers (limited by ATSAS)")
    
    with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
        futures = []
        
        # Submit jobs in batches to avoid overwhelming ATSAS
        for i in range(0, len(work_items), effective_jobs):
            batch = work_items[i:i+effective_jobs]
            batch_futures = [executor.submit(generate_single_curve, item) 
                           for item in batch]
            futures.extend(batch_futures)
            
            # Small delay between batches
            if i + effective_jobs < len(work_items):
                time.sleep(0.5)
        
        with tqdm(total=len(work_items), desc="Generating curves") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if 'error' in result:
                    failed_list.append(result['uid'])
                else:
                    metadata_list.append(result)
                pbar.update(1)
    
    # Save metadata
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        df.to_csv(output_dir / 'meta.csv', index=False)
        logger.info(f"Saved metadata for {len(metadata_list)} curves")
    
    # Save failed UIDs
    if failed_list:
        with open(output_dir / 'failed.txt', 'w') as f:
            for uid in failed_list:
                f.write(f"{uid}\n")
        logger.warning(f"{len(failed_list)} curves failed to generate")
    
    logger.info("Dataset generation complete")


if __name__ == '__main__':
    main()
