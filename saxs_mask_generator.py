#!/usr/bin/env python3
"""
create_mask.py - Create detector mask for Eiger 1M
=================================================

This script creates a basic mask file for the Dectris Eiger 1M detector.
In production, use the actual mask provided by your beamline or detector manufacturer.

The mask marks pixels that should be excluded from analysis:
- Dead pixels
- Module gaps
- Beamstop shadow
- Bad pixels from calibration
"""

import numpy as np
import argparse
from pathlib import Path


def create_eiger1m_mask(nx: int = 1030, ny: int = 1065, 
                        beamstop_radius: int = 50,
                        module_gap_width: int = 2) -> np.ndarray:
    """
    Create a basic mask for Eiger 1M detector
    
    Parameters:
    -----------
    nx, ny : int
        Detector dimensions in pixels
    beamstop_radius : int
        Radius of beamstop shadow in pixels
    module_gap_width : int
        Width of gaps between detector modules
    
    Returns:
    --------
    mask : ndarray
        Binary mask (0 = masked, 1 = valid)
    """
    
    # Initialize mask (1 = valid pixel)
    mask = np.ones((ny, nx), dtype=np.uint8)
    
    # Mask beamstop (assuming centered)
    cx, cy = nx // 2, ny // 2
    y, x = np.ogrid[:ny, :nx]
    beamstop_mask = (x - cx)**2 + (y - cy)**2 <= beamstop_radius**2
    mask[beamstop_mask] = 0
    
    # Mask module gaps (Eiger 1M has 2x2 modules)
    # Vertical gaps
    gap_x1 = nx // 2 - module_gap_width // 2
    gap_x2 = nx // 2 + module_gap_width // 2
    mask[:, gap_x1:gap_x2] = 0
    
    # Horizontal gaps  
    gap_y1 = ny // 2 - module_gap_width // 2
    gap_y2 = ny // 2 + module_gap_width // 2
    mask[gap_y1:gap_y2, :] = 0
    
    # Add some random bad pixels (simulating real detector defects)
    n_bad_pixels = int(0.001 * nx * ny)  # 0.1% bad pixels
    bad_x = np.random.randint(0, nx, n_bad_pixels)
    bad_y = np.random.randint(0, ny, n_bad_pixels)
    mask[bad_y, bad_x] = 0
    
    # Add edge masks (common in real detectors)
    edge_width = 5
    mask[:edge_width, :] = 0
    mask[-edge_width:, :] = 0
    mask[:, :edge_width] = 0
    mask[:, -edge_width:] = 0
    
    return mask


def save_mask_atsas_format(mask: np.ndarray, filename: Path) -> None:
    """
    Save mask in ATSAS format
    
    ATSAS expects masks in specific formats depending on the tool.
    This saves in a simple text format compatible with im2dat.
    """
    
    # Save as binary text file (0 and 1)
    with open(filename, 'w') as f:
        f.write(f"# Detector mask for Eiger 1M\n")
        f.write(f"# Dimensions: {mask.shape[1]} x {mask.shape[0]}\n")
        f.write(f"# Format: 0=masked, 1=valid\n")
        
        for row in mask:
            row_str = ' '.join(str(int(p)) for p in row)
            f.write(row_str + '\n')
    
    print(f"Saved mask to {filename}")
    print(f"Mask statistics:")
    print(f"  Total pixels: {mask.size}")
    print(f"  Valid pixels: {np.sum(mask)} ({100*np.sum(mask)/mask.size:.1f}%)")
    print(f"  Masked pixels: {mask.size - np.sum(mask)} ({100*(1-np.sum(mask)/mask.size):.1f}%)")


def save_mask_visualization(mask: np.ndarray, filename: Path) -> None:
    """Save a PNG visualization of the mask for inspection"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        ax.set_title('Detector Mask (white=valid, black=masked)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add colorbar
        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pixel validity')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        
        print(f"Saved mask visualization to {filename}")
        
    except ImportError:
        print("matplotlib not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description='Create detector mask for SAXS')
    parser.add_argument('--output', type=str, default='masks/eiger1m.msk',
                        help='Output mask file')
    parser.add_argument('--nx', type=int, default=1030,
                        help='Detector width in pixels')
    parser.add_argument('--ny', type=int, default=1065,
                        help='Detector height in pixels')
    parser.add_argument('--beamstop-radius', type=int, default=50,
                        help='Beamstop radius in pixels')
    parser.add_argument('--visualize', action='store_true',
                        help='Save PNG visualization of mask')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    # Generate mask
    print(f"Creating mask for {args.nx}x{args.ny} detector...")
    mask = create_eiger1m_mask(
        nx=args.nx,
        ny=args.ny,
        beamstop_radius=args.beamstop_radius
    )
    
    # Save mask
    save_mask_atsas_format(mask, output_path)
    
    # Save visualization if requested
    if args.visualize:
        viz_path = output_path.with_suffix('.png')
        save_mask_visualization(mask, viz_path)
    
    print("\nNote: This is a basic mask for testing. In production, use the")
    print("actual mask provided by your beamline or detector manufacturer.")


if __name__ == '__main__':
    main()
