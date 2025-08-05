# SAXS Dataset Generator with ATSAS

This project generates synthetic SAXS (Small-Angle X-ray Scattering) datasets for machine learning applications, following the methodology of Monge et al. 2024 but using the ATSAS suite.

## How It Works

The generator creates physically accurate SAXS curves by:

1. **Generating theoretical scattering curves** using ATSAS `bodies` in predict mode
   - Simple shapes: direct I(q) calculation via interactive `bodies` interface
   - Core-shell structures: component curves combined with `mixtures`
   - Polydispersity: applied via `mixtures` (Gaussian distribution 0-30%)

2. **Simulating realistic detector images** using `imsim`
   - Includes beam profile, detector geometry, noise characteristics
   - Two instrument configurations for different q-ranges

3. **Converting to 1D curves with error estimates** using `im2dat`
   - Radial averaging with proper error propagation
   - Detector mask applied to exclude bad pixels

The pipeline uses `bodies` in interactive mode (not CLI flags) and generates I(q) directly without PDB intermediates for efficiency.

## Requirements

### Software
- **ATSAS ≥ 3.0** (command-line tools: `bodies`, `mixtures`, `imsim`, `im2dat`, `autorg`)
  - Note: `mixture` in ATSAS 3.x, `mixtures` in ATSAS 4.x
  - `crysol` is not required as `bodies` generates I(q) directly in predict mode
- **Python 3.10+** with packages:
  - pandas
  - numpy
  - pyyaml
  - tqdm
- **GNU Parallel** (optional, for enhanced parallelization)

### Important Notes on ATSAS Versions

- **ATSAS 3.x**: Uses `mixture` for combining components
- **ATSAS 4.x**: Uses `mixtures` (merged `mixture`, `bilmix`, `lipmix`)
- The scripts automatically detect which version is available
- Interactive mode is required for `bodies` (no CLI flags for shape parameters)
- License typically limits to 2 parallel ATSAS processes

### System
- Linux (recommended) or Windows with WSL
- At least 16 GB RAM for parallel processing
- ~50 GB disk space for full dataset

## Installation

1. Install ATSAS from https://www.embl-hamburg.de/biosaxs/software.html
2. Ensure ATSAS binaries are in your PATH
3. Install Python dependencies:
   ```bash
   pip install pandas numpy pyyaml tqdm
   ```

## Project Structure

```
project/
├── configs/
│   ├── xeuss.yml       # Xeuss 1800HR configuration
│   └── nanoinx.yml     # NanoInXider HR configuration
├── masks/
│   └── eiger1m.msk     # Detector mask file (obtain from detector manufacturer)
├── scripts/
│   └── generate.py     # Main generation script
├── dataset/            # Output directory (created by script)
│   ├── saxs/          # Individual .dat files
│   ├── meta.csv       # Metadata for all curves
│   └── logs/          # Processing logs
└── README.md
```

## Usage

### Basic Generation

Generate a balanced dataset with default parameters:

```bash
python scripts/generate.py --n 37656 --jobs 20 --atsas-parallel 2 --out dataset/
```

### Parameters

- `--n`: Total number of curves to generate (default: 37656)
- `--jobs`: Number of parallel processes (default: 20)
- `--atsas-parallel`: Max parallel ATSAS processes due to license (default: 2)
- `--out`: Output directory (default: dataset/)
- `--seed`: Random seed for reproducibility (default: 42)
- `--retry`: Path to file with failed UIDs to retry

**Note**: ATSAS typically has a license limit of 2 parallel processes. The script respects this limit while using additional parallelism for non-ATSAS operations.

### Retry Failed Curves

If some curves fail to generate:

```bash
python scripts/generate.py --retry dataset/failed.txt --jobs 10 --out dataset/
```

## Shape Classes and Parameters

The generator creates 9 balanced shape classes (4184 curves each by default):

| Shape Class | Generator | Implementation | Parameters | Polydispersity |
|------------|-----------|----------------|------------|----------------|
| **Sphere** | bodies | Hollow sphere (r_inner=0) | radius: 50-1000 Å (log) | 0-30% via mixtures |
| **Cylinder** | bodies | Type 3 | R: 50-1000 Å (log)<br>L: 1-2×R | 0-30% via mixtures |
| **Oblate ellipsoid** | bodies | Type 1 | a: 50-1000 Å (log)<br>c/a: 0.1-0.77 | 0-30% via mixtures |
| **Prolate ellipsoid** | bodies | Type 1 | a: 50-1000 Å (log)<br>c/a: 1.3-5.0 | 0-30% via mixtures |
| **Core-shell sphere** | mixtures | Two components | r_core + t_shell ≤ 1000 Å | 0-30% |
| **Hollow sphere** | bodies | Type 7 | r_outer: 70-1000 Å<br>r_inner: 0.5-0.9×r_outer | 0-30% via mixtures |
| **Core-shell oblate** | mixtures | Two components | Similar to oblate | 0-30% |
| **Core-shell prolate** | mixtures | Two components | Similar to prolate | 0-30% |
| **Core-shell cylinder** | mixtures | Two components | r_core + t ≤ 1000 Å | 0-30% |

**Note**: Core-shell structures are not directly supported by `bodies` and are created by combining components using `mixtures`.

## Instrument Configurations

Two instrument setups are randomly selected for each curve:

| Instrument | q-range (Å⁻¹) | Beam FWHM | Flux (ph/s) | Purpose |
|------------|---------------|-----------|-------------|---------|
| **Xeuss 1800HR** | 0.0031-0.149 | 0.0016 | 3.43×10⁶ | Low q, high resolution |
| **NanoInXider HR** | 0.0019-0.445 | 0.0024 | 7.22×10⁶ | Wide q range |

## Output Format

### SAXS Curves (`dataset/saxs/*.dat`)
Standard ATSAS format with three columns:
- q (Å⁻¹): scattering vector
- I(q): intensity
- σ(q): error estimate

### Metadata (`dataset/meta.csv`)
CSV file with columns:
- `uid`: Unique identifier
- `shape_class`: One of the 9 shape types
- `generator`: Tool used (bodies/mixture)
- `true_params`: JSON-encoded shape parameters
- `instrument_cfg`: Xeuss1800HR or NanoInXider HR
- `seed`: Random seed used
- `timestamp`: Generation time

### Logs (`dataset/logs/*.log`)
Individual log files for each curve containing processing details and any errors.

## Quality Control

To verify dataset quality:

1. **Compare with theory**: First 100 curves should have χ² ≈ 1.0 when compared to noiseless `crysol` output
2. **Guinier analysis**: `autorg` should return Rg within 10% of true radius
3. **ML validation**: Train classifier, expect >90% accuracy on shape classification

## Performance Tips

- Use SSD for temporary files (set TMPDIR environment variable)
- Adjust `--jobs` based on available CPU cores and RAM
- Cache `crysol` output if regenerating with different noise levels
- Consider using SLURM for cluster deployment

## Troubleshooting

### Common Issues

1. **"ATSAS tool not found"**: Ensure ATSAS binaries are in PATH
2. **License errors**: Check ATSAS license file permissions and parallel limit
3. **Memory errors**: Reduce `--jobs` parameter
4. **Slow generation**: Profile with `--jobs 1` to identify bottlenecks
5. **Interactive mode failures**: See `scripts/demo_interactive.py` for examples

### Interactive Mode

ATSAS tools like `bodies` work interactively, not via CLI flags. To understand the correct input format:

```bash
python scripts/demo_interactive.py
```

This will demonstrate:
- Correct interactive input for `bodies`
- Using `mixtures` for polydispersity
- Creating core-shell structures

### Debug Mode

For detailed debugging, check individual log files in `dataset/logs/`

## Extension

To add new shapes:
1. Add specification to `SHAPE_SPECS` in `generate.py`
2. Implement parameter sampling logic
3. Update this README

To add new instruments:
1. Create new YAML config in `configs/`
2. Add to `configs` dictionary in `generate.py`

## Citation

If you use this dataset generator, please cite:
- Your paper (when published)
- ATSAS suite: Manalastas-Cantos et al., J. Appl. Cryst. (2021)
- Methodology: Monge et al., Acta Cryst. A (2024)

## License

This code is provided as-is for research purposes. ATSAS has its own licensing terms.