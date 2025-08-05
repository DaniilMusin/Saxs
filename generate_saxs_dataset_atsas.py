"""
generate_saxs_dataset_atsas.py
=============================

This script automates the generation of large numbers of synthetic SAXS
profiles using the `bodies` program from the ATSAS suite.  It supports
multiple body types (sphere, cylinder, ellipsoid, etc.), samples random
geometrical parameters, drives the interactive `bodies` dialogue, applies
standard preprocessing (logarithmic transform, integral normalisation,
standardisation and log‑space resampling) and finally stores the processed
data and labels to disk.  The design follows the workflow outlined in
Monge et al. and our previous answer.

**Note:** This script assumes that ATSAS is installed and that the
`bodies` executable is on your system PATH.  Because ATSAS is not
available in this environment, the script cannot be executed here.
However, it provides a ready‑to‑run template for your own machine.

To generate ~73 000 profiles similar to Monge et al., adjust the
`samples_per_shape` parameter accordingly (e.g. 73000 // len(SHAPE_SPECS)).
Be aware that generating tens of thousands of profiles is computationally
expensive and may take hours; consider parallelising across multiple
processes.

Usage:

    python3 generate_saxs_dataset_atsas.py --output-dir saxs_dataset_large \
        --samples-per-shape 8000 --points 256

This will generate 3 × 8000 = 24 000 samples (for three shapes defined
below), each resampled to 256 points.  The processed curves are saved
as ``X.npy`` and labels as ``y.npy`` in the specified output directory.
Metadata (shape parameters) are stored in ``metadata.csv``.
"""

import argparse
import csv
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class BodySpec:
    """Specification of a geometric body supported by `bodies`."""
    name: str
    body_number: int
    param_names: List[str]
    param_ranges: Dict[str, Tuple[float, float]]
    label: int


# Mapping of supported bodies and their parameters.
# Extend this list if you need additional shapes (e.g. hollow sphere).
SHAPE_SPECS: List[BodySpec] = [
    BodySpec(
        name="sphere",
        body_number=7,  # hollow-sphere with inner radius 0 is a solid sphere
        param_names=["outer_radius", "inner_radius"],
        param_ranges={"outer_radius": (30.0, 80.0), "inner_radius": (0.0, 0.0)},
        label=0,
    ),
    BodySpec(
        name="cylinder",
        body_number=3,
        param_names=["radius", "height"],
        param_ranges={"radius": (20.0, 60.0), "height": (50.0, 150.0)},
        label=1,
    ),
    BodySpec(
        name="ellipsoid",
        body_number=1,
        param_names=["a", "b", "c"],
        param_ranges={"a": (20.0, 60.0), "b": (20.0, 60.0), "c": (20.0, 60.0)},
        label=2,
    ),
]


def sample_parameters(spec: BodySpec) -> Dict[str, float]:
    """Sample random parameter values within the specified ranges."""
    params = {}
    for name in spec.param_names:
        low, high = spec.param_ranges[name]
        params[name] = np.random.uniform(low, high)
    return params


def build_bodies_input(
    spec: BodySpec,
    params: Dict[str, float],
    q_min: float,
    q_max: float,
    n_points: int,
    output_path: str,
) -> str:
    """Construct a newline‑separated answer string for the interactive bodies run."""
    lines: List[str] = []
    # Operation mode: predict ('p')
    lines.append("p")
    # Select body type by number
    lines.append(str(spec.body_number))
    # Fill in parameter values
    for name in spec.param_names:
        value = params[name]
        # Format floats with 6 decimal places to avoid scientific notation
        lines.append(f"{value:.6f}")
    # scale factor for intensity (keep 1)
    lines.append("1.0")
    # q_min and q_max
    lines.append(f"{q_min:.6f}")
    lines.append(f"{q_max:.6f}")
    # number of points
    lines.append(str(n_points))
    # Output filename
    lines.append(output_path)
    # Terminate with an empty line or newline; some versions require this
    lines.append("")
    return "\n".join(lines)


def run_bodies(input_str: str) -> None:
    """Execute the `bodies` program with the provided input string."""
    proc = subprocess.run(
        ["bodies"],
        input=input_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bodies failed with exit code {proc.returncode}:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )


def preprocess_curve(
    q: np.ndarray,
    intensity: np.ndarray,
    q_min: float,
    q_max: float,
    n_resample: int,
) -> np.ndarray:
    """Apply log, integral normalisation, standardisation and resampling."""
    intensity = np.maximum(intensity, 1e-12)
    logI = np.log10(intensity)
    # Integral normalisation
    area = np.trapz(intensity, q)
    if area > 0:
        logI /= area
    # Standardise
    logI = (logI - np.mean(logI)) / np.std(logI)
    # Resample to log‑spaced q grid
    q_new = np.logspace(np.log10(q_min), np.log10(q_max), n_resample)
    logI_interp = np.interp(q_new, q, logI)
    return logI_interp


def generate_dataset(
    specs: Iterable[BodySpec],
    samples_per_shape: int,
    q_min: float,
    q_max: float,
    n_points_raw: int,
    n_points_resampled: int,
    output_dir: str,
    random_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset by driving ATSAS `bodies` for each shape and sample.

    Parameters
    ----------
    specs : iterable of BodySpec
        Body specifications to generate.
    samples_per_shape : int
        Number of samples to generate per body type.
    q_min, q_max : float
        Range of q values in Å⁻¹ for generated curves.
    n_points_raw : int
        Number of points generated by `bodies` before resampling.
    n_points_resampled : int
        Number of points after resampling.
    output_dir : str
        Directory where ``X.npy``, ``y.npy`` and ``metadata.csv`` will be saved.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray
        Processed curves of shape (n_samples, n_points_resampled).
    y : ndarray
        Integer labels of length n_samples.
    """
    np.random.seed(random_seed)
    total_samples = len(specs) * samples_per_shape
    X = np.zeros((total_samples, n_points_resampled), dtype=np.float32)
    y = np.zeros((total_samples,), dtype=np.int32)
    metadata: List[Dict[str, float]] = []
    sample_idx = 0
    # Temporary working directory for raw curves
    with tempfile.TemporaryDirectory() as tmpdir:
        for spec in specs:
            for _ in range(samples_per_shape):
                params = sample_parameters(spec)
                tmp_file = os.path.join(tmpdir, f"curve_{sample_idx}.dat")
                input_str = build_bodies_input(
                    spec=spec,
                    params=params,
                    q_min=q_min,
                    q_max=q_max,
                    n_points=n_points_raw,
                    output_path=tmp_file,
                )
                # Run bodies
                run_bodies(input_str)
                # Load the generated curve (two columns: q and I)
                data = np.loadtxt(tmp_file)
                q_vals, intensity_vals = data[:, 0], data[:, 1]
                # Preprocess and store
                processed = preprocess_curve(q_vals, intensity_vals, q_min, q_max, n_points_resampled)
                X[sample_idx] = processed
                y[sample_idx] = spec.label
                metadata.append({"index": sample_idx, "shape": spec.name, **params})
                sample_idx += 1
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    # Write metadata
    fieldnames = set().union(*(row.keys() for row in metadata))
    fieldnames = list(fieldnames)
    with open(os.path.join(output_dir, "metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata:
            full_row = {fn: row.get(fn, "") for fn in fieldnames}
            writer.writerow(full_row)
    return X, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAXS dataset using ATSAS bodies.")
    parser.add_argument("--output-dir", type=str, default="saxs_dataset_atsas", help="Output directory.")
    parser.add_argument("--samples-per-shape", type=int, default=100, help="Number of samples per shape.")
    parser.add_argument("--points", type=int, default=256, help="Number of points in the resampled curve.")
    parser.add_argument("--raw-points", type=int, default=200, help="Number of points generated by bodies.")
    parser.add_argument("--qmin", type=float, default=0.005, help="Minimum q value (Å⁻¹).")
    parser.add_argument("--qmax", type=float, default=0.3, help="Maximum q value (Å⁻¹).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Generating {len(SHAPE_SPECS)} shapes × {args.samples_per_shape} samples each ="
        f" {len(SHAPE_SPECS) * args.samples_per_shape} curves."
    )
    X, y = generate_dataset(
        specs=SHAPE_SPECS,
        samples_per_shape=args.samples_per_shape,
        q_min=args.qmin,
        q_max=args.qmax,
        n_points_raw=args.raw_points,
        n_points_resampled=args.points,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )
    print(
        f"Finished. Dataset saved in {args.output_dir}. Shapes:"
        f" {[spec.name for spec in SHAPE_SPECS]}."
    )


if __name__ == "__main__":
    main()