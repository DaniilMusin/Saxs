"""
generate_saxs_dataset.py
=======================

This script generates a synthetic dataset of small‑angle X‑ray scattering (SAXS)
curves for several simple particle shapes (sphere, cylinder and ellipsoid),
applies basic preprocessing (logarithm of the intensity, integral
normalisation and standardisation) and stores the resulting curves and labels
to disk.  The goal is to provide a reproducible starting point for
machine‑learning experiments similar to those described in Monge et al.

Each generated curve uses the same q‑range (0.005–0.3 Å⁻¹) and the same
number of sampling points, which simplifies feeding them into neural
networks.  Parameters for each shape are sampled uniformly within user‑
defined ranges.  The script writes three files:

* ``X.npy`` – a NumPy array of shape ``(n_samples, n_points)`` with the
  processed scattering curves.
* ``y.npy`` – an integer vector of length ``n_samples`` with class labels
  (0 = sphere, 1 = cylinder, 2 = ellipsoid).
* ``metadata.csv`` – a CSV file listing the shape, the sampled geometric
  parameters and the file index.  This may be useful for later
  interpretation or regression tasks.

Because ATSAS is not available in this environment, the script uses
analytical expressions for form factors and carries out orientation
averaging numerically when necessary.  For high‑throughput generation or
more complex shapes (e.g. core–shell particles), replace or extend the
functions accordingly or call ``bodies``/``mixtures`` via subprocess if
ATSAS is installed.
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np


@dataclass
class ShapeSpec:
    """Specification of a particle shape for synthetic SAXS generation."""
    name: str
    label: int
    generator: Callable[[np.ndarray, dict], np.ndarray]
    param_ranges: dict


def sphere_intensity(q: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the scattering intensity for a solid sphere.

    The form factor for a homogeneous sphere of radius R is

    F(q) = 3 * [sin(qR) - qR cos(qR)] / (qR)^3

    The scattering intensity is |F(q)|^2.

    Parameters
    ----------
    q : ndarray
        One‑dimensional array of scattering vector magnitudes.
    params : dict
        Must contain key ``radius``.

    Returns
    -------
    ndarray
        Intensity values at the provided q points.
    """
    R = params["radius"]
    qR = q * R
    # avoid division by zero at q=0
    qR = np.where(qR == 0, 1e-10, qR)
    F = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR**3)
    I = F**2
    return I


def cylinder_intensity(q: np.ndarray, params: dict) -> np.ndarray:
    """
    Approximate scattering intensity for a finite cylinder averaged over all
    orientations.

    This implementation numerically integrates the single‑particle form factor
    over random orientation angles.  The cylinder is characterised by its
    radius ``r`` and length ``h``.

    The amplitude for a given orientation (inclination angle theta between
    cylinder axis and q) is

        F(q, theta) = 2 * J1(q r sin theta) / (q r sin theta)
                     * sin(0.5 * q h cos theta) / (0.5 * q h cos theta)

    where J1 is the first‑order Bessel function of the first kind.  The
    intensity is |F(q, theta)|^2.  We approximate the orientation average by
    sampling ``n_angles`` values of theta from 0 to pi/2 (due to symmetry).
    """
    from scipy.special import j1  # import here to avoid dependency if not used

    r = params["radius"]
    h = params["height"]
    # number of sampled orientations for averaging; increase for better accuracy
    n_angles = 50
    # sample theta uniformly in [0, pi/2]
    theta = np.linspace(0, 0.5 * np.pi, n_angles)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    I = np.zeros_like(q)
    for st, ct in zip(sin_theta, cos_theta):
        qrst = q * r * st
        # avoid singularities
        qrst = np.where(qrst == 0, 1e-10, qrst)
        qhct = q * h * ct / 2.0
        qhct = np.where(qhct == 0, 1e-10, qhct)
        F = (2 * j1(qrst) / (qrst)) * (np.sin(qhct) / (qhct))
        I += np.abs(F)**2
    # average over orientations
    return I / n_angles


def ellipsoid_intensity(q: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute scattering intensity for a triaxial ellipsoid by numerically
    averaging over orientations.  The semi‑axes are a, b, c.  We follow the
    approach similar to the sphere (averaging effective radius over
    orientations).

    For each orientation angle phi and psi we compute the radius

        R(φ, ψ) = ( (a * sinφ * cosψ)**2 + (b * sinφ * sinψ)**2
                    + (c * cosφ)**2 )^{-1/2}

    and then use the sphere form factor with radius R.  The orientational
    average is approximated by sampling ``n_phi × n_psi`` angles.
    """
    a, b, c = params["a"], params["b"], params["c"]
    n_phi = 20
    n_psi = 20
    phi = np.linspace(0, 0.5 * np.pi, n_phi)  # polar angle
    psi = np.linspace(0, 0.5 * np.pi, n_psi)  # azimuthal angle (0..pi/2 due to symmetry)
    I = np.zeros_like(q)
    for ph in phi:
        for ps in psi:
            # effective radius for this orientation
            inv_R2 = (a * np.sin(ph) * np.cos(ps))**2 + (b * np.sin(ph) * np.sin(ps))**2 + (c * np.cos(ph))**2
            R_eff = 1.0 / np.sqrt(inv_R2)
            # treat as sphere
            I += sphere_intensity(q, {"radius": R_eff})
    return I / (n_phi * n_psi)


def log_normalise_resample(intensity: np.ndarray, q: np.ndarray, q_min: float, q_max: float, n_points: int) -> np.ndarray:
    """
    Apply log transformation, integral normalisation, standardisation and
    resampling to a scattering intensity curve.

    Parameters
    ----------
    intensity : ndarray
        Raw intensity values I(q).
    q : ndarray
        Original q values corresponding to ``intensity``.
    q_min, q_max : float
        Desired range of q for the resampled curve.
    n_points : int
        Number of points in the resampled curve.

    Returns
    -------
    ndarray
        The processed intensity curve sampled at log‑spaced q values.
    """
    # ensure positivity
    intensity = np.maximum(intensity, 1e-12)
    # log10 intensity
    logI = np.log10(intensity)
    # integral normalisation – divide by area under the original curve
    area = np.trapz(intensity, q)
    if area > 0:
        logI /= area
    # standardise: zero mean, unit variance
    logI = (logI - np.mean(logI)) / np.std(logI)
    # resample to log‑spaced q grid
    q_new = np.logspace(np.log10(q_min), np.log10(q_max), n_points)
    # interpolate logI to new q grid
    logI_interp = np.interp(q_new, q, logI)
    return logI_interp


def sample_params(param_ranges: dict) -> dict:
    """
    Sample a dictionary of parameters uniformly within the provided ranges.
    ``param_ranges`` maps parameter names to (min, max).
    """
    return {name: np.random.uniform(low, high) for name, (low, high) in param_ranges.items()}


def generate_dataset(
    shapes: Iterable[ShapeSpec],
    n_per_shape: int,
    q_min: float = 0.005,
    q_max: float = 0.3,
    n_points_original: int = 200,
    n_points_resampled: int = 128,
    random_seed: int = 42,
    output_dir: str = "output_dataset",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset of scattering curves and apply preprocessing.

    Parameters
    ----------
    shapes : iterable of ShapeSpec
        The particle types to generate.
    n_per_shape : int
        Number of samples to generate per shape.
    q_min, q_max : float
        Range of q values for generation and resampling.
    n_points_original : int
        Number of q points in the raw intensity curve.
    n_points_resampled : int
        Number of points in the processed curve.
    random_seed : int
        Seed for reproducibility.
    output_dir : str
        Directory where the metadata CSV will be written.

    Returns
    -------
    X : ndarray
        Array of processed curves of shape (n_samples, n_points_resampled).
    y : ndarray
        Array of integer labels of length n_samples.
    """
    np.random.seed(random_seed)
    q = np.linspace(q_min, q_max, n_points_original)
    samples = []
    labels = []
    metadata_rows = []
    idx = 0
    for shape in shapes:
        for _ in range(n_per_shape):
            params = sample_params(shape.param_ranges)
            raw_intensity = shape.generator(q, params)
            processed = log_normalise_resample(raw_intensity, q, q_min, q_max, n_points_resampled)
            samples.append(processed)
            labels.append(shape.label)
            # record metadata
            metadata_rows.append({
                "index": idx,
                "shape": shape.name,
                **{k: v for k, v in params.items()},
            })
            idx += 1
    X = np.array(samples)
    y = np.array(labels, dtype=np.int64)
    # write metadata
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    # Determine all possible fieldnames across rows to avoid missing keys
    fieldnames = set()
    for row in metadata_rows:
        fieldnames.update(row.keys())
    fieldnames = list(fieldnames)
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_rows:
            # fill missing keys with empty strings
            complete_row = {fn: row.get(fn, "") for fn in fieldnames}
            writer.writerow(complete_row)
    # save arrays
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    return X, y


def main():
    # Define the shapes to generate.  Extend or modify the parameter ranges
    # according to your needs.  The ranges were chosen to produce nontrivial
    # intensities within the selected q‑range.
    shapes = [
        ShapeSpec(
            name="sphere",
            label=0,
            generator=sphere_intensity,
            param_ranges={"radius": (10.0, 50.0)},
        ),
        ShapeSpec(
            name="cylinder",
            label=1,
            generator=cylinder_intensity,
            param_ranges={"radius": (5.0, 30.0), "height": (20.0, 100.0)},
        ),
        ShapeSpec(
            name="ellipsoid",
            label=2,
            generator=ellipsoid_intensity,
            param_ranges={"a": (10.0, 40.0), "b": (10.0, 40.0), "c": (10.0, 40.0)},
        ),
    ]
    # Generate dataset
    n_per_shape = 50  # adjust the number of samples per shape as needed
    X, y = generate_dataset(
        shapes=shapes,
        n_per_shape=n_per_shape,
        q_min=0.005,
        q_max=0.3,
        n_points_original=200,
        n_points_resampled=128,
        random_seed=123,
        output_dir="saxs_dataset",
    )
    print(f"Generated dataset with {X.shape[0]} samples and {X.shape[1]} features per sample.")
    print("Classes distribution:", {shapes[i].name: int(np.sum(y == i)) for i in range(len(shapes))})


if __name__ == "__main__":
    main()