#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrappers around ATSAS CLI tools with logging and timeouts.
Tools: bodies, crysol, imsim, im2dat, datcmp, autorg.

Note: This code does not assume specific ATSAS install paths; it expects the
executables to be discoverable via PATH. Some CLI flags differ across ATSAS
versions; adapt where marked.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

DEFAULT_TIMEOUT = 60 * 10  # 10 minutes per step by default


def which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if p is None:
        raise RuntimeError(f"Executable '{exe}' not found in PATH")
    return p


def run(cmd: List[str], log_file: Path, timeout: int = DEFAULT_TIMEOUT, cwd: Optional[Path] = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout, cwd=str(cwd) if cwd else None
        )
        if proc.stdout:
            lf.write(proc.stdout)
        if proc.stderr:
            lf.write("\n[stderr]\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}; see {log_file}")


def run_bodies_predict(stdin_text: str, log_file: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    """
    Drive 'bodies' in predict mode by sending answers via stdin.
    You must construct stdin_text according to your local BODIES prompts.
    """
    which_or_die("bodies")
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("\n$ bodies  # (interactive via stdin)\n")
        proc = subprocess.run(
            ["bodies"], input=stdin_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )
        if proc.stdout:
            lf.write(proc.stdout)
        if proc.stderr:
            lf.write("\n[stderr]\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"bodies failed: see {log_file}")


def run_bodies_dam(stdin_text: str, log_file: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    """
    Drive 'bodies' in DAM mode by sending answers via stdin.
    You must construct stdin_text according to your local BODIES prompts.
    """
    which_or_die("bodies")
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("\n$ bodies  # (DAM via stdin)\n")
        proc = subprocess.run(
            ["bodies"], input=stdin_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )
        if proc.stdout:
            lf.write(proc.stdout)
        if proc.stderr:
            lf.write("\n[stderr]\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"bodies DAM failed: see {log_file}")


def run_crysol(
    pdb_path: Path,
    out_base: Path,
    qmax: float = 0.45,
    bins: int = 890,
    absolute: bool = True,
    lsq: str | None = "1e-5",
    log_file: Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path:
    """
    Run CRYSOL to compute intensities for IMSIM and return the '.abs' path.
    ATSAS 4.x uses --smax/--ns and '-p <prefix>'; older versions used
    --qmax/--bins/--absolute/--lsq and '--output'. We try 4.x flags first and
    fall back to legacy flags if needed.
    """
    which_or_die("crysol")
    base = out_base.with_suffix("")  # CRYSOL writes <base>.abs etc.
    log_target = log_file or (out_base.parent / "crysol.log")

    # First try ATSAS 4.x style flags
    cmd_modern = [
        "crysol",
        str(pdb_path),
        "--smax",
        f"{qmax}",
        "--ns",
        f"{bins}",
        "-p",
        str(base),
    ]
    try:
        run(cmd_modern, log_target, timeout=timeout)
    except Exception:
        # Fall back to legacy flags (ATSAS 3.x)
        cmd_legacy = ["crysol", str(pdb_path)]
        if qmax is not None:
            cmd_legacy += ["--qmax", f"{qmax}"]
        if bins is not None:
            cmd_legacy += ["--bins", f"{bins}"]
        if absolute:
            cmd_legacy += ["--absolute"]
        if lsq is not None:
            cmd_legacy += ["--lsq", str(lsq)]
        cmd_legacy += ["--output", str(base)]
        run(cmd_legacy, log_target, timeout=timeout)

    abs_path = base.with_suffix(".abs")
    if not abs_path.exists():
        raise RuntimeError(f"CRYSOL did not create {abs_path}")
    return abs_path


def build_imsim_args(
    abs_file: Path,
    detector: str,
    detector_distance_m: float,
    wavelength_A: float,
    flux_ph_s: float,
    exposure_s: float,
    out_frame: Path,
    seed: int,
    extra: Optional[List[str]] = None,
    output_format: Optional[str] = None,
) -> List[str]:
    """
    Convert config to IMSIM CLI flags. On Windows, prefer output_format="TIFF"
    as EDF writing may not be available in some builds.
    """
    which_or_die("imsim")
    args = [
        "imsim",
        str(abs_file),
        "--detector",
        detector,
        "--detector-distance",
        f"{detector_distance_m}",
        "--wavelength",
        f"{wavelength_A*1e-10}",  # A -> m
        "--flux",
        f"{flux_ph_s}",
        "--exptime",
        f"{exposure_s}",
        "--seed",
        str(seed),
    ]
    if output_format:
        args += ["-f", output_format]
    args += ["-o", str(out_frame)]
    if extra:
        args.extend(extra)
    return args


def run_im2dat(frame_file: Path, out_dat: Path, beamstop_mask: Optional[Path], axis_data: Optional[Path],
               log_file: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    which_or_die("im2dat")
    cmd = ["im2dat", str(frame_file), "-o", str(out_dat)]
    if beamstop_mask:
        cmd += ["--beamstop-mask", str(beamstop_mask)]
    if axis_data:
        cmd += ["--axis-data", str(axis_data)]
    run(cmd, log_file, timeout=timeout)


def convert_int_to_dat(int_path: Path, out_dat: Path, sample_description: str = "ATSAS generated") -> None:
    """
    Convert CRYSOL .int file to standard .dat format (q, I, error).
    This bypasses the IMSIM/IM2DAT pipeline that can have issues on Windows.
    """
    import numpy as np
    
    with open(int_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip header, get data lines
    data_lines = [line.strip() for line in lines[1:] if line.strip() and not line.startswith(' Dif')]

    q_vals = []
    intensities = []

    for line in data_lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                q = float(parts[0])
                I = float(parts[1])
                q_vals.append(q)
                intensities.append(I)
            except ValueError:
                continue

    # Add simple errors as sqrt(I) with minimum 1% of I
    errors = [max(np.sqrt(I), I*0.01) for I in intensities]

    # Write in .dat format
    out_dat.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dat, 'w', encoding='utf-8') as f:
        f.write(f'Sample description: {sample_description}\n')
        f.write('Sample:   c= 1.000 mg/ml  Code: \n')
        for q, I, err in zip(q_vals, intensities, errors):
            f.write(f'{q:.6e}   {I:.6e}   {err:.6e}\n')
