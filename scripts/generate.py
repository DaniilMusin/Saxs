#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator: bodies -> crysol -> imsim -> im2dat
Writes ATSAS-style *.dat (q, I, sigma) and meta.csv.
"""
from __future__ import annotations
import argparse, json, os, random, sys, tempfile, time, math, shutil
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import pandas as pd

from atsas_cli import which_or_die, run, run_crysol, run_im2dat, build_imsim_args, convert_int_to_dat
from instrument_cfg import Instrument
from param_sampler import draw_shape, available_shapes

def ensure_dirs(root: Path):
    (root / "saxs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(parents=True, exist_ok=True)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--uid-offset", type=int, default=0)
    p.add_argument("--out", type=str, default="dataset")
    p.add_argument("--cfgs", type=str, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--append-meta", action="store_true")
    p.add_argument("--mask", type=str, default=None, help="override beamstop mask path, wins over cfg")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--use-bodies", action="store_true", help="Use ATSAS bodies (DAM) via stdin for PDB generation")
    p.add_argument("--per-class", type=int, default=0, help="If >0, generate exactly this many curves per shape class")
    p.add_argument("--direct-crysol", action="store_true", help="Use CRYSOL .int files directly, bypassing IMSIM/IM2DAT (recommended for Windows)")
    p.add_argument("--skip-tool-check", action="store_true", help="Skip checking for ATSAS tools (demo mode)")
    return p.parse_args()

def _estimate_spacing_from_volume(volume_A3: float, target_beads: int = 40000) -> float:
    spacing = (volume_A3 / max(target_beads, 1)) ** (1.0 / 3.0)
    return max(2.0, min(25.0, spacing))

def _frange_centered(min_val: float, max_val: float, step: float) -> Iterable[float]:
    if step <= 0:
        yield 0.0
        return
    n = int(math.floor((max_val - min_val) / step))
    for i in range(n + 1):
        yield min_val + i * step

def _write_pdb(points: Iterable[Tuple[float, float, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        serial = 1
        for (x, y, z) in points:
            # PDB ATOM record: columns match typical CRYSOL expectations
            f.write(
                f"ATOM  {serial:5d}  C   DAM A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            serial += 1
        f.write("END\n")

def _generate_pdb_sphere(radius_A: float, out_path: Path) -> None:
    volume = (4.0 / 3.0) * math.pi * radius_A ** 3
    step = _estimate_spacing_from_volume(volume)
    r2 = radius_A * radius_A
    pts = []
    for x in _frange_centered(-radius_A, radius_A, step):
        x2 = x * x
        for y in _frange_centered(-radius_A, radius_A, step):
            y2 = y * y
            xy2 = x2 + y2
            if xy2 > r2:
                continue
            for z in _frange_centered(-radius_A, radius_A, step):
                if xy2 + z * z <= r2:
                    pts.append((x, y, z))
    _write_pdb(pts, out_path)

def _generate_pdb_cylinder(radius_A: float, length_A: float, out_path: Path) -> None:
    volume = math.pi * radius_A ** 2 * length_A
    step = _estimate_spacing_from_volume(volume)
    r2 = radius_A * radius_A
    half_L = 0.5 * length_A
    pts = []
    for x in _frange_centered(-radius_A, radius_A, step):
        x2 = x * x
        for y in _frange_centered(-radius_A, radius_A, step):
            if x2 + y * y > r2:
                continue
            for z in _frange_centered(-half_L, half_L, step):
                pts.append((x, y, z))
    _write_pdb(pts, out_path)

def _generate_pdb_ellipsoid(a_A: float, b_A: float, c_A: float, out_path: Path) -> None:
    volume = (4.0 / 3.0) * math.pi * a_A * b_A * c_A
    step = _estimate_spacing_from_volume(volume)
    pts = []
    for x in _frange_centered(-a_A, a_A, step):
        xn = (x / a_A) ** 2
        if xn > 1.0:
            continue
        for y in _frange_centered(-b_A, b_A, step):
            yn = (y / b_A) ** 2
            if xn + yn > 1.0:
                continue
            for z in _frange_centered(-c_A, c_A, step):
                zn = (z / c_A) ** 2
                if xn + yn + zn <= 1.0:
                    pts.append((x, y, z))
    _write_pdb(pts, out_path)

def _generate_pdb_parallelepiped(a_A: float, b_A: float, c_A: float, out_path: Path) -> None:
    volume = a_A * b_A * c_A
    step = _estimate_spacing_from_volume(volume)
    half_a, half_b, half_c = 0.5 * a_A, 0.5 * b_A, 0.5 * c_A
    pts = []
    for x in _frange_centered(-half_a, half_a, step):
        for y in _frange_centered(-half_b, half_b, step):
            for z in _frange_centered(-half_c, half_c, step):
                pts.append((x, y, z))
    _write_pdb(pts, out_path)

def generate_pdb_model(shape_class: str, params: Dict[str, float], out_path: Path) -> None:
    """Pure-Python DAM-like PDB generator (fallback when 'bodies' is unavailable).
    Shapes: sphere, cylinder, oblate/prolate (as ellipsoid), parallelepiped, etc."""
    if shape_class == "sphere":
        _generate_pdb_sphere(params["radius"], out_path)
        return
    if shape_class == "cylinder":
        _generate_pdb_cylinder(params["radius"], params["length"], out_path)
        return
    if shape_class in ("oblate", "prolate", "ellipsoid_triaxial", "ellipsoid_of_rotation"):
        _generate_pdb_ellipsoid(params["a"], params.get("b", params["a"]), params["c"], out_path)
        return
    if shape_class == "parallelepiped":
        _generate_pdb_parallelepiped(params["a"], params["b"], params["c"], out_path)
        return
    if shape_class in ("hollow_sphere", "hollow_cylinder", "elliptical_cylinder"):
        # For complex shapes, fallback to simple approximations
        if shape_class == "hollow_sphere":
            _generate_pdb_sphere(params["outer_radius"], out_path)
        elif shape_class == "hollow_cylinder":
            _generate_pdb_cylinder(params["outer_radius"], params["length"], out_path)
        elif shape_class == "elliptical_cylinder":
            _generate_pdb_cylinder(max(params["a"], params.get("b", params["a"])), params["length"], out_path)
        return
    if shape_class in ("dumbbell", "liposome", "membrane_protein"):
        # For very complex shapes, use simple sphere approximations
        if shape_class == "dumbbell":
            r_approx = (params["r1"] + params["r2"]) / 2
            _generate_pdb_sphere(r_approx, out_path)
        elif shape_class == "liposome":
            _generate_pdb_sphere(params["outer_radius"], out_path)
        elif shape_class == "membrane_protein":
            _generate_pdb_sphere(params["rmemb"], out_path)
        return
    if shape_class.startswith("core_shell_"):
        # Core-shell approximations using outer dimensions
        if shape_class == "core_shell_sphere":
            total_radius = params["core_radius"] + params["shell_thickness"]
            _generate_pdb_sphere(total_radius, out_path)
        elif shape_class in ("core_shell_oblate", "core_shell_prolate"):
            total_radius = params["equatorial_core_radius"] + params["shell_thickness"]
            axial_ratio = params["axial_ratio"]
            polar_radius = axial_ratio * total_radius
            _generate_pdb_ellipsoid(total_radius, total_radius, polar_radius, out_path)
        elif shape_class == "core_shell_cylinder":
            total_radius = params["core_radius"] + params["shell_thickness"]
            _generate_pdb_cylinder(total_radius, params["length"], out_path)
        return
    raise NotImplementedError(f"PDB generator not implemented for shape {shape_class}")

def main():
    args = parse_args()
    out_root = Path(args.out).resolve()
    ensure_dirs(out_root)
    # Check tools early
    if not args.skip_tool_check:
        for exe in ["crysol", "imsim", "im2dat"]:
            which_or_die(exe)
        if args.use_bodies:
            which_or_die("bodies")

    cfgs = [Instrument.from_yaml(Path(p)) for p in args.cfgs]
    meta_rows: List[Dict[str, Any]] = []
    meta_path = out_root / "meta.csv"
    need_header = not (args.append_meta and meta_path.exists())

    def build_bodies_dam_stdin(shape_class: str, params: Dict[str, float], pdb_out: Path) -> str:
        lines: List[str] = []
        # Enter DAM mode
        lines.append("d")
        
        # Map shape classes to ATSAS 4.x body type numbers
        if shape_class == "sphere":
            # Use hollow-sphere (7) with inner_radius=0 for solid sphere
            lines.append("7")  # hollow-sphere
            if "radius" in params:
                lines.append(f"{params['radius']:.6f}")  # outer_radius (ro)
                lines.append("0.0")  # inner_radius (ri) - 0 = solid
            else:
                lines.append(f"{params.get('outer_radius', 50.0):.6f}")
                lines.append(f"{params.get('inner_radius', 0.0):.6f}")
            lines.append("1.0")  # scale parameter
        elif shape_class == "hollow_sphere":
            lines.append("7")  # hollow-sphere
            lines.append(f"{params['outer_radius']:.6f}")  # ro
            lines.append(f"{params['inner_radius']:.6f}")  # ri
            lines.append("1.0")  # scale
        elif shape_class == "cylinder":
            lines.append("3")  # cylinder
            lines.append(f"{params['radius']:.6f}")  # r
            lines.append(f"{params['length']:.6f}")  # h
            lines.append("1.0")  # scale
        elif shape_class == "hollow_cylinder":
            lines.append("5")  # hollow-cylinder
            lines.append(f"{params['outer_radius']:.6f}")  # ro
            lines.append(f"{params['inner_radius']:.6f}")  # ri
            lines.append(f"{params['length']:.6f}")  # h
            lines.append("1.0")  # scale
        elif shape_class == "elliptical_cylinder":
            lines.append("4")  # elliptic-cylinder
            lines.append(f"{params['a']:.6f}")  # a
            lines.append(f"{params['b']:.6f}")  # b
            lines.append(f"{params['length']:.6f}")  # h
            lines.append("1.0")  # scale
        elif shape_class in ("oblate", "prolate"):
            lines.append("1")  # ellipsoid
            lines.append(f"{params['a']:.6f}")  # a
            lines.append(f"{params['b']:.6f}")  # b
            lines.append(f"{params['c']:.6f}")  # c
            lines.append("1.0")  # scale
        elif shape_class == "parallelepiped":
            lines.append("6")  # parallelepiped
            lines.append(f"{params['a']:.6f}")  # a
            lines.append(f"{params['b']:.6f}")  # b
            lines.append(f"{params['c']:.6f}")  # c
            lines.append("1.0")  # scale
            # ATSAS bodies asks for parameters again for parallelepiped
            lines.append(f"{params['a']:.6f}")  # a (repeated)
            lines.append(f"{params['b']:.6f}")  # b (repeated) 
            lines.append(f"{params['c']:.6f}")  # c (repeated)
        elif shape_class == "ellipsoid_triaxial":
            lines.append("1")  # ellipsoid (triaxial)
            lines.append(f"{params['a']:.6f}")  # a
            lines.append(f"{params['b']:.6f}")  # b
            lines.append(f"{params['c']:.6f}")  # c
            lines.append("1.0")  # scale
        elif shape_class == "ellipsoid_of_rotation":
            lines.append("2")  # ellipsoid of rotation
            a = params.get("a", params.get("radius", 50.0))
            ratio = params.get("ratio", params.get("c", a) / a if a else 1.0)
            lines.append(f"{a:.6f}")  # equatorial radius
            lines.append(f"{ratio:.6f}")  # aspect ratio (c/a)
            lines.append("1.0")  # scale
        elif shape_class == "dumbbell":
            lines.append("8")  # dumbbell
            lines.append(f"{params['r1']:.6f}")  # radius 1
            lines.append(f"{params['r2']:.6f}")  # radius 2
            lines.append(f"{params['center_distance']:.6f}")  # center distance
            lines.append("1.0")  # scale
        elif shape_class == "liposome":
            lines.append("9")  # liposome
            lines.append(f"{params['outer_radius']:.6f}")  # outer radius
            lines.append(f"{params['bilayer_thickness']:.6f}")  # bilayer thickness
            lines.append(f"{params['inner_radius']:.6f}")  # inner radius
            lines.append("1.0")  # scale
        elif shape_class == "membrane_protein":
            lines.append("10")  # membrane protein
            lines.append(f"{params['rmemb']:.6f}")  # membrane radius
            lines.append(f"{params['rtail']:.6f}")  # tail radius
            lines.append(f"{params['rhead']:.6f}")  # head radius
            lines.append(f"{params['delta']:.6f}")  # delta
            lines.append(f"{params['zcorona']:.6f}")  # z corona
            lines.append("1.0")  # scale
        # Core-shell shapes - these will be approximated using hollow variants or mixtures at curve level
        elif shape_class == "core_shell_sphere":
            # Approximate as hollow sphere
            total_radius = params["core_radius"] + params["shell_thickness"]
            lines.append("7")  # hollow-sphere
            lines.append(f"{total_radius:.6f}")  # outer_radius
            lines.append(f"{params['core_radius']:.6f}")  # inner_radius (core)
            lines.append("1.0")  # scale
        elif shape_class in ("core_shell_oblate", "core_shell_prolate"):
            # Approximate as solid ellipsoid with total dimensions
            total_radius = params["equatorial_core_radius"] + params["shell_thickness"]
            axial_ratio = params["axial_ratio"]
            polar_radius = axial_ratio * total_radius
            lines.append("1")  # ellipsoid
            lines.append(f"{total_radius:.6f}")  # a (equatorial)
            lines.append(f"{total_radius:.6f}")  # b (equatorial)
            lines.append(f"{polar_radius:.6f}")  # c (polar)
            lines.append("1.0")  # scale
        elif shape_class == "core_shell_cylinder":
            # Approximate as hollow cylinder
            total_radius = params["core_radius"] + params["shell_thickness"]
            lines.append("5")  # hollow-cylinder
            lines.append(f"{total_radius:.6f}")  # outer_radius
            lines.append(f"{params['core_radius']:.6f}")  # inner_radius
            lines.append(f"{params['length']:.6f}")  # length
            lines.append("1.0")  # scale
        else:
            raise NotImplementedError(f"BODIES shape {shape_class} not supported")
        
        # Symmetry selection (default P1)
        lines.append("1")  # P1 symmetry
        
        # Number of dummy atoms (default is fine)
        lines.append("2000")  # number of dummy atoms
        
        # Output file path
        lines.append(str(pdb_out))
        lines.append("")  # Empty line to finish
        return "\n".join(lines)

    def generate_one_uid(uid: int, sample_shape: str | None = None) -> None:
        rng = random.Random(args.seed + uid)
        sample = draw_shape(rng, shape=sample_shape)
        cfg = rng.choice(cfgs)
        log_file = out_root / "logs" / f"{uid:06d}.log"
        try:
            with tempfile.TemporaryDirectory(dir=out_root / "tmp") as td:
                td = Path(td)
                pdb_path = td / "model.pdb"
                pdb_generated = False
                generator_used = "pythonDAM"
                
                if args.use_bodies and shutil.which("bodies"):
                    try:
                        from atsas_cli import run_bodies_dam
                        stdin_text = build_bodies_dam_stdin(sample.shape_class, sample.params, pdb_path)
                        run_bodies_dam(stdin_text, log_file, timeout=args.timeout)
                        pdb_generated = True
                        generator_used = "bodies"
                    except Exception as e:
                        # Bodies failed, fallback to Python DAM
                        with open(log_file, "a", encoding="utf-8") as lf:
                            lf.write(f"\n[FALLBACK] Bodies failed: {e}\n")
                            lf.write(f"[FALLBACK] Using Python DAM generator\n")
                
                if not pdb_generated:
                    generate_pdb_model(sample.shape_class, sample.params, pdb_path)
                    generator_used = "pythonDAM"

                abs_path = run_crysol(pdb_path, out_base=td / "noiseless", qmax=0.45, bins=890,
                                      absolute=True, lsq="1e-5", log_file=log_file, timeout=args.timeout)

                out_dat = out_root / "saxs" / f"{uid:06d}__{cfg.name}.dat"
                
                if args.direct_crysol:
                    # Use .int file directly, bypass IMSIM/IM2DAT
                    int_path = td / "noiseless.int"
                    if int_path.exists():
                        convert_int_to_dat(int_path, out_dat, f"CRYSOL {sample.shape_class}")
                    else:
                        raise RuntimeError(f"CRYSOL .int file not found: {int_path}")
                else:
                    # Traditional IMSIM/IM2DAT pipeline
                    frame_path = td / "frame.tiff"
                    imsim_args = build_imsim_args(
                        abs_file=abs_path,
                        detector=cfg.detector,
                        detector_distance_m=cfg.detector_distance_m,
                        wavelength_A=cfg.wavelength_A,
                        flux_ph_s=cfg.flux_ph_s,
                        exposure_s=cfg.exposure_s,
                        out_frame=frame_path,
                        seed=rng.randrange(0, 2**31-1),
                        extra=None,
                        output_format="TIFF"
                    )
                    run(imsim_args, log_file, timeout=args.timeout)

                    beam_mask = Path(args.mask) if args.mask else cfg.beamstop_mask
                    if beam_mask and not Path(beam_mask).exists():
                        beam_mask = None
                    axis_data = cfg.axis_out if (getattr(cfg, "axis_out", None) and Path(cfg.axis_out).exists()) else None
                    run_im2dat(frame_path, out_dat, beam_mask, axis_data, log_file, timeout=args.timeout)

            meta_rows.append({
                "uid": uid,
                "shape_class": sample.shape_class,
                "generator": (("bodies+crysol" if args.use_bodies else "pythonDAM+crysol") + 
                            ("+direct" if args.direct_crysol else "+imsim+im2dat")),
                "true_params": json.dumps({**sample.params, "poly_sigma": sample.poly_sigma}, ensure_ascii=False),
                "instrument_cfg": cfg.name,
                "seed": args.seed + uid
            })
        except Exception as e:
            with open(out_root / "failed.txt", "a", encoding="utf-8") as f:
                f.write(f"{uid}\t{e}\n")
            print(f"[WARN] uid={uid} failed: {e}", file=sys.stderr)

    if args.per_class and args.per_class > 0:
        classes = available_shapes()
        uid = args.uid_offset
        for cls in classes:
            for _ in range(args.per_class):
                generate_one_uid(uid, sample_shape=cls)
                uid += 1
    else:
        for i in range(args.n):
            uid = args.uid_offset + i
            generate_one_uid(uid)

    if meta_rows:
        df = pd.DataFrame(meta_rows, columns=["uid","shape_class","generator","true_params","instrument_cfg","seed"])
        mode = "a" if (args.append_meta and meta_path.exists()) else "w"
        header = not (args.append_meta and meta_path.exists()) and need_header
        df.to_csv(meta_path, index=False, mode=mode, header=header)

if __name__ == "__main__":
    main()
