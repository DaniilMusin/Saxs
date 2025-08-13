#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sampler for shape classes.
Implements 5 base classes; core-shell classes are sketched for extension.

Deterministic: all randomness uses the provided rng instance.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import math


def log_uniform(rng, lo: float, hi: float) -> float:
    """Sample on a log10 scale deterministically using the given rng."""
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def uniform(rng, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


@dataclass
class Sample:
    shape_class: str
    params: Dict[str, float]
    poly_sigma: float


def poly_sigma_from_mean(rng, *values: float) -> float:
    mean_value = sum(values) / len(values)
    fraction = uniform(rng, 0.0, 0.3)
    return mean_value * fraction


def available_shapes() -> List[str]:
    """Return the full set of shapes supported by the sampler and bodies-stdin builder."""
    return [
        "sphere",
        "hollow_sphere",
        "cylinder",
        "hollow_cylinder",
        "elliptical_cylinder",
        "oblate",
        "prolate",
        "parallelepiped",
        "ellipsoid_triaxial",
        "ellipsoid_of_rotation",
        "dumbbell",
        "liposome",
        "membrane_protein",
        # Core-shell variants
        "core_shell_sphere",
        "core_shell_oblate",
        "core_shell_prolate",
        "core_shell_cylinder",
    ]


def draw_shape(rng, shape: str | None = None) -> Sample:
    """Draw a shape and parameters deterministically. If shape is provided, use it."""
    shapes = available_shapes()
    shape = shape or rng.choice(shapes)

    if shape == "sphere":
        R = log_uniform(rng, 25, 500)
        return Sample("sphere", {"radius": R}, poly_sigma_from_mean(rng, R))

    if shape == "hollow_sphere":
        R = log_uniform(rng, 25, 500)
        rin = uniform(rng, 0.0, max(1.0, R - 1.0))
        return Sample("hollow_sphere", {"outer_radius": R, "inner_radius": rin}, poly_sigma_from_mean(rng, R))

    if shape == "cylinder":
        R = log_uniform(rng, 25, 500)
        L = uniform(rng, 10.0 * R, 20.0 * R)
        return Sample("cylinder", {"radius": R, "length": L}, poly_sigma_from_mean(rng, R, L))

    if shape == "hollow_cylinder":
        R = log_uniform(rng, 25, 500)
        rin = uniform(rng, 1.0, max(1.0, R - 1.0))
        L = uniform(rng, 10.0 * R, 20.0 * R)
        return Sample("hollow_cylinder", {"outer_radius": R, "inner_radius": rin, "length": L}, poly_sigma_from_mean(rng, R, L))

    if shape == "elliptical_cylinder":
        a = log_uniform(rng, 25, 500)
        c = log_uniform(rng, 25, 500)
        # Ensure max(a,c) <= 500
        max_axis = max(a, c)
        if max_axis > 500:
            scale = 500.0 / max_axis
            a *= scale
            c *= scale
        L = uniform(rng, 10.0 * a, 20.0 * a)
        return Sample("elliptical_cylinder", {"a": a, "c": c, "length": L}, poly_sigma_from_mean(rng, a, c, L))

    if shape == "oblate":
        a = log_uniform(rng, 25, 500)
        c_over_a = uniform(rng, 0.1, 0.7)
        c = c_over_a * a
        return Sample("oblate", {"a": a, "b": a, "c": c}, poly_sigma_from_mean(rng, a, c))

    if shape == "prolate":
        a = log_uniform(rng, 25, 500)
        c_over_a = uniform(rng, 1.3, 5.0)
        c = c_over_a * a
        # Ensure total size doesn't exceed 500
        if c > 500:
            c = 500.0
            a = c / c_over_a
        return Sample("prolate", {"a": a, "b": a, "c": c}, poly_sigma_from_mean(rng, a, c))

    if shape == "parallelepiped":
        a = log_uniform(rng, 25, 500)
        b = log_uniform(rng, 25, 500)
        c = log_uniform(rng, 25, 500)
        # Ensure max(a,b,c) <= 500
        max_axis = max(a, b, c)
        if max_axis > 500:
            scale = 500.0 / max_axis
            a *= scale
            b *= scale
            c *= scale
        return Sample("parallelepiped", {"a": a, "b": b, "c": c}, poly_sigma_from_mean(rng, a, b, c))

    if shape == "ellipsoid_triaxial":
        a = log_uniform(rng, 25, 500)
        b = log_uniform(rng, 25, 500)
        c = log_uniform(rng, 25, 500)
        # Ensure max(a,b,c) <= 500
        max_axis = max(a, b, c)
        if max_axis > 500:
            scale = 500.0 / max_axis
            a *= scale
            b *= scale
            c *= scale
        return Sample("ellipsoid_triaxial", {"a": a, "b": b, "c": c}, poly_sigma_from_mean(rng, a, b, c))

    if shape == "ellipsoid_of_rotation":
        a = log_uniform(rng, 25, 500)
        ratio = uniform(rng, 0.1, 5.0)
        c = ratio * a
        # Ensure constraint: equatorial_radius * max(1, axial_ratio) <= 500
        if a * max(1.0, ratio) > 500:
            a = 500.0 / max(1.0, ratio)
            c = ratio * a
        return Sample("ellipsoid_of_rotation", {"a": a, "b": a, "c": c, "ratio": ratio}, poly_sigma_from_mean(rng, a, c))

    if shape == "dumbbell":
        r1 = log_uniform(rng, 25, 500)
        r2 = log_uniform(rng, 25, 500)
        max_r = max(r1, r2)
        cd_min = abs(r1 - r2)
        cd_max = 2 * max_r + 500
        center_distance = uniform(rng, max(1.0, cd_min), cd_max)
        return Sample("dumbbell", {"r1": r1, "r2": r2, "center_distance": center_distance}, poly_sigma_from_mean(rng, r1, r2, center_distance))

    if shape == "liposome":
        outer_radius = log_uniform(rng, 25, 500)
        max_thickness = min(100.0, max(1.0, outer_radius - 1.0))
        thickness = uniform(rng, 1.0, max_thickness)
        inner_radius = max(1.0, outer_radius - thickness)
        return Sample("liposome", {"outer_radius": outer_radius, "bilayer_thickness": thickness, "inner_radius": inner_radius}, poly_sigma_from_mean(rng, outer_radius, inner_radius))

    if shape == "membrane_protein":
        rmemb = uniform(rng, 10.0, 250.0)
        rtail = uniform(rng, 1.0, 100.0)
        rhead = uniform(rng, 1.0, 100.0)
        delta = uniform(rng, 1.0, 100.0)
        zcorona = uniform(rng, 0.0, 200.0)
        # Scale down if any parameter is too large
        max_param = max(rmemb, rtail, rhead, delta)
        if max_param > 500:
            scale = 500.0 / max_param
            rmemb *= scale
            rtail *= scale
            rhead *= scale
            delta *= scale
        return Sample("membrane_protein", {"rmemb": rmemb, "rtail": rtail, "rhead": rhead, "delta": delta, "zcorona": zcorona}, poly_sigma_from_mean(rng, rmemb, rtail, rhead))

    if shape == "core_shell_sphere":
        core_radius = log_uniform(rng, 1, 499)
        shell_thickness = uniform(rng, 1, min(500 - core_radius, 499))
        return Sample("core_shell_sphere", {"core_radius": core_radius, "shell_thickness": shell_thickness}, poly_sigma_from_mean(rng, core_radius, shell_thickness))

    if shape == "core_shell_oblate":
        core_radius = log_uniform(rng, 1, 499)
        shell_thickness = uniform(rng, 1, min(500 - core_radius, 499))
        axial_ratio = uniform(rng, 0.1, 0.7)  # R_polar/R_equat
        return Sample("core_shell_oblate", {"equatorial_core_radius": core_radius, "shell_thickness": shell_thickness, "axial_ratio": axial_ratio}, poly_sigma_from_mean(rng, core_radius, shell_thickness))

    if shape == "core_shell_prolate":
        core_radius = log_uniform(rng, 1, 499)
        shell_thickness = uniform(rng, 1, min(500 - core_radius, 499))
        axial_ratio = uniform(rng, 1.3, 5.0)  # R_polar/R_equat
        return Sample("core_shell_prolate", {"equatorial_core_radius": core_radius, "shell_thickness": shell_thickness, "axial_ratio": axial_ratio}, poly_sigma_from_mean(rng, core_radius, shell_thickness))

    if shape == "core_shell_cylinder":
        core_radius = log_uniform(rng, 1, 499)
        shell_thickness = uniform(rng, 1, min(500 - core_radius, 499))
        length_ratio = uniform(rng, 10, 20)  # L/R
        total_radius = core_radius + shell_thickness
        length = length_ratio * total_radius
        return Sample("core_shell_cylinder", {"core_radius": core_radius, "shell_thickness": shell_thickness, "length": length}, poly_sigma_from_mean(rng, core_radius, shell_thickness, length))

    raise NotImplementedError(shape)
