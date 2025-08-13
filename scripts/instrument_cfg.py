#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read a simple YAML config and convert it to IMSIM args.
YAML example fields:
  name: Xeuss1800HR
  detector: Eiger1M
  detector_distance_m: 1.8
  wavelength_A: 1.54
  flux_ph_s: 3.43e6
  exposure_s: 1200
  axis_out: axis.dat       # optional; if present, generator will use it
  beamstop_mask: masks/eiger1m.msk  # optional
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

@dataclass
class Instrument:
    name: str
    detector: str
    detector_distance_m: float
    wavelength_A: float
    flux_ph_s: float
    exposure_s: float
    axis_out: Optional[Path]
    beamstop_mask: Optional[Path]

    @staticmethod
    def from_yaml(path: Path) -> "Instrument":
        y = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return Instrument(
            name=str(y.get("name") or Path(path).stem),
            detector=str(y["detector"]),
            detector_distance_m=float(y["detector_distance_m"]),
            wavelength_A=float(y.get("wavelength_A", 1.54)),
            flux_ph_s=float(y.get("flux_ph_s", 1e6)),
            exposure_s=float(y.get("exposure_s", 60)),
            axis_out=Path(y["axis_out"]) if y.get("axis_out") else None,
            beamstop_mask=Path(y["beamstop_mask"]) if y.get("beamstop_mask") else None,
        )
