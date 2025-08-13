#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC script: computes Guinier Rg and I(0) via AUTORG for the first N curves.
Writes qc_report.csv
"""
from __future__ import annotations
import argparse, json, subprocess, shutil, csv
from pathlib import Path

def which(exe: str) -> str:
    p = shutil.which(exe)
    if p is None:
        raise RuntimeError(f"{exe} not found in PATH")
    return p

def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset/saxs")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    which("autorg")

    saxs_dir = Path(args.dataset)
    files = sorted(saxs_dir.glob("*.dat"))[:args.n]

    rows = []
    for f in files:
        # AUTORG
        try:
            out = run_cmd(["autorg", str(f)])
            # very rough parse: look for "Rg" and "I(0)" in output
            Rg = None; I0 = None
            for line in out.splitlines():
                if "Rg" in line and "=" in line:
                    try:
                        Rg = float(line.split("=")[1].split()[0])
                    except: pass
                if "I(0)" in line and "=" in line:
                    try:
                        I0 = float(line.split("=")[1].split()[0])
                    except: pass
        except Exception as e:
            Rg = None; I0 = None

        rows.append({"file": f.name, "Rg": Rg, "I0": I0})

    out_csv = saxs_dir.parent / "logs" / "qc_report.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["file","Rg","I0"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    main()
