"""
01_generateAppRes.py
--------------------

Purpose
    Convert time‐evolving 2D *conductivity* fields (σ in S/m) into per–time-step
    **apparent resistivity** (ρ_a in Ohm·m) triangular matrices for a Wenner-alpha
    (WA) acquisition scheme using pyGIMLi. The script is designed to batch-process
    many sequences with optional multiprocessing.

What this script does
    1) Loads a 4D NumPy array from disk with shape (N_seq, N_time, Ny, Nx) that
       stores conductivity σ [S/m].
    2) For each sequence and each time step:
         a. Flips X ( [:, ::-1] ) so that the model’s *left* boundary aligns with
            the electrode line at x = x_min (electrodes on the left edge).
         b. Converts σ → ρ (cellwise) via ρ = 1 / σ for the forward model.
         c. Builds a 2D rectangular mesh and lightly refines near the electrode line.
         d. Generates a WA acquisition scheme with N=32 electrodes by default
            (all parameters configurable via YAML).
         e. Runs a forward simulation with a simple noise model
            (relative level [%] + absolute floor [Ohm·m]).
         f. Reads `data['rhoa']` and reshapes the flat vector into a **row-decreasing
            triangular matrix** compatible with the WA layout (non-triangle entries
            are filled with NaN).
    3) Stacks all triangular matrices across time and saves one `.npy` per sequence.

Input
    - A single `.npy` file specified by `config.data_path` containing σ [S/m]
      with shape (N_seq, N_time, Ny, Nx). The array is memory-mapped for low RAM use.

Output
    - One NumPy file per processed sequence:
        {output_dir}/triangular_matrix_seq_{seq}.npy
      Each file has shape (N_time, N_rows, N_cols) where (N_rows, N_cols) is the
      triangular canvas; entries outside the triangle are NaN.

Assumptions & conventions
    - Units: input is **conductivity** σ in S/m; the forward model uses **resistivity**
      ρ = 1/σ in Ohm·m; the exported apparent resistivity `rhoa` is also in Ohm·m.
    - Geometry: rectangular domain [x_min, x_min+height] × [y_min, y_min+width].
      Electrodes lie on the **left boundary** (x = x_min) with evenly spaced
      y-coordinates between `electrodes.y.start` and `electrodes.y.end`.
    - Default electrode count is 32 (≈4 cm spacing if the y-range is set accordingly);
      adjust in YAML if your physical spacing differs.
    - Scheme: Wenner-alpha by default (`scheme: "wa"`). The triangular reshaper assumes
      WA ordering; if you change schemes, update `_triangular_from_rhoa(...)`.
    - Layout: the triangular matrix uses `triangular.first_row` measurements on the
      first row, then decreases by `triangular.step` per subsequent row. The total
      length must match the number of WA measurements; a mismatch raises an error.

Performance notes
    - Multiprocessing: sequences are distributed across processes; set
      `parallel.max_workers` or use the `--workers` CLI flag (default: a small,
      safe auto-choice).
    - Memory: the input array is loaded with `mmap_mode='r'`, so only the current
      slice is read into memory per time step.

CLI usage
    python 01_generateAppRes.py -c path/to/config.yml
    # Optional overrides:
    python 01_generateAppRes.py -c config.yml --workers 4 --sequences 0 3 7

Minimal YAML keys (examples)
    data_path: "../data/combined_conductivity_maps.npy"
    output_dir: "visualizations_large"
    scheme: "wa"
    seed: 1337
    noise:
      level: 0.5       # [%] relative
      abs: 1.0e-6      # [Ohm·m] floor
    domain:
      height: 1.0
      width: 3.0
      x_min: 0.0
      y_min: 0.0
    electrodes:
      y:
        start: 0.9     # [m]
        end: 2.1       # [m]
        n: 32
    triangular:
      first_row: 29    # measurements in row 0
      step: 3          # per-row decrement
    parallel:
      max_workers: null   # or an integer
    sequences: null       # or a list, e.g., [0, 2, 7]

Requirements
    - Python 3.x, NumPy, PyYAML, pyGIMLi (and its dependencies).
    - The WA triangular reshaping assumes a pyGIMLi `createData(..., schemeName="wa")`
      ordering. Keep the WA setting or adapt the reshaper accordingly.
"""


import os
import argparse
import time
import datetime as _dt
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

# =====================
# Config dataclasses
# =====================
@dataclass
class NoiseCfg:
    """Noise configuration for pyGIMLi ERT forward simulation."""
    level: float = 0.5  # relative noise level [%]
    abs: float = 1e-6   # absolute noise floor [Ohm m]


@dataclass
class DomainCfg:
    """Rectangular domain (x rightwards, y upwards). Units are meters."""
    height: float = 1.0
    width: float = 3.0
    x_min: float = 0.0
    y_min: float = 0.0


@dataclass
class TriangularCfg:
    """Triangular matrix layout for WA scheme export."""
    first_row: int = 29  # number of values in the first row
    step: int = 3        # decrement per subsequent row


@dataclass
class ParallelCfg:
    """Parallel processing config. None/'auto' -> pick a sensible default."""
    max_workers: Optional[int] = None  # None/auto -> set at runtime


@dataclass
class ElectrodesYCfg:
    """Electrode positions along the left boundary (x = x_min)."""
    start: float = 0.9   # y start (top group)
    end: float = 2.1     # y end (bottom group)
    n: int = 32          # number of electrodes


@dataclass
class AppConfig:
    """
    Application-wide configuration loaded from YAML.
    - data_path   : path to 4D conductivity array (seq, time, ny, nx) in S/m
    - output_dir  : where triangular matrices per sequence are saved
    - scheme      : acquisition scheme name for pyGIMLi (e.g., 'wa' for Wenner-alpha)
    - noise, seed : simulation parameters for reproducibility
    - domain      : rectangular model domain
    - electrodes  : y-positions of electrodes at x = x_min
    - triangular  : layout to reshape flat rhoa into a triangle (WA compatible)
    - sequences   : optional subset of sequence indices to process
    """
    data_path: str
    output_dir: str = "visualizations_large"
    scheme: str = "wa"
    noise: NoiseCfg = NoiseCfg()
    seed: int = 1337
    domain: DomainCfg = DomainCfg()
    electrodes: ElectrodesYCfg = ElectrodesYCfg()
    triangular: TriangularCfg = TriangularCfg()
    parallel: ParallelCfg = ParallelCfg()
    sequences: Optional[List[int]] = None  # optional subset

    @staticmethod
    def from_yaml(path: Path) -> "AppConfig":
        """Load config from YAML, preserving defaults for missing keys."""
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # Manually construct nested dataclasses to keep defaults when keys are missing
        noise = NoiseCfg(**(raw.get("noise", {}) or {}))
        domain = DomainCfg(**(raw.get("domain", {}) or {}))
        tri = TriangularCfg(**(raw.get("triangular", {}) or {}))
        par = ParallelCfg(**(raw.get("parallel", {}) or {}))
        ele_y = raw.get("electrodes", {}) or {}
        ycfg = ElectrodesYCfg(**(ele_y.get("y", {}) or {}))

        return AppConfig(
            data_path=raw.get("data_path", "../data/combined_conductivity_maps.npy"),
            output_dir=raw.get("output_dir", "visualizations_large"),
            scheme=raw.get("scheme", "wa"),
            noise=noise,
            seed=int(raw.get("seed", 1337)),
            domain=domain,
            electrodes=ycfg,
            triangular=tri,
            parallel=par,
            sequences=raw.get("sequences", None),
        )

# =====================
# Helpers (pure-ish)
# =====================

def _electrodes_y(cfg: AppConfig) -> np.ndarray:
    """Return evenly spaced electrode y-positions between start and end."""
    return np.linspace(cfg.electrodes.start, cfg.electrodes.end, cfg.electrodes.n)


def _create_refined_mesh(cfg: AppConfig):
    """
    Create a 2D rectangular mesh and add slight refinement near the electrode line
    (left boundary) to improve forward simulation accuracy.
    """
    x_min, y_min = cfg.domain.x_min, cfg.domain.y_min
    x_max = x_min + cfg.domain.height
    y_max = y_min + cfg.domain.width
    world = mt.createRectangle(start=[x_min, y_min], end=[x_max, y_max], marker=1)

    # Light refinement near electrodes (insert nodes close to the left side).
    for y in _electrodes_y(cfg):
        world.createNode([x_min, float(y)])
        world.createNode([x_min + 0.05, float(y)])

    return mt.createMesh(world, quality=34)


def _triangular_from_rhoa(cfg: AppConfig, rhoa: np.ndarray) -> np.ndarray:
    """
    Reshape a flat rhoa vector into a row-decreasing triangular matrix for WA.
    NaNs are filled outside the triangular footprint.
    """
    first = cfg.triangular.first_row
    step = cfg.triangular.step
    row_sizes = np.arange(first, 0, -step)

    # Sanity check to catch mismatch between scheme export and expected layout.
    if rhoa.size != int(row_sizes.sum()):
        raise ValueError(
            f"Unexpected data length {rhoa.size} for WA triangular "
            f"({int(row_sizes.sum())} expected)"
        )

    tri = np.full((len(row_sizes), int(row_sizes[0])), np.nan, dtype=float)
    s = 0
    for i, k in enumerate(row_sizes):
        k = int(k)
        tri[i, :k] = rhoa[s:s + k]
        s += k
    return tri


def _calc_apparent_resistivity_for_t(cfg: AppConfig, res_map_2d: np.ndarray):
    """
    Run a forward ERT simulation for one timestep:
    - Build mesh
    - Map cell centers to local resistivity values from res_map_2d
    - Create acquisition scheme and simulate with noise
    Returns pyGIMLi ERT 'data' object containing 'rhoa'.
    """
    mesh = _create_refined_mesh(cfg)

    # Map cell centers -> resistivity (nearest-neighbor in normalized coords).
    res_model = np.zeros(mesh.cellCount(), dtype=float)
    ny, nx = res_map_2d.shape

    x_min, y_min = cfg.domain.x_min, cfg.domain.y_min
    x_max = x_min + cfg.domain.height
    y_max = y_min + cfg.domain.width

    for c in mesh.cells():
        cx, cy = float(c.center()[0]), float(c.center()[1])
        # Normalize to [0,1] within domain bounds.
        fx = np.clip((cx - x_min) / (x_max - x_min), 0.0, 1.0)
        fy = np.clip((cy - y_min) / (y_max - y_min), 0.0, 1.0)
        ix = min(nx - 1, max(0, int(fx * nx)))
        iy = min(ny - 1, max(0, int(fy * ny)))
        res_model[c.id()] = res_map_2d[iy, ix]

    # Electrodes placed on the left edge (x = x_min) at specified y positions.
    elecs = [[x_min, float(y)] for y in _electrodes_y(cfg)]

    # Create scheme and simulate apparent resistivity with noise.
    data = ert.createData(elecs=elecs, schemeName=cfg.scheme)
    data = ert.simulate(
        mesh,
        scheme=data,
        res=res_model,
        noiseLevel=cfg.noise.level,
        noiseAbs=cfg.noise.abs,
        seed=cfg.seed,
    )

    # Replace negative values (non-physical) with NaN before reshaping.
    rhoa = data['rhoa']
    rhoa[rhoa < 0] = np.nan
    data['rhoa'] = rhoa
    return data

# =====================
# Worker (top-level)
# =====================

def process_one_sequence(cfg_dict: dict, seq_idx: int) -> int:
    """
    Compute all timesteps for one sequence and save the triangular-matrix stack.
    Returns the sequence index on success.

    Notes:
      - `cfg_dict` is used (instead of AppConfig) to ease pickling in multiprocessing.
      - Input conductivity is assumed to be in S/m; we convert to resistivity via 1/sigma.
      - X-axis is flipped ([:, ::-1]) to match left-edge electrode positioning.
    """
    cfg = AppConfig(**cfg_dict)

    # Lazy mmap inside the child process (memory efficient for large arrays).
    z = np.load(cfg.data_path, mmap_mode='r')
    if z.ndim != 4:
        raise ValueError(f"Expected 4D array (seq, time, ny, nx); got {z.shape}")

    n_seq, n_time, ny, nx = z.shape
    tris = []
    for t in range(n_time):
        # Read slice lazily; flip X; compute resistivity map for the forward model.
        sigma = z[seq_idx, t, :, ::-1]
        res_map = 1.0 / np.asarray(sigma, dtype=float)

        # Forward simulation -> flat rhoa -> triangular matrix.
        app = _calc_apparent_resistivity_for_t(cfg, res_map)
        tri = _triangular_from_rhoa(cfg, np.asarray(app['rhoa']))
        tris.append(tri)

    # Stack all timesteps and save for this sequence.
    tris = np.asarray(tris, dtype=float)
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"triangular_matrix_seq_{seq_idx}.npy")
    np.save(out_path, tris)
    return seq_idx

# =====================
# Runner
# =====================

def _auto_workers(par_cfg: ParallelCfg) -> int:
    """
    Select a reasonable default number of workers if not specified:
    min(os.cpu_count(), 4), with lower bound 1.
    """
    if par_cfg.max_workers in (None, "auto"):
        return max(1, min(os.cpu_count() or 1, 4))
    try:
        v = int(par_cfg.max_workers)
        return max(1, v)
    except Exception:
        return max(1, min(os.cpu_count() or 1, 4))


def main():
    """Parse args, load config, and run per-sequence processing (parallelized)."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '-c', type=str, required=True, help='Path to YAML config')
    ap.add_argument('--workers', type=int, default=None, help='Override max workers')
    ap.add_argument('--sequences', type=int, nargs='*', default=None,
                    help='Optional subset of sequence indices')
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(Path(args.config))
    if args.workers is not None:
        cfg.parallel.max_workers = int(args.workers)
    if args.sequences is not None and len(args.sequences) > 0:
        cfg.sequences = list(map(int, args.sequences))

    # Probe file to derive valid sequence indices and report shapes early.
    z = np.load(cfg.data_path, mmap_mode='r')
    if z.ndim != 4:
        raise ValueError(f"Expected 4D array (seq, time, ny, nx); got {z.shape}")
    n_seq, n_time, ny, nx = z.shape

    # Build sequence list (validate any user-provided subset).
    if cfg.sequences is None:
        seq_list = list(range(n_seq))
    else:
        seq_list = [i for i in cfg.sequences if 0 <= int(i) < n_seq]
        if not seq_list:
            raise ValueError("No valid sequence indices after validation.")

    os.makedirs(cfg.output_dir, exist_ok=True)

    max_workers = args.workers if args.workers else _auto_workers(cfg.parallel)

    t0 = time.perf_counter()
    print(f"[start] {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
          f"workers={max_workers}  sequences={len(seq_list)}/{n_seq}")

    # Dataclass -> plain dict for pickling across processes
    cfg_dict = dict(
        data_path=cfg.data_path,
        output_dir=cfg.output_dir,
        scheme=cfg.scheme,
        noise=cfg.noise,
        seed=cfg.seed,
        domain=cfg.domain,
        electrodes=cfg.electrodes,
        triangular=cfg.triangular,
        parallel=cfg.parallel,
        sequences=cfg.sequences,
    )

    # Parallel dispatch over sequences; exceptions are propagated.
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(process_one_sequence, cfg_dict, i): i for i in seq_list}
        for fut in as_completed(fut_map):
            seq = fut_map[fut]
            ret = fut.result()  # propagate exceptions from workers
            done += 1
            print(f"[done] sequence {ret}  ({done}/{len(seq_list)})")

    dt = time.perf_counter() - t0
    print(f"[end]   {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[time]  elapsed = {timedelta(seconds=int(dt))} ({dt:.3f} s)")


if __name__ == '__main__':
    main()
