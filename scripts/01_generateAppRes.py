
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
    level: float = 0.5
    abs: float = 1e-6

@dataclass
class DomainCfg:
    width: float = 1.0
    height: float = 3.0
    x_min: float = 0.0
    y_min: float = 0.0

@dataclass
class TriangularCfg:
    first_row: int = 29
    step: int = 3

@dataclass
class ParallelCfg:
    max_workers: Optional[int] = None  # None/auto -> set at runtime

@dataclass
class ElectrodesYCfg:
    start: float = 0.9
    end: float = 2.1
    n: int = 32

@dataclass
class AppConfig:
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
    return np.linspace(cfg.electrodes.start, cfg.electrodes.end, cfg.electrodes.n)


def _create_refined_mesh(cfg: AppConfig):
    x_min, y_min = cfg.domain.x_min, cfg.domain.y_min
    x_max = x_min + cfg.domain.width
    y_max = y_min + cfg.domain.height
    world = mt.createRectangle(start=[x_min, y_min], end=[x_max, y_max], marker=1)
    # light refinement near electrodes (left edge)
    for y in _electrodes_y(cfg):
        world.createNode([x_min, float(y)])
        world.createNode([x_min + 0.05, float(y)])
    return mt.createMesh(world, quality=34)


def _triangular_from_rhoa(cfg: AppConfig, rhoa: np.ndarray) -> np.ndarray:
    first = cfg.triangular.first_row
    step = cfg.triangular.step
    row_sizes = np.arange(first, 0, -step)
    if rhoa.size != int(row_sizes.sum()):
        raise ValueError(
            f"Unexpected data length {rhoa.size} for WA triangular ({int(row_sizes.sum())} expected)"
        )
    tri = np.full((len(row_sizes), int(row_sizes[0])), np.nan, dtype=float)
    s = 0
    for i, k in enumerate(row_sizes):
        k = int(k)
        tri[i, :k] = rhoa[s:s+k]
        s += k
    return tri


def _calc_apparent_resistivity_for_t(cfg: AppConfig, res_map_2d: np.ndarray):
    mesh = _create_refined_mesh(cfg)

    # map cell centers -> resistivity
    res_model = np.zeros(mesh.cellCount(), dtype=float)
    ny, nx = res_map_2d.shape

    x_min, y_min = cfg.domain.x_min, cfg.domain.y_min
    x_max = x_min + cfg.domain.width
    y_max = y_min + cfg.domain.height

    for c in mesh.cells():
        cx, cy = float(c.center()[0]), float(c.center()[1])
        # normalize to [0,1] within domain bounds
        fx = np.clip((cx - x_min) / (x_max - x_min), 0.0, 1.0)
        fy = np.clip((cy - y_min) / (y_max - y_min), 0.0, 1.0)
        ix = min(nx - 1, max(0, int(fx * nx)))
        iy = min(ny - 1, max(0, int(fy * ny)))
        res_model[c.id()] = res_map_2d[iy, ix]

    elecs = [[x_min, float(y)] for y in _electrodes_y(cfg)]
    data = ert.createData(elecs=elecs, schemeName=cfg.scheme)
    data = ert.simulate(
        mesh,
        scheme=data,
        res=res_model,
        noiseLevel=cfg.noise.level,
        noiseAbs=cfg.noise.abs,
        seed=cfg.seed,
    )

    rhoa = data['rhoa']
    rhoa[rhoa < 0] = np.nan
    data['rhoa'] = rhoa
    return data

# =====================
# Worker (top-level)
# =====================

def process_one_sequence(cfg_dict: dict, seq_idx: int) -> int:
    """Compute all timesteps for a sequence and save triangular-matrix stack.
    Returns the seq_idx on success.

    cfg_dict is used instead of AppConfig to keep the object picklable across processes.
    """
    cfg = AppConfig(**cfg_dict)

    # Lazy mmap inside the child
    z = np.load(cfg.data_path, mmap_mode='r')
    if z.ndim != 4:
        raise ValueError(f"Expected 4D array (seq, time, ny, nx); got {z.shape}")

    n_seq, n_time, ny, nx = z.shape
    tris = []
    for t in range(n_time):
        # read slice lazily; flip X; compute resistivity
        sigma = z[seq_idx, t, :, ::-1]
        res_map = 1.0 / np.asarray(sigma, dtype=float)

        app = _calc_apparent_resistivity_for_t(cfg, res_map)
        tri = _triangular_from_rhoa(cfg, np.asarray(app['rhoa']))
        tris.append(tri)

    tris = np.asarray(tris, dtype=float)
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"triangular_matrix_seq_{seq_idx}.npy")
    np.save(out_path, tris)
    return seq_idx

# =====================
# Runner
# =====================

def _auto_workers(par_cfg: ParallelCfg) -> int:
    if par_cfg.max_workers in (None, "auto"):
        return max(1, min(os.cpu_count() or 1, 4))
    try:
        v = int(par_cfg.max_workers)
        return max(1, v)
    except Exception:
        return max(1, min(os.cpu_count() or 1, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '-c', type=str, required=True, help='Path to YAML config')
    ap.add_argument('--workers', type=int, default=None, help='Override max workers')
    ap.add_argument('--sequences', type=int, nargs='*', default=None, help='Optional subset of sequence indices')
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(Path(args.config))
    if args.workers is not None:
        cfg.parallel.max_workers = int(args.workers)
    if args.sequences is not None and len(args.sequences) > 0:
        cfg.sequences = list(map(int, args.sequences))

    # Probe file and compute target sequence list
    z = np.load(cfg.data_path, mmap_mode='r')
    if z.ndim != 4:
        raise ValueError(f"Expected 4D array (seq, time, ny, nx); got {z.shape}")
    n_seq, n_time, ny, nx = z.shape

    if cfg.sequences is None:
        seq_list = list(range(n_seq))
    else:
        # validate and clamp
        seq_list = [i for i in cfg.sequences if 0 <= int(i) < n_seq]
        if not seq_list:
            raise ValueError("No valid sequence indices after validation.")

    os.makedirs(cfg.output_dir, exist_ok=True)

    max_workers = args.workers if args.workers else _auto_workers(cfg.parallel)

    t0 = time.perf_counter()
    print(f"[start] {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  workers={max_workers}  sequences={len(seq_list)}/{n_seq}")

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

    # Parallel dispatch
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(process_one_sequence, cfg_dict, i): i for i in seq_list}
        for fut in as_completed(fut_map):
            seq = fut_map[fut]
            ret = fut.result()  # propagate exceptions
            done += 1
            print(f"[done] sequence {ret}  ({done}/{len(seq_list)})")

    dt = time.perf_counter() - t0
    print(f"[end]   {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[time]  elapsed = {timedelta(seconds=int(dt))} ({dt:.3f} s)")


if __name__ == '__main__':
    main()