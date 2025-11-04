import os
import argparse
import time
import datetime as _dt
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

# =============================================================
# Simplified parallel version (no Ctrl+C/graceful stop handling)
# =============================================================
# - Parallelize by sequence using ProcessPoolExecutor
# - No parent->child signalling; just run to completion
# - Lazy mmap inside each child to avoid copying huge arrays
# - Top-level, picklable worker
# =============================================================

# ---- Config ----
DATA_PATH = '../data/combined_conductivity_maps.npy'
OUTPUT_DIR = 'visualizations_large'
SCHEME = 'wa'  # Wenner-alpha
DOMAIN_W, DOMAIN_H = 1.0, 3.0
X_MIN, X_MAX = 0.0, DOMAIN_W
Y_MIN, Y_MAX = 0.0, DOMAIN_H
ELECTRODES_Y = np.linspace(0.9, 2.1, 32)

# Determined at runtime from file
ARRAY_SHAPE = None  # (n_seq, n_time, ny, nx)


# ---- Helpers (pure functions) ----
def _create_refined_mesh():
    world = mt.createRectangle(start=[X_MIN, Y_MIN], end=[X_MAX, Y_MAX], marker=1)
    # light refinement near electrodes (left edge)
    for y in ELECTRODES_Y:
        world.createNode([X_MIN, y])
        world.createNode([X_MIN + 0.05, y])
    return mt.createMesh(world, quality=34)


def _calc_apparent_resistivity_for_t(res_map_2d):
    mesh = _create_refined_mesh()

    # map cell centers -> resistivity
    res_model = np.zeros(mesh.cellCount(), dtype=float)
    ny, nx = res_map_2d.shape
    for c in mesh.cells():
        cx, cy = float(c.center()[0]), float(c.center()[1])
        ix = min(nx - 1, max(0, int(cx / DOMAIN_W * nx)))
        iy = min(ny - 1, max(0, int(cy / DOMAIN_H * ny)))
        res_model[c.id()] = res_map_2d[iy, ix]

    elecs = [[X_MIN, float(y)] for y in ELECTRODES_Y]
    data = ert.createData(elecs=elecs, schemeName=SCHEME)
    data = ert.simulate(mesh, scheme=data, res=res_model, noiseLevel=0.5, noiseAbs=1e-6, seed=1337)

    rhoa = data['rhoa']
    rhoa[rhoa < 0] = np.nan
    data['rhoa'] = rhoa
    return data


def _to_triangular(apparent):
    # Wenner-alpha expected row sizes (29, 26, ..., 2)
    row_sizes = np.arange(29, 0, -3)
    arr = np.asarray(apparent['rhoa'])
    if arr.size != int(row_sizes.sum()):
        raise ValueError(
            f"Unexpected data length {arr.size} for WA triangular ({row_sizes.sum()} expected)"
        )
    tri = np.full((len(row_sizes), int(row_sizes[0])), np.nan, dtype=float)
    s = 0
    for i, k in enumerate(row_sizes):
        tri[i, :int(k)] = arr[s:s+int(k)]
        s += int(k)
    return tri


# ---- Worker (top-level & picklable) ----
def process_one_sequence(seq_idx: int) -> int:
    """Compute all timesteps for a sequence and save triangular-matrix stack.
    Returns the seq_idx on success.
    """
    # Lazy mmap inside the child
    z = np.load(DATA_PATH, mmap_mode='r')
    n_seq, n_time, ny, nx = z.shape

    tris = []
    for t in range(n_time):
        # read slice lazily; flip X; compute resistivity
        sigma = z[seq_idx, t, :, ::-1]
        res_map = 1.0 / np.asarray(sigma, dtype=float)

        app = _calc_apparent_resistivity_for_t(res_map)
        tri = _to_triangular(app)
        tris.append(tri)

    tris = np.asarray(tris, dtype=float)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"triangular_matrix_seq_{seq_idx}.npy")
    np.save(out_path, tris)
    return seq_idx


# ---- Runner ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=None, help='Max parallel processes (default=min(CPU,4))')
    args = p.parse_args()

    max_workers = args.workers or max(1, min(os.cpu_count() or 1, 4))

    # Probe file to get shape and validate
    z = np.load(DATA_PATH, mmap_mode='r')
    if z.ndim != 4:
        raise ValueError(f"Expected 4D array (seq, time, ny, nx); got {z.shape}")
    n_seq, n_time, ny, nx = z.shape

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[start] {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  workers={max_workers}  sequences={n_seq}")

    # Parallel dispatch
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(process_one_sequence, i): i for i in range(n_seq)}
        for fut in as_completed(fut_map):
            seq = fut_map[fut]
            # Propagate exceptions if any
            ret = fut.result()
            done += 1
            print(f"[done] sequence {ret}  ({done}/{n_seq})")

    dt = time.perf_counter() - t0
    print(f"[end]   {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[time]  elapsed = {timedelta(seconds=int(dt))} ({dt:.3f} s)")


if __name__ == '__main__':
    main()
