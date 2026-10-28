#!/usr/bin/env python3
"""
generate_strength_labels.py

Adds strength labels (and recomputed stiffness) to lattices from the truss dataset,
at any relative density, using the periodic beam solver in lattice_strength.py.

Examples (run from this folder; needs numpy + scipy; pandas only for --verify):
  python generate_strength_labels.py --verify                 # check solver vs stiffness-vec.csv
  python generate_strength_labels.py --n 20000 --rho 0.05 0.10 0.15 0.20 --out labels.csv
  python generate_strength_labels.py --n 0 --rho 0.15          # ALL lattices at density 0.15

Output columns
  id, rho, C1111 C1122 C1133 C2222 C2233 C3333 C2323 C3131 C1212   (stiffness / E_s)
  y_x y_y y_z   first-yield strength / sigma_y   (uniaxial stress along that axis)
  b_x b_y b_z   strut-buckling strength / E_s    (uniaxial compression along that axis)
  n_struts, ok
Strength for a real material (E_s, sigma_y):  min(sigma_y * y, E_s * b)
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lattice_strength as ls

COLS = ["id", "rho", "C1111", "C1122", "C1133", "C2222", "C2233", "C3333", "C2323", "C3131", "C1212",
        "y_x", "y_y", "y_z", "b_x", "b_y", "b_z", "n_struts", "ok"]


def work(task):
    ids, adj, pos, rhos = task
    rows = []
    for n, i in enumerate(ids):
        for rho in rhos:
            try:
                r = ls.solve_lattice(adj[n], pos[n], rho=rho)
                rows.append([i, rho, *r["stiff9"], *r["y"], *r["b"], r["n_struts"], 1])
            except Exception:
                rows.append([i, rho] + [np.nan] * 15 + [0, 0])
    return np.array(rows, dtype=float)


def tasks(A, P, ids, rhos, chunk):
    for s in range(0, len(ids), chunk):
        sub = ids[s:s + chunk]
        rows = (sub[:, None] * 27 + np.arange(27)).ravel()
        yield (sub, A[rows].toarray().reshape(-1, 27, 27), P[rows].toarray().reshape(-1, 27, 3), rhos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "strength_labels.csv"))
    ap.add_argument("--n", type=int, default=2000, help="number of random lattices (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.15], help="relative densities")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--verify", action="store_true",
                    help="compare solver stiffness at rho=0.15 with stiffness-vec.csv on 200 random lattices")
    a = ap.parse_args()

    t0 = time.time()
    A = sp.load_npz(os.path.join(a.data_dir, "adjacency-matrix.npz")).tocsr()
    P = sp.load_npz(os.path.join(a.data_dir, "node-position.npz")).tocsr()
    N = A.shape[0] // 27
    print(f"loaded {N} lattices in {time.time() - t0:.1f}s")
    rng = np.random.default_rng(a.seed)

    if a.verify:
        import pandas as pd
        S = pd.read_csv(os.path.join(a.data_dir, "stiffness-vec.csv"), header=None).values
        ids = np.sort(rng.choice(N, 200, replace=False))
        out = np.vstack([work(t) for t in tasks(A, P, ids, [0.15], 100)])
        err = np.abs(out[:, 2:11] - S[ids]).max(1) / np.abs(S[ids]).max(1)
        print(f"stiffness vs dataset on {len(ids)} lattices: median rel. error {np.median(err):.1e}, "
              f"max {err.max():.1e}  (expect ~1e-6)")
        return

    ids = np.arange(N) if a.n == 0 else np.sort(rng.choice(N, min(a.n, N), replace=False))
    print(f"labelling {len(ids)} lattices x {len(a.rho)} densities with {a.workers} workers")
    chunks, t1 = [], time.time()
    with Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(work, tasks(A, P, ids, a.rho, a.chunk))):
            chunks.append(res)
            done = sum(len(c) for c in chunks)
            if k % 10 == 0:
                print(f"  {done}/{len(ids) * len(a.rho)} rows, {time.time() - t1:.0f}s", flush=True)
    out = np.vstack(chunks)
    out = out[np.lexsort((out[:, 1], out[:, 0]))]
    fmt = ["%d"] + ["%.4f"] + ["%.8e"] * 15 + ["%d", "%d"]
    np.savetxt(a.out, out, delimiter=",", header=",".join(COLS), comments="", fmt=fmt)
    print(f"wrote {a.out}  ({len(out)} rows, {int(out[:, -1].sum())} ok) in {time.time() - t1:.0f}s")


if __name__ == "__main__":
    main()
