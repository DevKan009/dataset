"""
lattice_strength.py
Periodic Timoshenko-beam solver for the Zheng et al. truss dataset.

For one lattice (27-node octant graph) it returns
  * the 9 stiffness components (to check against stiffness-vec.csv), and
  * first-order STRENGTH labels from the same linear solve:
      y_x, y_y, y_z : first-yield strength / sigma_y   (uniaxial macro stress along axis)
      b_x, b_y, b_z : strut-buckling strength / E_s     (uniaxial compression along axis)
    For a real material:  strength = min(sigma_y * y, E_s * b).

Conventions (verified against the dataset, see check_stiffness()):
  unit cell [-1,1]^3 (side 2), octant [0,1]^3 mirrored about x=0,y=0,z=0,
  base material E_s = 1, nu = 0.3, circular struts, one radius for all struts,
  radius set so that relative density = 0.15.
"""
import numpy as np

E_S, NU, RHO, CELL = 1.0, 0.3, 0.15, 2.0
V = CELL ** 3
G_S = E_S / (2 * (1 + NU))
KAPPA = 6 * (1 + NU) / (7 + 6 * NU)          # Timoshenko shear factor, circular section
SIGNS = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
K_EFF = 1.0                                   # pinned-pinned Euler buckling (conservative)

# Voigt order [11,22,33,23,13,12], engineering shear strains
_EPS = np.zeros((6, 3, 3))
for k in range(3):
    _EPS[k, k, k] = 1.0
for k, (i, j) in enumerate([(1, 2), (0, 2), (0, 1)], start=3):
    _EPS[k, i, j] = _EPS[k, j, i] = 0.5


def build_periodic_frame(adj, pos, flip=False):
    """Return (n_nodes, ia, ib, d) for the struts of one periodic unit cell."""
    p = 1.0 - pos if flip else pos
    ii, jj = np.where(np.triu(adj, 1) > 0)
    a, b = p[ii], p[jj]
    A = (a[None] * SIGNS[:, None]).reshape(-1, 3)
    B = (b[None] * SIGNS[:, None]).reshape(-1, 3)
    M0 = len(A)
    q = np.round(np.vstack([A, B]) * 1e6).astype(np.int64)
    q = np.where(q >= 1_000_000, q - 2_000_000, q)            # +1 face == -1 face
    key = ((q[:, 0] + 1_000_000) * 2_000_003 + (q[:, 1] + 1_000_000)) * 2_000_003 + (q[:, 2] + 1_000_000)
    uniq, inv = np.unique(key, return_inverse=True)
    ia, ib = inv[:M0], inv[M0:]
    dq = np.round((B - A) * 1e6).astype(np.int64)
    swap = (ia > ib) | ((ia == ib) & ((dq[:, 0] < 0) | ((dq[:, 0] == 0) & (dq[:, 1] < 0))
                                        | ((dq[:, 0] == 0) & (dq[:, 1] == 0) & (dq[:, 2] < 0))))
    ia, ib = np.where(swap, ib, ia), np.where(swap, ia, ib)
    dq = np.where(swap[:, None], -dq, dq)
    rows = np.unique(np.column_stack([ia, ib, dq]), axis=0)     # same strut seen from neighbouring cells
    return len(uniq), rows[:, 0], rows[:, 1], rows[:, 2:] / 1e6


def _local_k(L, r):
    M = len(L)
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    phi = 12 * E_S * I / (G_S * KAPPA * A * L ** 2)
    kb = E_S * I / (L ** 3 * (1 + phi))
    K = np.zeros((M, 12, 12))

    def put(i, j, v):
        K[:, i, j] = v
        K[:, j, i] = v
    put(0, 0, E_S * A / L); put(6, 6, E_S * A / L); put(0, 6, -E_S * A / L)
    put(3, 3, G_S * J / L); put(9, 9, G_S * J / L); put(3, 9, -G_S * J / L)
    for (v1, t1, v2, t2, s) in [(1, 5, 7, 11, 1.0), (2, 4, 8, 10, -1.0)]:   # xy-plane, xz-plane
        put(v1, v1, 12 * kb); put(v2, v2, 12 * kb); put(v1, v2, -12 * kb)
        put(v1, t1, s * 6 * L * kb); put(v1, t2, s * 6 * L * kb)
        put(v2, t1, -s * 6 * L * kb); put(v2, t2, -s * 6 * L * kb)
        put(t1, t1, (4 + phi) * L ** 2 * kb); put(t2, t2, (4 + phi) * L ** 2 * kb)
        put(t1, t2, (2 - phi) * L ** 2 * kb)
    return K, A, I, J


def _rotations(d, L):
    ex = d / L[:, None]
    h = np.zeros_like(ex)
    h[np.arange(len(ex)), np.argmin(np.abs(ex), axis=1)] = 1.0
    ey = np.cross(h, ex)
    ey /= np.linalg.norm(ey, axis=1, keepdims=True)
    ez = np.cross(ex, ey)
    R = np.stack([ex, ey, ez], axis=1)
    T = np.zeros((len(L), 12, 12))
    for b in range(4):
        T[:, 3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
    return T


def _chains(n, ia, ib, d, L):
    """Group collinear struts that meet at 2-strut nodes into straight members."""
    M = len(L)
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    inc = [[] for _ in range(n)]
    for s in range(M):
        inc[ia[s]].append((s, 1.0))
        inc[ib[s]].append((s, -1.0))
    for node in range(n):
        if len(inc[node]) == 2:
            (s1, g1), (s2, g2) = inc[node]
            o1, o2 = g1 * d[s1] / L[s1], g2 * d[s2] / L[s2]
            if np.linalg.norm(o1 + o2) < 1e-5:
                parent[find(s1)] = find(s2)
    return np.array([find(s) for s in range(M)])


def solve_lattice(adj, pos, flip=False, rho=RHO, debug=False):
    n, ia, ib, d = build_periodic_frame(adj, pos, flip)
    L = np.linalg.norm(d, axis=1)
    r = np.sqrt(rho * V / (np.pi * L.sum()))
    Kl, A, I, J = _local_k(L, r)
    T = _rotations(d, L)
    Kg = np.einsum('mji,mjk,mkl->mil', T, Kl, T)
    M, N6 = len(L), 6 * n
    idx = np.concatenate([6 * ia[:, None] + np.arange(6), 6 * ib[:, None] + np.arange(6)], axis=1)
    Kglob = np.bincount((idx[:, :, None] * N6 + idx[:, None, :]).ravel(),
                        weights=Kg.ravel(), minlength=N6 * N6).reshape(N6, N6)
    Uaff = np.zeros((6, M, 12))
    Uaff[:, :, 6:9] = np.einsum('kij,mj->kmi', _EPS, d)
    Faff = np.einsum('mef,kmf->kme', Kg, Uaff)
    F = np.stack([np.bincount(idx.ravel(), weights=Faff[k].ravel(), minlength=N6) for k in range(6)])
    free = np.arange(3, N6)                                       # pin one node's translation
    Ut = np.zeros((6, N6))
    Ut[:, free] = np.linalg.solve(Kglob[np.ix_(free, free)], -F[:, free].T).T
    Ue = Ut[:, idx] + Uaff
    Fe = np.einsum('mef,lmf->lme', Kg, Ue)
    C = np.einsum('kme,lme->kl', Ue, Fe) / V
    C = 0.5 * (C + C.T)
    out = dict(C=C, r=r, n_struts=M, n_nodes=n,
               stiff9=np.array([C[0, 0], C[0, 1], C[0, 2], C[1, 1], C[1, 2], C[2, 2], C[3, 3], C[4, 4], C[5, 5]]))

    # ---- strength labels from the same linear solve -------------------------------------
    floc = np.einsum('mij,kmj->kmi', T, Fe)                      # local end forces for each unit strain
    S = np.linalg.inv(C)
    chain = _chains(n, ia, ib, d, L)
    cids = np.unique(chain)
    y, b = np.zeros(3), np.zeros(3)
    for j in range(3):                                           # unit uniaxial macro stress along axis j
        f = np.einsum('k,kmi->mi', S[:, j], floc)
        N = f[:, 6]                                              # axial force, tension positive
        Tq = np.abs(f[:, 9])
        M1 = np.hypot(f[:, 4], f[:, 5]); M2 = np.hypot(f[:, 10], f[:, 11])
        tau = Tq * r / J
        s1 = np.sqrt((np.abs(N) / A + M1 * r / I) ** 2 + 3 * tau ** 2)
        s2 = np.sqrt((np.abs(N) / A + M2 * r / I) ** 2 + 3 * tau ** 2)
        y[j] = 1.0 / max(s1.max(), s2.max())
        # buckling: members that are compressed when the macro stress is compressive (N > 0 here)
        lam = np.inf
        for c in cids:
            sel = chain == c
            Pc = N[sel].max()
            if Pc > 1e-12:
                Pcr = np.pi ** 2 * E_S * I / (K_EFF * L[sel].sum()) ** 2
                lam = min(lam, Pcr / Pc)
        b[j] = lam
    out['y'], out['b'] = y, b
    if debug:
        out['_dbg'] = (Fe, d, idx, N6)
    return out
