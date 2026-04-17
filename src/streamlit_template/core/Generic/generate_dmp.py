# vilma_core/generate_dmp.py
from pathlib import Path
from typing import Dict, Any
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_resource(show_spinner=False)
def _cached_generate_dmp_plot(
    T_tuple,
    Y_tuple,
    nbasis,
    alpha_z,
    beta_z,
    lam,
    out_plot_str,
):
    """
    Cached DMP generation + plot.
    Heavy math + matplotlib → cached safely.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # restore arrays
    T = np.array(T_tuple)
    Y = np.array(Y_tuple)
    N = len(T)

    # goal & start
    y0 = Y[0].copy()
    g = Y[-1].copy()

    # finite differences
    def _finite_diff(y, t):
        yd = np.zeros_like(y)
        ydd = np.zeros_like(y)
        for i in range(1, len(t)):
            dt = max(1e-6, t[i] - t[i - 1])
            yd[i] = (y[i] - y[i - 1]) / dt
        for i in range(1, len(t) - 1):
            dt1 = max(1e-6, t[i] - t[i - 1])
            dt2 = max(1e-6, t[i + 1] - t[i])
            ydd[i] = 2 * (
                (y[i + 1] - y[i]) / dt2
                - (y[i] - y[i - 1]) / dt1
            ) / (dt1 + dt2)
        ydd[0] = ydd[1]
        ydd[-1] = ydd[-2]
        return yd, ydd

    # RBF features
    def _rbf_features(s, nbasis, width_scale=1.5):
        alpha = 4.0
        s_phase = np.exp(-alpha * s)
        c = np.linspace(s_phase.min(), s_phase.max(), nbasis)
        d = np.diff(c).mean() if nbasis > 1 else 1.0
        h = (width_scale / (d**2)) if d > 0 else 1.0
        phi = np.exp(-h * (s_phase[:, None] - c[None, :])**2)
        return phi / (phi.sum(axis=1, keepdims=True) + 1e-8)

    Yd, Ydd = _finite_diff(Y, T)
    tau = 1.0

    Phi = _rbf_features(T, nbasis)
    W = np.zeros((nbasis, 2))

    for d in range(2):
        f = (tau**2) * Ydd[:, d] - alpha_z * (
            beta_z * (g[d] - Y[:, d]) - tau * Yd[:, d]
        )
        A = Phi.T @ Phi + lam * np.eye(nbasis)
        b = Phi.T @ f
        W[:, d] = np.linalg.solve(A, b)

    # rollout
    Yg = np.zeros_like(Y)
    Yg[0] = y0
    y = y0.copy()
    yd = np.zeros(2)
    dt = (T[-1] - T[0]) / max(1, N - 1)

    for i in range(1, N):
        f = Phi[i] @ W
        ydd = (alpha_z * (beta_z * (g - y) - yd) + f)
        yd += ydd * dt
        y += yd * dt
        Yg[i] = y

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5))
    ax.plot(Y[:, 0], Y[:, 1], label="Original", linewidth=2)
    ax.plot(Yg[:, 0], Yg[:, 1], "--", label="DMP")
    ax.invert_yaxis()
    ax.set_title("Original vs DMP (normalized)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_plot_str, dpi=150)
    plt.close(fig)

    return out_plot_str, Yg

def _load_traj_csv(traj_csv: str):
    T, X, Y = [], [], []
    with Path(traj_csv).open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:

            # TIME (same for both formats)
            T.append(float(row["TIME"]))

            # ---- NEW FIX: detect column names automatically ----
            # Your new trajectory CSV uses:  X, Y, Z
            # Old version uses: x, y
            x_key = "X" if "X" in row else "x"
            y_key = "Y" if "Y" in row else "y"

            X.append(float(row[x_key]))
            Y.append(float(row[y_key]))

    T = np.array(T)
    X = np.array(X)
    Y = np.array(Y)

    # normalize time to [0, 1]
    if len(T) > 1:
        t0, t1 = T[0], T[-1]
        Tn = (T - t0) / (t1 - t0) if t1 > t0 else np.linspace(0, 1, len(T))
    else:
        Tn = np.linspace(0, 1, len(T))

    return Tn, np.vstack([X, Y]).T  # (N,2)


def _finite_diff(y: np.ndarray, t: np.ndarray):
    """velocity, accel via finite differences; y: (N,D)"""
    N = len(t); D = y.shape[1]
    yd = np.zeros_like(y)
    ydd = np.zeros_like(y)
    for i in range(1, N):
        dt = max(1e-6, t[i]-t[i-1])
        yd[i] = (y[i] - y[i-1]) / dt
    for i in range(1, N-1):
        dt1 = max(1e-6, t[i]-t[i-1]); dt2 = max(1e-6, t[i+1]-t[i])
        ydd[i] = 2*((y[i+1]-y[i])/dt2 - (y[i]-y[i-1])/dt1) / (dt1+dt2)
    ydd[0] = ydd[1]; ydd[-1] = ydd[-2]
    return yd, ydd

def _rbf_features(s: np.ndarray, nbasis: int = 30, width_scale: float = 1.5):
    """Radial basis over canonical phase s∈[0,1] (decay from 1→0)."""
    # canonical phase
    # we map time t∈[0,1] to s(t)=exp(-alpha*t) to emphasize early motion
    alpha = 4.0
    s_phase = np.exp(-alpha * s)  # still monotonic
    # centers uniformly over s_phase range
    c = np.linspace(s_phase.min(), s_phase.max(), nbasis)
    # widths
    d = (np.diff(c).mean() if nbasis > 1 else 1.0)
    h = (width_scale / (d**2)) if d > 0 else 1.0
    # features
    phi = np.exp(-h * (s_phase[:, None] - c[None, :])**2)
    # normalize rows
    phi_sum = np.sum(phi, axis=1, keepdims=True) + 1e-8
    return phi / phi_sum

def generate_dmp(
    traj_csv: str = "data/Generic/dmp/hand_traj.csv",
    out_npy: str = "data/Generic/dmp/dmp_reach.npy",
    out_plot: str = "data/Generic/dmp/original_vs_dmp_horizontal.png",
    nbasis: int = 30,
    alpha_z: float = 25.0,
    beta_z: float = 6.25,
    lam: float = 1e-6
) -> Dict[str, Any]:
    """
    Fit a simple 2D DMP to a normalized trajectory and regenerate it.
    """
    T, Y = _load_traj_csv(traj_csv)  # T∈[0,1], Y shape (N,2)
    N = len(T)
    if N < 5:
        raise RuntimeError("Trajectory too short for DMP.")

    # goal & start
    y0 = Y[0].copy()
    g  = Y[-1].copy()

    # derivatives
    Yd, Ydd = _finite_diff(Y, T)
    tau = 1.0  # since T normalized to [0,1]

    # Solve forcing term weights for each dimension
    # tau^2 * ydd = alpha_z*(beta_z*(g - y) - tau*yd) + f(s)
    # => f = tau^2*ydd - alpha_z*(beta_z*(g - y) - tau*yd)
    Phi = _rbf_features(T, nbasis=nbasis)  # (N,K)
    W = np.zeros((nbasis, 2))
    for d in range(2):
        f = (tau**2) * Ydd[:, d] - alpha_z * (beta_z * (g[d] - Y[:, d]) - tau * Yd[:, d])
        # ridge regression: (Phi^T Phi + lam I) w = Phi^T f
        A = Phi.T @ Phi + lam * np.eye(nbasis)
        b = Phi.T @ f
        W[:, d] = np.linalg.solve(A, b)

   
    # cached call

    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    plot_path, Yg = _cached_generate_dmp_plot(
        tuple(T.tolist()),
        tuple(Y.tolist()),
        nbasis,
        alpha_z,
        beta_z,
        lam,
        str(out_plot),
    )

    # save npy (lightweight, not cached)
    out_dir = Path(out_npy).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, Yg)


    return {
        "dmp_npy": str(Path(out_npy).resolve()),
        "plot": str(Path(out_plot).resolve()),
        "points": int(N),
        "nbasis": int(nbasis)
    }
