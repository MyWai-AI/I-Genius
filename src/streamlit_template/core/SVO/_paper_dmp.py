# Purpose: Core DMP math: learn, rollout, and save/load model parameters.
# Notes: Structured stage with stable defaults for repeatable runs.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DMPModel:
    n_dmps: int
    n_bfs: int
    alpha_z: float
    beta_z: float
    alpha_s: float
    reg_lambda: float
    dt: float
    y0: np.ndarray
    goal: np.ndarray
    weights: np.ndarray
    centers: np.ndarray
    widths: np.ndarray


def _canonical_phase(n_steps: int, dt: float, alpha_s: float) -> np.ndarray:
    """Compute the canonical phase variable sequence."""
    t = np.arange(n_steps, dtype=np.float64) * dt
    return np.exp(-alpha_s * t)


def _basis_functions(s: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Evaluate Gaussian basis functions for phase values."""
    # psi_{i}(s) = exp(-h_i * (s - c_i)^2)
    return np.exp(-widths[None, :] * (s[:, None] - centers[None, :]) ** 2)


def _generate_centers_widths(n_bfs: int, alpha_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Create basis centers and widths for the DMP forcing term."""
    centers = np.exp(-alpha_s * np.linspace(0.0, 1.0, n_bfs))
    widths = np.empty_like(centers)
    widths[:-1] = 1.0 / np.maximum((centers[1:] - centers[:-1]) ** 2, 1e-8)
    widths[-1] = widths[-2] if n_bfs > 1 else 1.0
    return centers, widths


def learn_dmp(
    y_demo: np.ndarray,
    n_bfs: int,
    alpha_z: float,
    beta_z: float,
    alpha_s: float,
    reg_lambda: float,
) -> tuple[DMPModel, dict[str, np.ndarray]]:
    """Fit a DMP model to a demonstrated trajectory."""
    y_demo = np.asarray(y_demo, dtype=np.float64)
    n_steps, n_dmps = y_demo.shape
    dt = 1.0 / max(1, n_steps - 1)

    y0 = y_demo[0].copy()
    goal = y_demo[-1].copy()

    yd = np.gradient(y_demo, dt, axis=0)
    ydd = np.gradient(yd, dt, axis=0)

    s = _canonical_phase(n_steps=n_steps, dt=dt, alpha_s=alpha_s)
    centers, widths = _generate_centers_widths(n_bfs=n_bfs, alpha_s=alpha_s)
    psi = _basis_functions(s=s, centers=centers, widths=widths)
    psi_sum = np.sum(psi, axis=1, keepdims=True) + 1e-12

    # DMP target forcing term:
    # f_target = (tau^2 y_dd - alpha_z(beta_z(g-y) - tau y_d)) / (g - y0)
    # with tau=1 and handling near-zero goal gap.
    goal_gap = goal - y0
    safe_gap = np.where(np.abs(goal_gap) < 1e-6, np.sign(goal_gap) * 1e-6 + 1e-6, goal_gap)

    f_target = np.empty_like(y_demo)
    for d in range(n_dmps):
        f_target[:, d] = (
            ydd[:, d] - alpha_z * (beta_z * (goal[d] - y_demo[:, d]) - yd[:, d])
        ) / safe_gap[d]

    # Linear regression on normalized basis term: f(s) = (sum_i psi_i w_i / sum_i psi_i) * s
    Phi = (psi / psi_sum) * s[:, None]
    reg = reg_lambda * np.eye(n_bfs)
    weights = np.zeros((n_dmps, n_bfs), dtype=np.float64)
    for d in range(n_dmps):
        lhs = Phi.T @ Phi + reg
        rhs = Phi.T @ f_target[:, d]
        weights[d] = np.linalg.solve(lhs, rhs)

    model = DMPModel(
        n_dmps=n_dmps,
        n_bfs=n_bfs,
        alpha_z=alpha_z,
        beta_z=beta_z,
        alpha_s=alpha_s,
        reg_lambda=reg_lambda,
        dt=dt,
        y0=y0,
        goal=goal,
        weights=weights,
        centers=centers,
        widths=widths,
    )

    return model, {
        's': s,
        'psi': psi,
        'f_target': f_target,
        'phi': Phi,
    }


def rollout_dmp(
    model: DMPModel,
    timesteps: int,
    y0: np.ndarray | None = None,
    goal: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Generate a trajectory from a learned DMP model."""
    y = np.zeros((timesteps, model.n_dmps), dtype=np.float64)
    yd = np.zeros_like(y)
    ydd = np.zeros_like(y)
    s_track = np.zeros(timesteps, dtype=np.float64)
    f_track = np.zeros_like(y)

    y_curr = model.y0.copy() if y0 is None else np.asarray(y0, dtype=np.float64).copy()
    goal_curr = model.goal.copy() if goal is None else np.asarray(goal, dtype=np.float64).copy()
    yd_curr = np.zeros(model.n_dmps, dtype=np.float64)
    s = 1.0

    goal_gap = goal_curr - y_curr
    safe_gap = np.where(np.abs(goal_gap) < 1e-6, np.sign(goal_gap) * 1e-6 + 1e-6, goal_gap)

    for t in range(timesteps):
        psi = np.exp(-model.widths * (s - model.centers) ** 2)
        denom = np.sum(psi) + 1e-12

        f = (model.weights @ psi) / denom
        f = f * s * safe_gap

        ydd_curr = model.alpha_z * (model.beta_z * (goal_curr - y_curr) - yd_curr) + f
        yd_curr = yd_curr + ydd_curr * model.dt
        y_curr = y_curr + yd_curr * model.dt

        s = s - model.alpha_s * s * model.dt

        y[t] = y_curr
        yd[t] = yd_curr
        ydd[t] = ydd_curr
        s_track[t] = s
        f_track[t] = f

    return {
        'y': y,
        'yd': yd,
        'ydd': ydd,
        's': s_track,
        'f': f_track,
    }


def save_model(path: str, model: DMPModel) -> None:
    """Serialize a learned DMP model to disk."""
    np.savez(
        path,
        n_dmps=model.n_dmps,
        n_bfs=model.n_bfs,
        alpha_z=model.alpha_z,
        beta_z=model.beta_z,
        alpha_s=model.alpha_s,
        reg_lambda=model.reg_lambda,
        dt=model.dt,
        y0=model.y0,
        goal=model.goal,
        weights=model.weights,
        centers=model.centers,
        widths=model.widths,
    )


def load_model(path: str) -> DMPModel:
    """Load a serialized DMP model from disk."""
    d = np.load(path)
    return DMPModel(
        n_dmps=int(d['n_dmps']),
        n_bfs=int(d['n_bfs']),
        alpha_z=float(d['alpha_z']),
        beta_z=float(d['beta_z']),
        alpha_s=float(d['alpha_s']),
        reg_lambda=float(d['reg_lambda']),
        dt=float(d['dt']),
        y0=d['y0'].astype(np.float64),
        goal=d['goal'].astype(np.float64),
        weights=d['weights'].astype(np.float64),
        centers=d['centers'].astype(np.float64),
        widths=d['widths'].astype(np.float64),
    )
