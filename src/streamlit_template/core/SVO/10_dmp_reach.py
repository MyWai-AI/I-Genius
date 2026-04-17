import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


BASE_PATH = "data/SVO"
TRAJ_PATH = os.path.join(BASE_PATH, "trajectories")
DMP_PATH = os.path.join(BASE_PATH, "dmp")
PLOT_PATH = os.path.join(BASE_PATH, "plots")

os.makedirs(DMP_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

reach = np.load(os.path.join(TRAJ_PATH, "reach_segment.npy"))

# ---------- Minimal DMP ----------
def train_dmp(y, n_basis=20, alpha_z=25.0, beta_z=6.25, alpha_x=4.0):
    T = len(y)
    dt = 1.0 / (T - 1)
    t = np.linspace(0, 1.0, T)

    # Canonical system
    x = np.exp(-alpha_x * t)

    # Basis functions
    c = np.exp(-alpha_x * np.linspace(0, 1, n_basis))
    h = np.ones(n_basis) * n_basis**1.5 / c

    dy = np.gradient(y, dt)
    ddy = np.gradient(dy, dt)

    f_target = (ddy - alpha_z*(beta_z*(y[-1] - y) - dy)) / (x + 1e-8)

    psi = np.exp(-h[None, :] * (x[:, None] - c[None, :])**2)
    w = np.linalg.lstsq(psi, f_target, rcond=None)[0]

    return w, c, h


def rollout_dmp(w, c, h, y0, g, T,
                alpha_z=25.0, beta_z=6.25, alpha_x=4.0):

    dt = 1.0 / (T - 1)

    y = np.zeros(T)
    dy = 0.0
    y[0] = y0
    x = 1.0

    for i in range(1, T):

        dx = -alpha_x * x
        x += dx * dt

        psi = np.exp(-h * (x - c)**2)
        f = (psi @ w) / (np.sum(psi) + 1e-8)
        f *= x

        ddy = alpha_z*(beta_z*(g - y[i-1]) - dy) + f

        dy += ddy * dt
        y[i] = y[i-1] + dy * dt

    return y

# ---------------------------------

T = reach.shape[0]

repro = np.zeros_like(reach)
adapted = np.zeros_like(reach)

new_start = reach[0] + np.array([0.05, 0.0, 0.0])
new_goal  = reach[-1] + np.array([0.05, 0.0, 0.0])

weights = {}

for dim, name in enumerate(["x", "y", "z"]):
    w, c, h = train_dmp(reach[:, dim])

    weights[name] = {
        "w": w,
        "c": c,
        "h": h
    }

    repro[:, dim] = rollout_dmp(
        w, c, h,
        reach[0, dim],
        reach[-1, dim],
        T
    )

    adapted[:, dim] = rollout_dmp(
        w, c, h,
        new_start[dim],
        new_goal[dim],
        T
    )


np.save(os.path.join(DMP_PATH, "reach_dmp_repro.npy"), repro)
np.save(os.path.join(DMP_PATH, "reach_dmp_adapted.npy"), adapted)

# Plot
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(reach[:,0], reach[:,1], reach[:,2], label="Demo")
ax.plot(repro[:,0], repro[:,1], repro[:,2], label="Repro")
ax.plot(adapted[:,0], adapted[:,1], adapted[:,2], label="Adapted")

ax.set_title("DMP — Reach")
ax.legend()

plt.savefig(os.path.join(PLOT_PATH, "dmp_reach.png"), dpi=300)
plt.close()

print("Reach DMP complete.")

SKILL_PATH = os.path.join(BASE_PATH, "skills")
os.makedirs(SKILL_PATH, exist_ok=True)

reach_skill = {
    "skill_name": "reach",
    "frame": "camera",
    "params": {
        "alpha_z": 25.0,
        "beta_z": 6.25,
        "alpha_x": 4.0,
        "n_basis": 20
    },
    "dimensions": weights
}

with open(os.path.join(SKILL_PATH, "reach_skill.pkl"), "wb") as f:
    pickle.dump(reach_skill, f)

print("Reach skill saved.")
