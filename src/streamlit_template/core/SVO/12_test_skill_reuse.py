import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

BASE_PATH = "data/SVO"
SKILL_PATH = os.path.join(BASE_PATH, "skills")
PLOT_PATH = os.path.join(BASE_PATH, "plots")
DMP_PATH  = os.path.join(BASE_PATH, "dmp")

os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(DMP_PATH, exist_ok=True)

# ---- DMP rollout ----
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
# ---------------------------------------

# Load skill
with open(os.path.join(SKILL_PATH, "move_skill.pkl"), "rb") as f:
    skill = pickle.load(f)

T = 200
start = np.array([0.2, 0.1, 0.7])
goal  = np.array([0.6, 0.15, 0.75])

traj = np.zeros((T, 3))

for i, name in enumerate(["x", "y", "z"]):
    w = skill["dimensions"][name]["w"]
    c = skill["dimensions"][name]["c"]
    h = skill["dimensions"][name]["h"]

    traj[:, i] = rollout_dmp(w, c, h, start[i], goal[i], T)

# ---- Save trajectory ----
np.save(os.path.join(DMP_PATH, "move_skill_test.npy"), traj)

df = pd.DataFrame(traj, columns=["x", "y", "z"])
df.to_csv(os.path.join(DMP_PATH, "move_skill_rollout.csv"), index=False)

# ---- Save plot ----
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(traj[:,0], traj[:,1], traj[:,2])
ax.set_title("Loaded Skill Rollout")

plt.savefig(os.path.join(PLOT_PATH, "move_skill_test.png"), dpi=300)
plt.close()

print("Skill reuse test saved.")
