from pathlib import Path
import csv
import numpy as np
from scipy.signal import savgol_filter


def generate_dmp_xyz(
    traj_csv: str,
    out_npy: str,
    smooth_window: int = 7,
    polyorder: int = 2,
):
 

    traj_csv = Path(traj_csv)
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    # Helpers

    def _get(row, *keys):
        for k in keys:
            if k in row and row[k] != "":
                return float(row[k])
        raise KeyError(f"Missing columns: tried {keys}, got {list(row.keys())}")

    # Load trajectory

    T, X, Y, Z = [], [], [], []

    with traj_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            T.append(float(row["TIME"]))
            X.append(_get(row, "X", "X_m", "X_px"))
            Y.append(_get(row, "Y", "Y_m", "Y_px"))
            Z.append(_get(row, "Z", "Z_m"))

    T = np.array(T)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    if len(X) < smooth_window:
        raise ValueError("Trajectory too short for smoothing")

    # Savitzky–Golay window must be odd
    if smooth_window % 2 == 0:
        smooth_window += 1

    # Smooth XYZ

    Xs = savgol_filter(X, smooth_window, polyorder)
    Ys = savgol_filter(Y, smooth_window, polyorder)
    Zs = savgol_filter(Z, smooth_window, polyorder)

    dmp_xyz = np.stack([Xs, Ys, Zs], axis=1)


    # Save

    np.save(out_npy, dmp_xyz)

    return {
        "dmp_npy": str(out_npy),
        "num_points": len(dmp_xyz),
        "start_xyz": dmp_xyz[0].tolist(),
        "end_xyz": dmp_xyz[-1].tolist(),
        "x_range_m": (float(Xs.min()), float(Xs.max())),
        "y_range_m": (float(Ys.min()), float(Ys.max())),
        "z_range_m": (float(Zs.min()), float(Zs.max())),
    }
