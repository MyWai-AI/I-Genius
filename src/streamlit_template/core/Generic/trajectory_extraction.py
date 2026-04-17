# # trajectory_extraction.py  — Hybrid 2D + 3D (CLARITY CORRECTED ONLY)

# from pathlib import Path
# from typing import Optional
# import csv
# import numpy as np
# import cv2
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import streamlit as st


#  UTILS


# def _read_frames_index(frames_dir: str):
#     idx = Path(frames_dir) / "frames_index.csv"
#     if not idx.exists():
#         raise FileNotFoundError(f"frames_index.csv not found at {idx}")
#     with idx.open("r", newline="", encoding="utf-8") as f:
#         return list(csv.DictReader(f))


# def _group_hands_by_frame(hands_csv: str):
#     p = Path(hands_csv)
#     if not p.exists():
#         raise FileNotFoundError(f"Hands CSV missing: {p}")

#     groups = {}
#     with p.open("r", newline="", encoding="utf-8") as f:
#         for r in csv.DictReader(f):
#             fi = int(float(r["frame_idx"]))
#             groups.setdefault(fi, []).append(r)
#     return groups


# def _best_object_by_frame(objects_csv: Optional[str]):
#     if not objects_csv:
#         return {}

#     p = Path(objects_csv)
#     if not p.exists():
#         return {}

#     best = {}
#     with p.open("r", newline="", encoding="utf-8") as f:
#         for r in csv.DictReader(f):
#             fi = int(float(r["frame_idx"]))
#             conf = float(r["confidence"])
#             if fi not in best or conf > float(best[fi]["confidence"]):
#                 best[fi] = r
#     return best


# def _nan_interp(y):
#     y = y.copy()
#     n = len(y)
#     idx = np.arange(n)
#     ok = ~np.isnan(y)

#     if ok.sum() == 0:
#         return y

#     first = np.argmax(ok)
#     last = n - 1 - np.argmax(ok[::-1])
#     y[:first] = y[first]
#     y[last + 1:] = y[last]

#     ok = ~np.isnan(y)
#     y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
#     return y


# def _moving_avg(y, k=5):
#     if k <= 1:
#         return y
#     pad = k // 2
#     ypad = np.pad(y, (pad, pad), mode="edge")
#     kernel = np.ones(k) / k
#     return np.convolve(ypad, kernel, mode="valid")


# def _load_intrinsics(base_path: Path):
#     intr_file = base_path / "camera_intrinsics.npz"
#     if not intr_file.exists():
#         return None

#     d = np.load(intr_file)
#     return {
#         "fx": float(d["fx"]),
#         "fy": float(d["fy"]),
#         "cx": float(d["cx"]),
#         "cy": float(d["cy"]),
#         "scale": float(d["depth_scale"]),
#     }


# def deproject(u, v, depth, intr):
#     fx, fy = intr["fx"], intr["fy"]
#     cx, cy = intr["cx"], intr["cy"]
#     Z = depth * intr["scale"]
#     if Z <= 0:
#         return np.nan, np.nan, np.nan
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy
#     return X, Y, Z


# # CACHED TRAJECTORY PLOT (CLARITY CORRECTED)

# @st.cache_resource(show_spinner=False)
# def _cached_trajectory_plot(
#     Xh, Yh, Zh,
#     Xo, Yo, Zo,
#     start_i, end_i,
#     pick_idx,
#     cluster_k,
#     dist_thresh,
#     out_plot_str
# ):
#     frames = np.arange(start_i, end_i + 1)

#     Xh = np.array(Xh[start_i:end_i + 1])
#     Yh = np.array(Yh[start_i:end_i + 1])
#     Zh = np.array(Zh[start_i:end_i + 1])
#     Xo = np.array(Xo[start_i:end_i + 1])
#     Yo = np.array(Yo[start_i:end_i + 1])
#     Zo = np.array(Zo[start_i:end_i + 1])

#     Dist = np.sqrt((Xh - Xo)**2 + (Yh - Yo)**2 + (Zh - Zo)**2)

#     valid_z = not np.all(np.isnan(Zh))
#     rows = 4 if valid_z else 3

#     fig, ax = plt.subplots(rows, 1, figsize=(14, 10), sharex=True)

#     data = [Xh, Yh] + ([Zh] if valid_z else [])
#     labels = ["X", "Y"] + (["Z"] if valid_z else [])

#     clusters = None
#     if cluster_k > 1:
#         M = np.vstack([Xh, Yh, Zh]).T
#         M = np.nan_to_num(M, nan=np.nanmean(M))
#         try:
#             clusters = KMeans(cluster_k, n_init=10).fit(M).labels_
#         except:
#             clusters = None

#     cmap = plt.get_cmap("tab10")

#     for k, (y, label) in enumerate(zip(data, labels)):
#         if clusters is None:
#             ax[k].plot(frames, y, lw=3)
#         else:
#             for cl in np.unique(clusters):
#                 idxs = np.where(clusters == cl)[0]
#                 ax[k].plot(
#                     frames[idxs],
#                     y[idxs],
#                     "o-",
#                     markersize=6,
#                     linewidth=2.5,
#                     alpha=0.85,
#                     color=cmap(cl),
#                     label=f"Cluster {cl}" if k == 0 else None,
#                 )

#         ax[k].set_ylabel(label)
#         ax[k].grid(True)

#     # Distance plot (clarity improvement)
#     d_ax = ax[-1]
#     d_ax.plot(frames, Dist, color="black", lw=2, label="Hand–Object Distance")
#     d_ax.axhline(dist_thresh, color="red", linestyle="--", label="Grasp Threshold")
#     d_ax.set_ylabel("Distance")
#     d_ax.legend()
#     d_ax.grid(True)

#     if pick_idx is not None:
#         for a in ax:
#             a.axvline(pick_idx, color="blue", linestyle="--")
#             a.text(
#                 pick_idx + 1,
#                 a.get_ylim()[1] * 0.9,
#                 "Grasp",
#                 color="blue",
#             )

#     ax[-1].set_xlabel("Frame index")

#     if clusters is not None:
#         ax[0].legend()

#     plt.tight_layout()
#     plt.savefig(out_plot_str, dpi=200)
#     plt.close()

#     return out_plot_str



# # MAIN FUNCTION (UNCHANGED LOGIC)


# def extract_hand_object_trajectory(
#     frames_dir="data/Generic/frames",
#     hands_csv="data/Generic/hands/hands_landmarks.csv",
#     objects_csv="data/Generic/objects/objects_detections.csv",
#     out_csv="data/Generic/dmp/hand_traj.csv",
#     out_plot="data/Generic/dmp/trajectory_plot.png",
#     fingertip_id=8,
#     smooth_window=5,
#     dist_thresh=0.06,
#     cluster_k=3,
#     use_3d=True
# ):
#     frames_dir = Path(frames_dir)
#     base_path = frames_dir.parent

#     idx_rows = _read_frames_index(frames_dir)
#     frame_map = {int(r["frame_idx"]): r for r in idx_rows}

#     hands = _group_hands_by_frame(hands_csv)
#     objs = _best_object_by_frame(objects_csv)
#     intr = _load_intrinsics(base_path) if use_3d else None

#     T, Xh, Yh, Zh, Xo, Yo, Zo = [], [], [], [], [], [], []

#     ft_x = f"lm{fingertip_id}_x"
#     ft_y = f"lm{fingertip_id}_y"

#     H = W = None

#     for _, row in sorted(frame_map.items()):
#         fi = int(row["frame_idx"])
#         T.append(float(row["time_sec"]))

#         rgb_path = base_path / "frames" / row["filename"]
#         if H is None:
#             im = cv2.imread(str(rgb_path))
#             H, W = im.shape[:2]

#         depth_img = None
#         if row.get("depth_file"):
#             dp = base_path / "frames" / row["depth_file"]
#             if dp.exists():
#                 depth_img = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)

#         # Hand
#         candidates = hands.get(fi, [])
#         if not candidates:
#             Xh.append(np.nan); Yh.append(np.nan); Zh.append(np.nan)
#         else:
#             h = candidates[0]
#             u = float(h[ft_x]) * W
#             v = float(h[ft_y]) * H
#             if use_3d and intr and depth_img is not None:
#                 d = float(depth_img[int(v), int(u)])
#                 X, Y, Z = deproject(u, v, d, intr)
#             else:
#                 X, Y, Z = (u / W, v / H, np.nan)
#             Xh.append(X); Yh.append(Y); Zh.append(Z)

#         # Object
#         if fi in objs:
#             o = objs[fi]
#             uo = (int(o["xmin"]) + int(o["xmax"])) / 2
#             vo = (int(o["ymin"]) + int(o["ymax"])) / 2
#             if use_3d and intr and depth_img is not None:
#                 d = float(depth_img[int(vo), int(uo)])
#                 Xo_, Yo_, Zo_ = deproject(uo, vo, d, intr)
#             else:
#                 Xo_, Yo_, Zo_ = (uo / W, vo / H, np.nan)
#         else:
#             Xo_, Yo_, Zo_ = (np.nan, np.nan, np.nan)

#         Xo.append(Xo_); Yo.append(Yo_); Zo.append(Zo_)

#     # Smooth
#     Xh = _moving_avg(_nan_interp(np.array(Xh)), smooth_window)
#     Yh = _moving_avg(_nan_interp(np.array(Yh)), smooth_window)
#     Zh = _moving_avg(_nan_interp(np.array(Zh)), smooth_window)
#     Xo = _nan_interp(np.array(Xo))
#     Yo = _nan_interp(np.array(Yo))
#     Zo = _nan_interp(np.array(Zo))

#     Dist = np.sqrt((Xh - Xo)**2 + (Yh - Yo)**2 + (Zh - Zo)**2)
#     pick_idx = np.where(Dist < dist_thresh)[0]
#     pick_idx = int(pick_idx[0]) if len(pick_idx) else None

#     start_i = max(0, pick_idx - 10) if pick_idx else 0
#     end_i = len(T) - 1

#     out_path = Path(out_csv)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     with out_path.open("w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["TIME", "X", "Y", "Z"])
#         for i in range(start_i, end_i + 1):
#             w.writerow([T[i], Xh[i], Yh[i], Zh[i]])

#     out_plot = Path(out_plot)
#     out_plot.parent.mkdir(parents=True, exist_ok=True)

#     _cached_trajectory_plot(
#         Xh.tolist(), Yh.tolist(), Zh.tolist(),
#         Xo.tolist(), Yo.tolist(), Zo.tolist(),
#         start_i, end_i,
#         pick_idx,
#         cluster_k,
#         dist_thresh,
#         str(out_plot),
#     )

#     return {
#         "traj_csv": str(out_path),
#         "traj_plot": str(out_plot),
#         "pick_index": pick_idx,
#         "start_index": start_i,
#         "end_index": end_i,
#     }






# old file


# trajectory_extraction.py  — Hybrid 2D + 3D

from pathlib import Path
from typing import Optional, Dict, Any, List
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st



# UTILS


def _read_frames_index(frames_dir: str):
    idx = Path(frames_dir) / "frames_index.csv"
    if not idx.exists():
        raise FileNotFoundError(f"frames_index.csv not found at {idx}")
    rows = []
    with idx.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def _group_hands_by_frame(hands_csv: str):
    p = Path(hands_csv)
    if not p.exists():
        raise FileNotFoundError(f"Hands CSV missing: {p}")

    groups = {}
    with p.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fi = int(float(r["frame_idx"]))
            groups.setdefault(fi, []).append(r)
    return groups


def _best_object_by_frame(objects_csv: Optional[str]):
    if not objects_csv:
        return {}
    p = Path(objects_csv)
    if not p.exists():
        return {}

    best = {}
    with p.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fi = int(float(r["frame_idx"]))
            conf = float(r["confidence"])
            if fi not in best or conf > float(best[fi]["confidence"]):
                best[fi] = r
    return best


def _nan_interp(y):
    y = y.copy()
    n = len(y)
    idx = np.arange(n)
    ok = ~np.isnan(y)

    if ok.sum() == 0:
        return y

    # edge fill
    first = np.argmax(ok)
    last = n - 1 - np.argmax(ok[::-1])
    y[:first] = y[first]
    y[last+1:] = y[last]

    ok = ~np.isnan(y)
    y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


def _moving_avg(y, k=5):
    if k <= 1:
        return y
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(ypad, kernel, mode="valid")


def _load_intrinsics(base_path: Path):
    # Try SVO path first as requested, then base path
    candidates = [
        base_path.parent / "SVO" / "camera_intrinsics.npz",
        base_path / "camera_intrinsics.npz"
    ]
    
    intr_file = None
    for c in candidates:
        if c.exists():
            intr_file = c
            break
            
    if intr_file is None:
        # Fallback to defaults (approx RealSense D435 1080p)
        print("[WARN] camera_intrinsics.npz not found. Using defaults.")
        return {
            "fx": 1380.0,
            "fy": 1380.0,
            "cx": 960.0,
            "cy": 540.0,
            "scale": 1.0  # SVO .npy is usually in meters
        }

    try:
        d = np.load(intr_file)
        return {
            "fx": float(d["fx"]),
            "fy": float(d["fy"]),
            "cx": float(d["cx"]),
            "cy": float(d["cy"]),
            "scale": float(d["depth_scale"])
        }
    except Exception as e:
        print(f"[ERROR] Failed to load intrinsics: {e}. Using defaults.")
        return {
            "fx": 1380.0,
            "fy": 1380.0,
            "cx": 960.0,
            "cy": 540.0,
            "scale": 1.0
        }


def deproject(u, v, depth, intr):
    fx = intr["fx"]; fy = intr["fy"]
    cx = intr["cx"]; cy = intr["cy"]
    Z = depth * intr["scale"]
    if Z <= 0:
        return np.nan, np.nan, np.nan
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z


# Caching

@st.cache_resource(show_spinner=False)
def _cached_trajectory_plot(
    Xh, Yh, Zh,
    start_i, end_i,
    pick_idx,
    cluster_k,
    out_plot_str
):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import KMeans

    frames = np.arange(start_i, end_i + 1)
    data = [
        np.array(Xh[start_i:end_i+1]),
        np.array(Yh[start_i:end_i+1]),
        np.array(Zh[start_i:end_i+1]),
    ]
    labels = ["X", "Y", "Z"]

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    clusters = None
    if cluster_k > 1:
        M = np.vstack(data).T
        M = np.nan_to_num(M, nan=np.nanmean(M))
        try:
            clusters = KMeans(cluster_k).fit(M).labels_
        except:
            clusters = None

    cmap = plt.get_cmap("tab10")

    for k in range(3):
        y = data[k]
        if clusters is None:
            ax[k].plot(frames, y, lw=2)
        else:
            for cl in np.unique(clusters):
                idxs = np.where(clusters == cl)[0]
                ax[k].plot(
                    frames[idxs],
                    y[idxs],
                    ".",
                    color=cmap(cl),
                    label=f"Cluster {cl}" if k == 0 else None,
                )

        ax[k].set_ylabel(labels[k])
        ax[k].grid(True)

    if pick_idx is not None:
        for a in ax:
            a.axvline(pick_idx, color="blue", linestyle="--")
            a.text(
                pick_idx + 3,
                a.get_ylim()[1] * 0.9,
                "Grasp",
                color="blue",
            )

    ax[-1].set_xlabel("Frame index")
    if clusters is not None:
        ax[0].legend()

    plt.tight_layout()
    plt.savefig(out_plot_str, dpi=200)
    plt.close()

    return out_plot_str



# MAIN FUNCTION


def extract_hand_object_trajectory(
    frames_dir="data/Generic/frames",
    hands_csv="data/Generic/hands/hands_landmarks.csv",
    objects_csv="data/Generic/objects/objects_detections.csv",
    out_csv="data/Generic/dmp/hand_traj.csv",
    out_plot="data/Generic/dmp/trajectory_plot.png",
    fingertip_id=8,
    smooth_window=5,
    dist_thresh=0.06,
    cluster_k=3,
    use_3d=True
):
    frames_dir = Path(frames_dir)
    base_path = frames_dir.parent

    # Load data
    idx_rows = _read_frames_index(frames_dir)
    frame_map = {int(r["frame_idx"]): r for r in idx_rows}

    hands = _group_hands_by_frame(hands_csv)
    objs = _best_object_by_frame(objects_csv)
    intr = _load_intrinsics(base_path) if use_3d else None

    H = W = None

    T = []
    Xh, Yh, Zh = [], [], []
    Xo, Yo, Zo = [], [], []

    ft_x = f"lm{fingertip_id}_x"
    ft_y = f"lm{fingertip_id}_y"


    # LOOP FRAMES

    for _, row in sorted(frame_map.items()):
        fi = int(row["frame_idx"])
        T.append(float(row["time_sec"]))

        # load image size (first frame)
        # load image size (first frame)
        rgb_path = frames_dir / row["filename"]
        if H is None:
            im = cv2.imread(str(rgb_path))
            if im is None:
                raise RuntimeError(f"Cannot read RGB frame: {rgb_path}")
            H, W = im.shape[:2]

        # depth path (optional)
        depth_file = row.get("depth_file")
        depth_img = None
        depth_scale_override = None # New flag

        if depth_file:
            dp = frames_dir / depth_file
            if dp.exists():
                depth_img = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)

        if depth_img is None:
            # SVO usually at data/SVO. frames_dir is data/Generic/frames/{hash}
            # so data is frames_dir.parent.parent
            svo_npy = frames_dir.parent.parent / "SVO" / f"frame_{fi:06d}.npy"
            if svo_npy.exists():
                try:
                    # Load depth map (meters)
                    depth_loaded = np.load(str(svo_npy))
                    depth_img = depth_loaded
                    depth_scale_override = 1.0 # SVO is already in meters
                except Exception as e:
                    print(f"[ERROR] Failed to load SVO depth {svo_npy}: {e}")
            else:
                # Debug only first few failures to avoid spam
                if fi < 5:
                    print(f"[DEBUG] SVO depth file not found: {svo_npy}")

        # HAND POINT

        candidates = hands.get(fi, [])
        if not candidates:
            Xh.append(np.nan); Yh.append(np.nan); Zh.append(np.nan)
        else:
            h = candidates[0]  # simplest choice
            u = float(h[ft_x]) * W
            v = float(h[ft_y]) * H

            if use_3d and intr and depth_img is not None:
                d = float(depth_img[int(round(v)), int(round(u))])
                # Temporarily override scale if needed
                real_intr = intr.copy()
                if depth_scale_override is not None:
                    real_intr["scale"] = depth_scale_override
                
                X,Y,Z = deproject(u, v, d, real_intr)
            else:
                X,Y,Z = (u/W, v/H, np.nan)

            Xh.append(X); Yh.append(Y); Zh.append(Z)

        # OBJECT CENTER

        if fi in objs:
            o = objs[fi]
            uo = (int(o["xmin"]) + int(o["xmax"])) / 2
            vo = (int(o["ymin"]) + int(o["ymax"])) / 2

            if use_3d and intr and depth_img is not None:
                d = float(depth_img[int(round(vo)), int(round(uo))])
                
                real_intr = intr.copy()
                if depth_scale_override is not None:
                    real_intr["scale"] = depth_scale_override

                Xo_,Yo_,Zo_ = deproject(uo, vo, d, real_intr)
            else:
                Xo_,Yo_,Zo_ = (uo/W, vo/H, np.nan)
        else:
            Xo_,Yo_,Zo_ = (np.nan, np.nan, np.nan)

        Xo.append(Xo_); Yo.append(Yo_); Zo.append(Zo_)


    # CLEAN + SMOOTH

    Xh = _moving_avg(_nan_interp(np.array(Xh)), smooth_window)
    Yh = _moving_avg(_nan_interp(np.array(Yh)), smooth_window)
    Zh = _moving_avg(_nan_interp(np.array(Zh)), smooth_window)

    Xo = _nan_interp(np.array(Xo))
    Yo = _nan_interp(np.array(Yo))
    Zo = _nan_interp(np.array(Zo))


    # PICK DETECTION

    Dist = np.sqrt((Xh - Xo)**2 + (Yh - Yo)**2 + (Zh - Zo)**2)
    valid = ~np.isnan(Dist)
    pick_idx = None
    if valid.any():
        hits = np.where(Dist < dist_thresh)[0]
        if len(hits) > 0:
            pick_idx = int(hits[0])

    start_i = 0
    end_i = len(T) - 1


    # SAVE TRAJECTORY CSV

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["TIME", "X", "Y", "Z"])
        for i in range(start_i, end_i+1):
            w.writerow([T[i], Xh[i], Yh[i], Zh[i]])



    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    _cached_trajectory_plot(
        Xh.tolist(),
        Yh.tolist(),
        Zh.tolist(),
        start_i,
        end_i,
        pick_idx,
        cluster_k,
        str(out_plot),
    )

    return {
        "traj_csv": str(out_path),
        "traj_plot": str(out_plot),
        "pick_index": pick_idx,
        "start_index": start_i,
        "end_index": end_i,
    }
