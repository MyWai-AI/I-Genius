import pandas as pd
import numpy as np

try:
    df = pd.read_csv("data/trajectories/hand_traj.csv")
    print("Trajectory Stats:")
    print(df[["X", "Y", "Z"]].describe())
    
    xyz = df[["X", "Y", "Z"]].dropna().values
    span = xyz.max(axis=0) - xyz.min(axis=0)
    print(f"\nMotion Span (X,Y,Z): {span}")
    
    dmp = np.load("data/dmp/object_xyz_dmp.npy")
    print(f"\nDMP Shape: {dmp.shape}")
    print(f"DMP Min: {dmp.min(axis=0)}")
    print(f"DMP Max: {dmp.max(axis=0)}")
    print(f"DMP Span: {dmp.max(axis=0) - dmp.min(axis=0)}")
    
except Exception as e:
    print(e)
