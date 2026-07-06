import pandas as pd
import numpy as np
from pathlib import Path
import glob

def analyze_detections():
    print("--- 1. Analyzing Object Detections (Pixel Space) ---")
    try:
        df = pd.read_csv("data/objects/objects_detections.csv")
        # Calculate centroids
        df["cx"] = (df["xmin"] + df["xmax"]) / 2
        df["cy"] = (df["ymin"] + df["ymax"]) / 2
        
        print(df[["cx", "cy"]].describe())
        
        y_range = df["cy"].max() - df["cy"].min()
        print(f"\nPixel Y Range: {y_range:.1f} pixels (Image H=1080 usually)")
        if y_range < 20:
            print("(!) The object hardly moved vertically in the image.")
        else:
            print(f"Object moved {y_range:.1f} pixels vertically.")

    except Exception as e:
        print(f"Could not read objects csv: {e}")

def analyze_depth_at_centroids():
    print("\n--- 2. Analyzing Depth Maps (Z Space) ---")
    try:
        # Load objects to get centroids
        df = pd.read_csv("data/objects/objects_detections.csv")
        df["cx"] = ((df["xmin"] + df["xmax"]) / 2).astype(int)
        df["cy"] = ((df["ymin"] + df["ymax"]) / 2).astype(int)
        
        # Find SVO files
        npy_files = sorted(glob.glob("data/SVO/frame_*.npy"))
        if not npy_files:
            print("No .npy files found in data/SVO/")
            return
            
        print(f"Found {len(npy_files)} depth frames.")
        
        depths = []
        for i, row in df.iterrows():
            frame_idx = int(row['frame_idx'])
            # Assuming frame_idx matches file list order or naming
            # Let's try to match by name
            fname = f"data/SVO/frame_{frame_idx:06d}.npy"
            if Path(fname).exists():
                dmap = np.load(fname)
                # Sample depth at centroid
                # check bounds
                H, W = dmap.shape
                cx, cy = row['cx'], row['cy']
                
                if 0 <= cx < W and 0 <= cy < H:
                    z_val = dmap[cy, cx]
                    depths.append(z_val)
        
        if depths:
            ds = pd.Series(depths)
            print("\nDepth Values at Object Centroid (Meters):")
            print(ds.describe())
            print(f"Depth Span: {ds.max() - ds.min():.4f} m")
        else:
            print("No matching depth samples extracted.")

    except Exception as e:
        print(f"Depth analysis failed: {e}")

if __name__ == "__main__":
    analyze_detections()
    analyze_depth_at_centroids()
