import numpy as np
import os

path = r"c:\Users\MohammadhoseinAbdoll\vilma-agent\data\SVO\frame_000000.npy"
if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    try:
        data = np.load(path)
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Min: {np.nanmin(data)}")
        print(f"Max: {np.nanmax(data)}")
        print(f"Mean: {np.nanmean(data)}")
        
        if data.ndim == 2:
            print(f"Sample data (corner): \n{data[:5, :5]}")
        elif data.ndim == 3:
             print(f"Sample data (corner channel 0): \n{data[:5, :5, 0]}")
        
    except Exception as e:
        print(f"Error loading npy: {e}")
