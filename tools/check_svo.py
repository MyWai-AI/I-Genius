import numpy as np
base = 'data/SVO'
sid = 'OpenArm001'
sr = np.load(f'{base}/dmp/{sid}/skill_reuse_traj.npy')
rp = np.load(f'{base}/segmentation/{sid}/release_pos.npy')
pp = np.load(f'{base}/segmentation/{sid}/post_release_pos.npy')

ra = np.load(f'{base}/dmp/{sid}/reach_adapted.npy')
ma = np.load(f'{base}/dmp/{sid}/move_adapted.npy')

print(f"reach_adapted: {ra.shape}, last={ra[-1]}")
print(f"move_adapted: {ma.shape}")
if len(ma) > 0:
    print(f"  move first: {ma[0]}")
    print(f"  move last:  {ma[-1]}")
print(f"release_pos: {rp}")
print(f"post_release_pos: {pp}")
print()
print(f"SR traj[83] (grasp): {sr[83]}")
print(f"SR traj[84] (move start): {sr[84]}")
print(f"SR traj[193] (release): {sr[193]}")
print(f"SR traj[194] (post-release): {sr[194]}")
print(f"SR traj[-1] (end): {sr[-1]}")
print()

# Check: does the move endpoint match release?
print("=== Key comparison ===")
print(f"move_adapted[-1]:       {ma[-1] if len(ma)>0 else 'N/A'}")
print(f"release_pos:            {rp}")
print(f"hand_smooth[release]:   {np.load(f'{base}/hands/{sid}/hand_3d_smooth.npy')[193]}")
print(f"Distance move_end to release: {np.linalg.norm(ma[-1] - rp) if len(ma)>0 else 'N/A'}")
