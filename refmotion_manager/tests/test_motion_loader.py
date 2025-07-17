# loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,json,glob
import torch
from scipy.signal import savgol_filter
from refmotion_manager.motion_loader import RefMotionLoader


# load dataset for demonstration
from test_loader_cfg import ref_motion_cfg
ref_motion_cfg.time_between_frames=0.02
ref_motion_cfg.motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/fall*")
ref_motion_cfg.device="cpu"
ref_motion_cfg.ref_length_s=5
ref_motion = RefMotionLoader(ref_motion_cfg)


# Visualize selected data
#fields = ["root_pos_x", "root_vel_x_w", "left_hip_pitch_joint_dof_pos", "left_hip_pitch_joint_dof_vel","root_vel_y_w", "root_vel_z_w", "root_ang_vel_z_w"]
fields = ["root_pos_x", "root_pos_y","root_pos_z"]
# Setup subplots: one row, one subplot per field
fig, axes = plt.subplots(len(fields), 1, figsize=(10, 12), sharex=True)

#print(ref_motion.trajectory_fields)
index = [ref_motion.trajectory_fields.index(key) for key in fields]

for traj_idx in range(len(ref_motion.trajectories)):
    data = ref_motion.trajectories[traj_idx][10:100, index]
    for idx, key in enumerate(fields):
        axes[idx].plot(data[:, idx], label=f"raw traj {traj_idx} {key}")
        axes[idx].set_ylabel(key)
        axes[idx].grid(True)
        axes[idx].legend()

axes[-1].set_xlabel("Frames")
plt.tight_layout()
plt.show()
