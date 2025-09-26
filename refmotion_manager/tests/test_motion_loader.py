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
ref_motion_cfg.motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/Mm*_fps30.pkl")
ref_motion_cfg.device="cuda:0"
ref_motion_cfg.ref_length_s=0.8
ref_motion_cfg.clip_num = 5
ref_motion = RefMotionLoader(ref_motion_cfg)

# Visualize selected data
fields = ["root_pos_x", "root_vel_x_w", "left_hip_pitch_joint_dof_pos", "left_hip_pitch_joint_dof_vel","root_vel_y_w", "root_vel_z_w", "root_ang_vel_z_w"]
fields = ref_motion_cfg.expressive_goal_fields[:3]
# Setup subplots: one row, one subplot per field
fig, axes = plt.subplots(len(fields), 1, figsize=(10, 12), sharex=True)

frame_start = 0;
frame_end = 47;

#print(ref_motion.trajectory_fields)
index = [ref_motion.trajectory_fields.index(key) for key in fields]
for traj_idx in range(len(ref_motion.trajectories)):
    data = ref_motion.trajectories[traj_idx][frame_start:frame_end, index].cpu().numpy()
    for idx, key in enumerate(fields):
        time = np.linspace(0, (frame_end-frame_start)/30, frame_end - frame_start)
        axes[idx].plot(time, data[:, idx], label=f"raw traj {traj_idx} {key}")
        axes[idx].set_ylabel(key)
        axes[idx].grid(True)
        axes[idx].legend()

axes[-1].set_xlabel("Frames")
plt.tight_layout()


fig, axes = plt.subplots(len(fields), 1, figsize=(10, 12), sharex=True)
for clip_idx in range(len(ref_motion.clip_idxs)):
    frame_num = int(ref_motion_cfg.ref_length_s/0.02)
    step_data = []
    amp_ref = []
    for idx in range (frame_num):
        ref_motion.step()
        step_data.append(ref_motion.expressive_goal[clip_idx,:3])
        amp_ref.append(ref_motion.amp_expert[clip_idx,:])
        if idx == ref_motion.preloaded_s.shape[1]-3:
            ref_motion.reset()

    #import pdb;pdb.set_trace()
    step_data = torch.stack(step_data, dim=0).squeeze().cpu().numpy()  # convert all at once

    amp_ref = torch.stack(amp_ref,dim=0).squeeze().cpu().numpy()

    for idx, key in enumerate(fields):
        time = np.linspace(0, ref_motion_cfg.ref_length_s, frame_num)
        axes[idx].plot(time, step_data[:, idx], label=f"preload traj {clip_idx} {key}")
        axes[idx].set_ylabel(key)
        axes[idx].grid(True)
        axes[idx].legend()



# plot amp ref
fields = ref_motion_cfg.style_fields[:2]
fig, axes = plt.subplots(len(fields), 1, figsize=(10, 12), sharex=True)
for clip_idx in range(len(ref_motion.clip_idxs)):
    frame_num = int(ref_motion_cfg.ref_length_s/0.02)
    for idx, key in enumerate(fields):
        time = np.linspace(0, ref_motion_cfg.ref_length_s, frame_num)
        axes[idx].plot(time, amp_ref[:, idx], label=f"current states of preload traj {clip_idx} {key}")
        axes[idx].plot(time, amp_ref[:, idx+21], label=f"next states of preload traj {clip_idx} {key}")
        axes[idx].set_ylabel(key)
        axes[idx].grid(True)
        axes[idx].legend()



plt.show()
