import os
import glob
import sys
import time
import argparse
import json
import os.path as osp
import scipy.ndimage.filters as filters
sys.path.append(os.getcwd())
import torch
import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf
from refmotion_manager.motion_loader import RefMotionLoader, RefMotionCfg

# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger("vis_mj")


def add_visual_capsule(scene, pos, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom

    #viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(
          scene.geoms[scene.ngeom-1],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[radius, 0, 0],
          pos=0.1*np.array(pos),
          mat=np.eye(3).flatten(),
          rgba=0.5*np.array(rgba)
      )



def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        logger.info("Reset")
    #else:
    #    logger.info("not mapped", chr(keycode))
    
    

@hydra.main(version_base=None, config_path="./cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    #1) loading ref traj by refmotion_manager
    # load dataset for demonstration
    from refmotion_manager.tests.test_loader_cfg import ref_motion_cfg
    ref_motion_cfg.time_between_frames = 0.02

    dataset = cfg.get("dataset",None)
    motion_files = "./"+os.path.join(dataset.folder, dataset.file)
    print(motion_files)
    ref_motion_cfg.motion_files= glob.glob(motion_files)
    ref_motion_cfg.device="cpu"
    ref_motion_cfg.ref_length_s=None
    ref_motion_cfg.clip_num = 1
    ref_motion = RefMotionLoader(ref_motion_cfg)
    
    #-) or gettting ref traj from a real-time motion capture system

    #2) build robot model
    asset = cfg.get("asset",None)
    humanoid_xml = os.path.join(asset.root, asset.xml)
    logger.info(f"humanoid xlm is {humanoid_xml}")
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    joint_names = [mj_model.joint(i).name for i in range(1, mj_model.njnt)]

    #3) cfg 
    dt = 0.02
    root_fields= ["root_pos_x","root_pos_y", "root_pos_z","root_rot_w","root_rot_x","root_rot_y","root_rot_z"]
    joint_dof_fields = [key+"_dof_pos" for key in joint_names]
    data_fields = root_fields + joint_dof_fields
    print(f" data fields : {data_fields}")
    index = [ref_motion.trajectory_fields.index(key) for key in data_fields]

    robot_bodies = ["left_elbow_link", "right_elbow_link","left_hip_pitch_link","right_hip_pitch_link","left_shoulder_roll_link","right_shoulder_roll_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link"] #"left_wrist_roll_link","right_wrist_roll_link"]
    body_index = [[ref_motion.trajectory_fields.index(key + kk) for kk in ["_pos_x_w", "_pos_y_w", "_pos_z_w"]] for key in robot_bodies]
    print(f" body index : {body_index}")

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back, show_left_ui=False, show_right_ui=False) as viewer:
        cam = viewer.cam
        cam.distance = 4.0 ;cam.azimuth = 135; cam.elevation = -10; cam.lookat = [0,0,0]
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING;cam.trackbodyid=1;

        # adding gemo to dislay key joint position
        for _ in range(20):
            add_visual_capsule(viewer.user_scn, np.zeros(3), 0.05, np.array([1, 0, 0, 1]))
        for _ in range(20):
            add_visual_capsule(viewer.user_scn, np.zeros(3), 0.05,  np.array([0, 1, 0, 1]))
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            step_start = time.time()
            mj_data.qpos[:] = ref_motion.data[:,index].cpu().numpy()
        
            mujoco.mj_forward(mj_model, mj_data)
            ref_motion.step()

            if ref_motion.frame_idx > ref_motion.clip_frame_num:
                ref_motion.reset()
                logger.info(f"Reset")

            # visualizing robot joints
            if robot_bodies is not None:
                for i,idx in enumerate(body_index):
                    viewer.user_scn.geoms[20+i].pos = ref_motion.data[:, idx]
            
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
