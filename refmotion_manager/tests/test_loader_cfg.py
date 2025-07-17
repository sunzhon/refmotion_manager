from refmotion_manager.motion_loader import RefMotionLoader
# load dataset for demonstration:
import glob
import os
from refmotion_manager.motion_loader import RefMotionLoader
from refmotion_manager.motion_loader import RefMotionCfg


############## Demo trajectory ###################
num_envs=4096

# 27 joints
motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/data/lus2_joint27/pkl/Append_Select_S29_*")
motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint27/pkl/CMU_CMU_12_12_04_p*")
#motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/data/lus2_joint27/pkl/*")
random_start=True
amp_obs_frame_num = 2 # minimal is 1, no history amp obs

############## Init states (carefully)  ###################
# constants
INIT_ROOT_STATE_FIELDS = [
            "root_pos_x",
            "root_pos_y",
            "root_pos_z",
            "root_rot_w",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
            "root_vel_x_b",
            "root_vel_y_b",
            "root_vel_z_b",
            "root_ang_vel_x_b",
            "root_ang_vel_y_b",
            "root_ang_vel_z_b",
        ]

# 27 joints
all_joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
'left_wrist_roll_joint', 'right_wrist_roll_joint'
]


# NOTE, this should follow the order of that in env when loading usd model
INIT_STATE_FIELDS = INIT_ROOT_STATE_FIELDS + [key+"_dof_pos" for key in all_joint_names] + [key+"_dof_vel" for key in all_joint_names]


############## Style  (AMP) ########################
# not need to include that for normal command (base_velocity command)
#style_goal_fields = None

style_root_fields = [
        "root_pos_z",
        "root_rot_w",
        "root_rot_x",
        "root_rot_y",
        "root_rot_z",
        "root_vel_x_b",
        "root_vel_y_b",
        "root_vel_z_b",
        "root_ang_vel_x_b",
        "root_ang_vel_y_b",
        "root_ang_vel_z_b",
        ]

# 27 joints
style_joint_name = [
        'left_hip_roll_joint', 'right_hip_roll_joint',  'left_hip_yaw_joint', 'right_hip_yaw_joint', 
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 
        'left_knee_joint', 'right_knee_joint', 
        'torso_joint',
        'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
        'left_elbow_joint', 'right_elbow_joint', 
        'left_wrist_yaw_joint', 'right_wrist_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
        'left_wrist_roll_joint', 'right_wrist_roll_joint'
        ]

style_body_name = ["left_elbow_link", "right_elbow_link","left_hip_pitch_link","right_hip_pitch_link","left_shoulder_roll_link","right_shoulder_roll_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link","left_wrist_roll_link","right_wrist_roll_link"] + ["left_elbow_link", "left_hip_pitch_link", "right_hip_pitch_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link"]

style_fields = style_root_fields + \
        ([key + "_dof_pos" for key in style_joint_name] if style_joint_name is not None else []) + \
        ([key + "_dof_vel" for key in style_joint_name] if style_joint_name is not None else []) + \
        ([k1 + k2 for k1 in style_body_name for k2 in ["_pos_x_b", "_pos_y_b", "_pos_z_b"]] if style_body_name is not None else [])


# carefully to change this
style_goal_fields=[
            "root_rot_w",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
        ]

track_style_goal_weight =  1.47
style_reward_coef = 10.0

##############  Expressive (Mimic) ########################

#link_name =  ["right_knee_link", "left_elbow_link"]
expressive_link_name = ["left_elbow_link", "right_elbow_link","left_hip_pitch_link","right_hip_pitch_link","left_shoulder_roll_link","right_shoulder_roll_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link","left_wrist_roll_link","right_wrist_roll_link"]
expressive_link_name =  None #["left_elbow_link", "left_hip_pitch_link"] #,"right_hip_pitch_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link"]

# 27 joints robot
upper_joint_name = ["torso_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint","left_shoulder_roll_joint","right_shoulder_roll_joint",  "left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint","right_wrist_yaw_joint", 'left_wrist_roll_joint', 'right_wrist_roll_joint']

lower_joint_name = ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "left_knee_joint","right_knee_joint"]

joint_name = upper_joint_name + lower_joint_name

# not need to include that for normal command (base_velocity command)
expressive_goal_fields = [key+"_dof_pos" for key in joint_name] if joint_name is not None else None
#expressive_goal_fields += [key+"_dof_vel" for key in joint_name] if joint_name is not None else []
expressive_goal_fields += [key1+key2 for key1 in expressive_link_name for key2 in ["_pos_x_b","_pos_y_b","_pos_z_b"]] if expressive_link_name is not None else []
expressive_goal_fields += [key1+key2 for key1 in expressive_link_name for key2 in ["_vel_x_b","_vel_y_b","_vel_z_b"]] if expressive_link_name is not None else []


###################### amp data #################
ref_motion_cfg = RefMotionCfg(
        motion_files=motion_files,
        init_state_fields=INIT_STATE_FIELDS,
        style_goal_fields=style_goal_fields, # as input for the policy
        style_fields=style_fields, # as for style rewards
        expressive_goal_fields = expressive_goal_fields, # only as input for the policy
        expressive_joint_name = joint_name, # for tracking rewards
        expressive_link_name = expressive_link_name, # for tracking rewards
        time_between_frames=0.02,  # time between two frames of state and next_state
        shuffle=False, #shuffle different trajectories
        random_start=random_start,
        amp_obs_frame_num=amp_obs_frame_num, #-1+1,
        augment_frame_num=1000, # episode length, NOTE, keep it equal to self.env.episode_length_s/(self.env.dt*self.cfg.decimation)
        augment_trajectory_num=num_envs,
        frame_begin=10,
        frame_end=None,
)



