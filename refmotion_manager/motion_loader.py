import os
import glob
import json
import joblib
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from .utils import motion_util
from .utils.pose3d import QuaternionNormalize

from dataclasses import MISSING


# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger("motion_loader")

from isaaclab.utils import configclass

@configclass
class RefMotionCfg:
    """Configuration for the AMP networks."""
    time_between_frames: float=0.1  # time between two frames
    shuffle:bool=False
    motion_files: str= MISSING
    
    init_state_fields: list= MISSING 
    style_goal_fields: list= None
    style_fields: list= MISSING 
    device: str = "cuda:0"
    trajectory_num: int=100
    ref_length_s: float= 20
    frame_begin: int=0
    frame_end: int=100
    expressive_goal_fields:list=MISSING
    expressive_joint_name:list=MISSING
    expressive_link_name:list=MISSING
    random_start: bool=False
    amp_obs_frame_num: int=1
    specify_init_values=None



class RefMotionLoader:
    def __init__(
        self, 
        cfg: RefMotionCfg,
        **kwargs,
    ):
        """
        Initialize the AMPLoader.

        Args:
            time_between_frames: Amount of time in seconds between transition.
            data_dir: Directory containing motion data.
            num_preload_transitions: Number of transitions to preload.
            motion_files: List of motion files.
            device: Device to run the model on (e.g., 'cpu' or 'cuda').
        """

        #logger.info(f"[Datset Info] Motion files: \n{motion_files}")
        self.cfg = cfg
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_durations = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []  # the duration of a frame
        self.trajectory_num_frames = []

        # Load motion data from files
        assert len(self.cfg.motion_files) > 0, "No motion files found."
        for file_idx, motion_file in enumerate(self.cfg.motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            logger.info(f"[Dataset Info] start to reading {motion_file}")
            for traj_name, motion_json in joblib.load(motion_file).items(): # a motion file may have more traj
                motion_data = np.array(motion_json["Frames"])
                # motion_data = self.reorder_from_pybullet_to_isaac(motion_data)
                self.trajectory_fields = motion_json["Fields"]

                self.cfg.frame_begin = self.cfg.frame_begin if self.cfg.frame_begin is not None else 0
                self.cfg.frame_end = motion_data.shape[0] if self.cfg.frame_end is None else self.cfg.frame_end
                
                logger.info(f"[Dataset Info] Load {motion_file}")
                logger.info(f"[Dataset Info] It has {motion_data.shape[0]} frames.")
                motion_data = motion_data[self.cfg.frame_begin: self.cfg.frame_end, :]
                logger.info(f"[Dataset Info] Select {motion_data.shape[0]} frames from frame_begin is {self.cfg.frame_begin} and frame_end is {self.cfg.frame_end}.")
                #logger.info(f"[Dataset Info] Re-ordered trajectory fields: {self.style_fields}")

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = motion_data[f_i, [self.trajectory_fields.index("root_rot_" + key) for key in ["x", "y", "z", "w"]]]
                    root_rot = QuaternionNormalize(root_rot)  # unit(normalize) quaternion (x,y,z,w), to make norm to be 1
                    root_rot = motion_util.standardize_quaternion(root_rot)  # standlize quaternion, make w > 0
                    motion_data[f_i, [self.trajectory_fields.index("root_rot_" + key) for key in ["x", "y", "z", "w"]]] = root_rot


                # adding init status transition
                if cfg.specify_init_values is not None:
                    logger.info(f"Specify init values: {cfg.specify_init_values}")
                    first_frame = np.copy(motion_data[0,:])
                    last_frame = np.copy(motion_data[-1,:])
                    init_frame = np.copy(first_frame)
                    head_transition_frames=[]
                    tail_transition_frames=[]
                    for key, value in cfg.specify_init_values.items():
                        init_frame[self.trajectory_fields.index(key)] = value

                    for idx in range(150):
                        blend = float(idx/150.0)
                        head_transition_frames.append(self.blend_frame_pose(init_frame, first_frame, blend))
                        tail_transition_frames.append(self.blend_frame_pose(last_frame, init_frame, blend))

                    head_transition_frames = np.array(head_transition_frames)
                    tail_transition_frames = np.array(tail_transition_frames)
                    motion_data=np.concatenate([head_transition_frames,motion_data, tail_transition_frames], axis=0)
                    logger.info(f"[Dataset Info] Adding specify init and end  300 frames, so motion data shape: {motion_data.shape[0]}.")

                self.trajectories.append(torch.tensor(motion_data, dtype=torch.float32, device=cfg.device))

                self.trajectory_idxs.append(file_idx)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_durations.append(traj_len)  # it is time [s], frame len* duration
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

                logger.info(f"[Dataset Info] Select {traj_len} s motion from the trajectory.")
                #logger.info(f"[Dataset Info] The trajectory fields are: {self.trajectory_fields}")

        # Normalize trajectory weights
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(
            self.trajectory_weights
        )
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_durations = np.array(self.trajectory_durations)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        self.root_rot_index = [
            self.cfg.style_fields.index("root_rot_" + key) for key in ["x", "y", "z", "w"]
        ]

        # Preload transitions if specified
        logger.info(f"[Dataset Info] Preloading data into self.preloaded_s")

        if self.cfg.ref_length_s is None:
            self.cfg.ref_length_s = float(min(self.trajectory_durations))

        self.augment_frame_num = int(min(int(self.cfg.ref_length_s/self.cfg.time_between_frames), min(self.trajectory_num_frames) - self.cfg.amp_obs_frame_num-1))
        assert self.augment_frame_num <= min(self.trajectory_num_frames), f"required frame num  {self.augment_frame_num} should less than the loaded trajtectory frame number {self.trajectory_num_frames}"

        if cfg.trajectory_num is None:
            self.cfg.trajectory_num = self.num_motions
        else:
            self.cfg.trajectory_num = cfg.trajectory_num

        # Define the field names for the dataset.
        ## Loading preloaded dataset
        traj_idxs = self.weighted_traj_idx_sample_batch(size=self.cfg.trajectory_num)

        # Preallocate based on expected sizes
        B = self.cfg.trajectory_num
        T = self.augment_frame_num + self.cfg.amp_obs_frame_num
        D = self.get_frame_at_time(0, 0).shape[0]  # Assuming all frames same shape
        
        preloaded_s = torch.empty((B, T, D), dtype=torch.float32, device=self.cfg.device)
        for i, traj_idx in enumerate(traj_idxs):
            times = self.traj_time_sample_batch(traj_idx, size=T)
            for j, frame_time in enumerate(times):
                preloaded_s[i, j] = self.get_frame_at_time(traj_idx, frame_time)
        
        self.preloaded_s = preloaded_s.requires_grad_(False)

        # Select init state and necessary fields
        assert set(self.cfg.init_state_fields).issubset(set(self.trajectory_fields)), f"The arguments field {self.cfg.init_state_fields} should have same elements with the dataset fields {self.trajectory_fields} in {motion_file}"
        self.init_state_fields_index = [self.trajectory_fields.index(key) for key in self.cfg.init_state_fields]
        self.base_velocity_index_w = [self.trajectory_fields.index(key) for key in ["root_vel_x_w","root_vel_y_w","root_ang_vel_z_w"]]
        self.base_velocity_index_b = [self.trajectory_fields.index(key) for key in ["root_vel_x_b","root_vel_y_b","root_ang_vel_z_b"]]
        self.base_velocity_index = [self.trajectory_fields.index(key) for key in ["root_vel_x_b","root_vel_y_b","root_ang_vel_z_w"]]
        
        # Select root velocity
        if cfg.style_goal_fields is not None:
            self.style_goal_index = [self.trajectory_fields.index(key) for key in cfg.style_goal_fields]

        if cfg.expressive_goal_fields is not None:
            self.expressive_goal_index = [self.trajectory_fields.index(key) for key in cfg.expressive_goal_fields]

        self.root_pos_index = [self.trajectory_fields.index(key) for key in ["root_pos_x","root_pos_y","root_pos_z"]] 
        self.root_quat_index = [self.trajectory_fields.index(key) for key in ["root_rot_w","root_rot_x","root_rot_y","root_rot_z"]] 
        self.root_lin_vel_index_w = [self.trajectory_fields.index(key) for key in ["root_vel_x_w","root_vel_y_w","root_vel_z_w"]] 
        self.root_ang_vel_index_w = [self.trajectory_fields.index(key) for key in ["root_ang_vel_x_w","root_ang_vel_y_w","root_ang_vel_z_w"]] 

        self.root_lin_vel_index_b = [self.trajectory_fields.index(key) for key in ["root_vel_x_b","root_vel_y_b","root_vel_z_b"]] 
        self.root_ang_vel_index_b = [self.trajectory_fields.index(key) for key in ["root_ang_vel_x_b","root_ang_vel_y_b","root_ang_vel_z_b"]] 

        # Select fields
        #logger.info(f"[Dataset Info] The SELECTED trajectory fields are: {self.style_fields}")
        assert set(self.cfg.style_fields).issubset(set(self.trajectory_fields)), f"The arguments field {self.cfg.style_fields} should have same elements with the dataset fields {self.trajectory_fields} in {motion_file}"
        self.style_field_index = [self.trajectory_fields.index(key) for key in self.cfg.style_fields]
        self.selected_preloaded_s = self.preloaded_s[:,:,self.style_field_index]

        self.frame_idx = torch.zeros(self.cfg.trajectory_num).to(self.cfg.device).to(torch.long).requires_grad_(False) # num_env/num_traj frame_idx
        self.start_idx = torch.zeros(self.cfg.trajectory_num).to(self.cfg.device).to(torch.long).requires_grad_(False) # num_env/num_traj frame_idx
        self.traj_idxs = torch.arange(self.preloaded_s.shape[0], device=self.preloaded_s.device).requires_grad_(False)
    
        logger.info(f"[Dataset Info] self.preloaded_s dim are {len(self.preloaded_s)} and {self.preloaded_s[0].shape} in which augment_frame_num: {self.augment_frame_num} and amp_obs_history_leng: {self.cfg.amp_obs_frame_num}")
        logger.info(f"[Dataset Info] Trajectory number: {len(self.trajectory_idxs)}")

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size: int):
        """Batch sample traj idxs."""
        if self.cfg.shuffle:
            return np.random.choice(
                self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True
            )
        else:
            repeated_array = np.tile(
                self.trajectory_idxs, (size // len(self.trajectory_idxs)) + 1
            )
            return repeated_array[:size]


    def traj_time_sample_batch(self, traj_idx, size):
        """Sample random time for multiple trajectories."""
        subst = (self.cfg.time_between_frames + self.trajectory_frame_durations[traj_idx])  # frame period
        if self.cfg.shuffle:
            time_samples = (self.trajectory_durations[traj_idx] * np.random.uniform(size=int(size))- subst)
        else:
            if self.cfg.random_start:
                #NOTE: data frame num must be large than 2*num_steps_per_env
                random_start_threshold = 1-size/(1+self.trajectories[traj_idx].shape[0])
                start_time = np.random.uniform(high = random_start_threshold, size=1)*self.trajectory_durations[traj_idx]
            else:
                start_time = 0.0
            time_samples = start_time + np.arange(0, size * self.cfg.time_between_frames, self.cfg.time_between_frames)
            time_samples = np.minimum(time_samples, self.trajectory_durations[traj_idx]-subst)
        return time_samples

    def slerp(self, val0, val1, blend):
        """Spherical linear interpolation."""
        return (1.0 - blend) * val0 + blend * val1


    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = (float(time) / self.trajectory_durations[traj_idx])  # percentage of the time on the trajectory
        n = self.trajectories[traj_idx].shape[0]  # number of traj_idx trajectory
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_begin = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        # return self.slerp(frame_begin, frame_end, blend)
        return self.blend_frame_pose(frame_begin, frame_end, blend)


    def blend_frame_pose(self, frame0, frame1, blend):
        """
        Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between the two frames.

        Returns:
            An interpolation of the two frames.

        """
        if "root_rot" not in self.trajectory_fields:
            blend_fd = self.slerp(frame0, frame1, blend)
        else:
            # Define the field names for the dataset.
            root_rot_index = [self.trajectory_fields.index("root_rot_" + key) for key in ["x", "y", "z", "w"]]
            other_field_index = [
                i for i in range(len(self.trajectory_fields)) if i not in root_rot_index
            ]

            fd0 = frame0[root_rot_index]
            fd1 = frame1[root_rot_index]
            rot_blend_fd = transformations.quaternion_slerp(
                fd0.cpu().numpy(), fd1.cpu().numpy(), blend
            )
            rot_blend_fd = motion_util.standardize_quaternion(rot_blend_fd)

            other_blend_fd = self.slerp(
                frame0[other_field_index], frame1[other_field_index], blend
            )

            blend_fd = frame0.clone()
            blend_fd[root_rot_index] = rot_blend_fd
            blend_fd[other_field_index] = other_blend_fd

            # blend_fd = torch.cat((other_blend_fd, rot_blend_fd), dim=-1, dtype=torch.float32, device=self.device)
        return blend_fd

    def expert_obs_generator(self):
        """Generates a batch of AMP transitions."""
        if self.cfg.shuffle:
            traj_idxs = np.random.choice(self.selected_preloaded_s.shape[0], size=self.selected_preloaded_s.shape[0])  # sample random sequence of frame
        else:
            traj_idxs = np.arange(self.selected_preloaded_s.shape[0])

        self.frame_idx=0
        start_idx = np.random.randint(self.preloaded_s.shape[1]-self.augment_frame_num-2+1)
        self.init_states = self.preloaded_s[:, start_idx, self.init_state_fields_index] # amp_obs_frame_num is the start frame
        for idx in range(self.augment_frame_num):
            self.frame_idx = idx
            frame_idx = start_idx + idx
            amp_expert = self.selected_preloaded_s[traj_idxs, frame_idx:frame_idx+self.cfg.amp_obs_frame_num, :]  # differet trajectories and frame times
            
            # get base velocity
            self.base_velocity_w = self.preloaded_s[:,:,self.base_velocity_index_w][traj_idxs, frame_idx+1,:]
            self.base_velocity_b = self.preloaded_s[:,:,self.base_velocity_index_b][traj_idxs, frame_idx+1,:]
            self.base_velocity = self.preloaded_s[:,:,self.base_velocity_index][traj_idxs, frame_idx+1,:]
            # get goal of style, current and next frames
            if self.cfg.style_goal_fields is not None:
                self.style_goal = self.preloaded_s[:,:,self.style_goal_index][traj_idxs, frame_idx+1,:]

            # expressive goal states of current and next frames
            if self.cfg.expressive_goal_fields is not None:
                self.expressive_goal = self.preloaded_s[:,:,self.expressive_goal_index][traj_idxs, frame_idx+1,:]

            # expressive states for calculating rewards , using the current frame
            if self.cfg.expressive_joint_name is not None:
                self.expressive_joint_pos = self.preloaded_s[:,:,self.expressive_joint_pos_index][traj_idxs, frame_idx,:]
                self.expressive_joint_vel = self.preloaded_s[:,:,self.expressive_joint_vel_index][traj_idxs, frame_idx,:]
            if self.cfg.expressive_link_name is not None:
                self.expressive_link_pos_w = self.preloaded_s[:,:,self.expressive_link_pos_index_w][traj_idxs, frame_idx,:].reshape(self.cfg.trajectory_num,-1,3)
                self.expressive_link_vel_w = self.preloaded_s[:,:,self.expressive_link_vel_index_w][traj_idxs, frame_idx,:].reshape(self.cfg.trajectory_num,-1,3)
                self.expressive_link_pos_b = self.preloaded_s[:,:,self.expressive_link_pos_index_b][traj_idxs, frame_idx,:].reshape(self.cfg.trajectory_num,-1,3)
                self.expressive_link_vel_b = self.preloaded_s[:,:,self.expressive_link_vel_index_b][traj_idxs, frame_idx,:].reshape(self.cfg.trajectory_num,-1,3)

            self.root_pos_w = self.preloaded_s[:,:,self.root_pos_index][traj_idxs, frame_idx,:]
            self.root_quat_w = self.preloaded_s[:,:,self.root_quat_index][traj_idxs, frame_idx,:]
            self.root_lin_vel_w = self.preloaded_s[:,:,self.root_lin_vel_index_w][traj_idxs, frame_idx,:]
            self.root_ang_vel_w = self.preloaded_s[:,:,self.root_ang_vel_index_w][traj_idxs, frame_idx,:]
            self.root_lin_vel_b = self.preloaded_s[:,:,self.root_lin_vel_index_b][traj_idxs, frame_idx,:]
            self.root_ang_vel_b = self.preloaded_s[:,:,self.root_ang_vel_index_b][traj_idxs, frame_idx,:]

            #yield s_f.reshape(s_f.shape[0],-1)
            yield amp_expert.flatten(1,2)



    def step(self):
        """Generates a batch of AMP transitions."""
        with torch.no_grad():
            # Define trajectory indices and frame positions
            self.traj_idxs = torch.arange(self.preloaded_s.shape[0], device=self.preloaded_s.device)
            self.abs_frame_idx = self.start_idx + self.frame_idx
    
            # I) Initial state
            self.init_states = self.preloaded_s[self.traj_idxs, self.start_idx, :][:, self.init_state_fields_index]
    
            # II) AMP observation
            amp_seq = [
                self.preloaded_s[
                    self.traj_idxs[i], 
                    self.abs_frame_idx[i] : self.abs_frame_idx[i] + self.cfg.amp_obs_frame_num, 
                    self.style_field_index
                ].reshape(1, -1) 
                for i in range(len(self.traj_idxs))
            ]
            self.amp_expert = torch.cat(amp_seq, dim=0)
    
            # III) Goal (next frame data)
            self.next_frame_idx = self.abs_frame_idx + 1
    
            # Increment frame counter
            self.frame_idx += 1

    # all data
    @property
    def data(self):
        """Root position in world frame at current frame"""
        return self.preloaded_s[self.traj_idxs, self.abs_frame_idx]

    # Root state properties
    @property
    def root_pos_w(self):
        """Root position in world frame at current frame"""
        if not hasattr(self, 'traj_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_pos_w")
        return self.preloaded_s[self.traj_idxs, self.abs_frame_idx][:, self.root_pos_index]
    
    @property
    def root_quat_w(self):
        """Root quaternion in world frame at current frame"""
        if not hasattr(self, 'traj_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_quat_w")
        return self.preloaded_s[self.traj_idxs, self.abs_frame_idx][:, self.root_quat_index]
    
    @property
    def root_lin_vel_w(self):
        """Root linear velocity in world frame at current frame"""
        if not hasattr(self, 'traj_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_lin_vel_w")
        return self.preloaded_s[self.traj_idxs, self.abs_frame_idx][:, self.root_lin_vel_index_w]
    
    @property
    def root_ang_vel_w(self):
        """Root angular velocity in world frame at current frame"""
        if not hasattr(self, 'traj_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_ang_vel_w")
        return self.preloaded_s[self.traj_idxs, self.abs_frame_idx][:, self.root_ang_vel_index_w]
    
    @property
    def root_lin_vel_b(self):
        """Root linear velocity in body frame at current frame"""
        if not hasattr(self, 'traj_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_lin_vel_b")
        return self.preloaded_s[self.traj_idxs, self.abs_frame_idx][:, self.root_lin_vel_index_b]
    
    @property
    def root_ang_vel_b(self):
        """Root angular velocity in body frame at current frame"""
        if not hasattr(self, 'traj_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_ang_vel_b")
        return self.preloaded_s[self.traj_idxs, self.abs_frame_idx][:, self.root_ang_vel_index_b]
    
    # Goal/Target properties (next frame data)
    @property
    def base_velocity_w(self):
        """Base velocity in world frame for next frame (goal)"""
        return self.preloaded_s[self.traj_idxs, self.next_frame_idx][:, self.base_velocity_index_w]
    
    @property
    def base_velocity_b(self):
        """Base velocity in body frame for next frame (goal)"""
        return self.preloaded_s[self.traj_idxs, self.next_frame_idx][:, self.base_velocity_index_b]

    @property
    def base_velocity(self):
        """Base velocity in body frame for next frame (goal)"""
        return self.preloaded_s[self.traj_idxs, self.next_frame_idx][:, self.base_velocity_index]
    
    @property
    def style_goal(self):
        """Style goal for next frame"""
        if self.cfg.style_goal_fields is None:
            return None
        return self.preloaded_s[self.traj_idxs, self.next_frame_idx][:, self.style_goal_index]
    
    @property
    def expressive_goal(self):
        """Expressive goal for next frame"""
        if self.cfg.expressive_goal_fields is None:
            return None
        return self.preloaded_s[self.traj_idxs, self.next_frame_idx][:, self.expressive_goal_index]





    def reset(self, env_ids:torch.Tensor=None):
        if env_ids==None:
            self.frame_idx[:] = 0
            self.start_idx[:] = np.random.randint(self.preloaded_s.shape[1] - self.augment_frame_num-2+1)
        else:
            self.frame_idx[env_ids] = 0
            self.start_idx[env_ids] = np.random.randint(self.preloaded_s.shape[1] - self.augment_frame_num-2+1)

    def sw_quat(self, frame_data):
        # switch root-rot quaternion from xyzw to wxyz
        if self.switch_quaternion:
            root_rot = frame_data[self.root_rot_index]
            root_rot = torch.cat((root_rot[-1:], root_rot[:-1]))
            frame_data[self.root_rot_index] = root_rot
        return frame_data

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

