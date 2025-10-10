import os
import glob
import json
import joblib
import logging
from dataclasses import MISSING
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from pybullet_utils import transformations

from .utils import motion_util
from .utils.pose3d import QuaternionNormalize

try:
    from isaaclab.utils import configclass
except ImportError:
    from dataclasses import dataclass, field
    configclass=dataclass
 
# Configure logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG if you want more
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("motion_loader")

@configclass
class RefMotionCfg:
    """Configuration for the Reference Motion Loader."""
    
    # Required parameters
    motion_files: str = MISSING
    init_state_fields: List[str] = MISSING
    style_fields: List[str] = MISSING
    expressive_goal_fields: List[str] = MISSING
    #expressive_joint_name: List[str] = MISSING
    #expressive_link_name: List[str] = MISSING
    
    # Optional parameters with sensible defaults
    style_goal_fields: Optional[List[str]] = None
    time_between_frames: float = 0.02
    shuffle: bool = False
    device: str = "cuda:0"
    clip_num: int = 1
    trajectory_num: Optional[int] = None # Depresse this parameter in the future, please use clip_num
    ref_length_s: float = 20.0
    frame_begin: int = 0
    frame_end: Optional[int] = None
    random_start: bool = False
    amp_obs_frame_num: int = 2
    specify_init_values: Optional[Dict[str, float]] = None


class RefMotionLoader:
    """
    Load and manage reference motion data for imitation learning.
    
    Features:
    - Multi-trajectory support with proper weighting
    - Frame preprocessing and quaternion normalization
    - Efficient preloading of motion sequences
    - Flexible field selection for different learning objectives
    - Robust error handling and device management
    """
    
    def __init__(self, cfg: RefMotionCfg, **kwargs):
        """
        Initialize the Reference Motion Loader.

        Args:
            cfg: Configuration object containing all necessary parameters
            **kwargs: Additional keyword arguments for future extensibility
        """
        self._validate_config(cfg)
        self.cfg = self._setup_device(cfg)
        self._initialize_data_structures()
        self._load_trajectories()
        self._preload_reference_motions()
        self._setup_field_indices()
        self._init_amp_ref_obs()
        self._log_initialization_summary()

    def _validate_config(self, cfg: RefMotionCfg) -> None:
        """Validate the configuration parameters."""
        if not hasattr(cfg, 'motion_files') or not cfg.motion_files:
            raise ValueError("Configuration must specify motion_files")
        
        if not hasattr(cfg, 'init_state_fields') or not cfg.init_state_fields:
            raise ValueError("Configuration must specify init_state_fields")
            
        if not hasattr(cfg, 'style_fields') or not cfg.style_fields:
            raise ValueError("Configuration must specify style_fields")


    def _setup_device(self, cfg: RefMotionCfg) -> RefMotionCfg:
        """Safely configure the computation device."""
        device_str = str(cfg.device)

        if device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                cfg.device = "cpu"
            else:
                try:
                    _ = torch.tensor([1.0], device=device_str)
                    logger.info(f"Using CUDA device: {device_str}")
                except RuntimeError as e:
                    logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU.")
                    cfg.device = "cpu"
        else:
            cfg.device = "cpu"

        logger.info(f"Data will be loaded on device: {cfg.device}")
        return cfg

    @property
    def torch_device(self) -> torch.device:
        """Return torch.device object for computations."""
        return torch.device(self.cfg.device)


    def _initialize_data_structures(self) -> None:
        """Initialize all data storage structures."""
        # Trajectory metadata
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_durations = []  # Trajectory length in seconds
        self.trajectory_weights = []
        self.trajectory_frame_durations = []  # Duration per frame
        self.trajectory_frame_num = []
        
        # Field information
        self.trajectory_fields = None

    def _load_trajectories(self) -> None:
        """Load and process all motion trajectories."""
        logger.info(f"Loading {len(self.cfg.motion_files)} motion files")
        
        for file_idx, motion_file in enumerate(self.cfg.motion_files):
            logger.info(f"Processing motion file {file_idx + 1}/{len(self.cfg.motion_files)}: {motion_file}")
            self._process_motion_file(motion_file, file_idx)
        
        self._normalize_trajectory_weights()

    def _process_motion_file(self, motion_file: str, file_idx: int) -> None:
        """Process a single motion file containing one or more trajectories."""
        try:
            motion_data = joblib.load(motion_file)
            self.trajectory_names.append(os.path.splitext(motion_file)[0])
            
            for traj_name, motion_json in motion_data.items():
                self._process_single_trajectory(motion_json, motion_file, file_idx)
                
        except Exception as e:
            logger.error(f"Failed to load motion file {motion_file}: {e}")
            raise

    def _process_single_trajectory(self, motion_json: Dict, motion_file: str, file_idx: int) -> None:
        """Process a single trajectory from motion data."""
        # Extract motion data
        motion_data = np.array(motion_json["Frames"])
        self.trajectory_fields = motion_json["Fields"]
        frame_duration = float(motion_json["FrameDuration"])
        
        logger.info(f"Loaded trajectory with {motion_data.shape[0]} frames "
                   f"(duration: {frame_duration:.4f}s per frame)")

        # Apply frame range selection
        motion_data = self._select_frame_range(motion_data)
        
        # Normalize quaternions
        motion_data = self._normalize_quaternions(motion_data)
        
        # Add transition frames if specified
        motion_data = self._add_transition_frames(motion_data, motion_json)
        
        # Convert to tensor and store
        motion_tensor = torch.tensor(motion_data, dtype=torch.float32, device=self.cfg.device)
            
        self.trajectories.append(motion_tensor)
        self._store_trajectory_metadata(motion_json, motion_tensor, file_idx, frame_duration)

    def _select_frame_range(self, motion_data: np.ndarray) -> np.ndarray:
        """Select specified frame range from motion data."""
        frame_begin = self.cfg.frame_begin if self.cfg.frame_begin is not None else 0
        frame_end = self.cfg.frame_end if self.cfg.frame_end is not None else motion_data.shape[0]
        
        if frame_begin >= frame_end:
            raise ValueError(f"Invalid frame range: begin={frame_begin}, end={frame_end}")
            
        selected_data = motion_data[frame_begin:frame_end, :]
        logger.info(f"Selected frames {frame_begin} to {frame_end} "
                   f"({selected_data.shape[0]} frames total)")
        
        return selected_data

    def _normalize_quaternions(self, motion_data: np.ndarray) -> np.ndarray:
        """Normalize and standardize root rotation quaternions."""
        if self.trajectory_fields is None:
            return motion_data
            
        quat_indices = [
            self.trajectory_fields.index(f"root_rot_{key}") 
            for key in ["x", "y", "z", "w"]
        ]
        
        for frame_idx in range(motion_data.shape[0]):
            root_rot = motion_data[frame_idx, quat_indices]
            root_rot = QuaternionNormalize(root_rot)  # Normalize to unit quaternion
            root_rot = motion_util.standardize_quaternion(root_rot)  # Standardize (w > 0)
            motion_data[frame_idx, quat_indices] = root_rot
            
        return motion_data

    def _add_transition_frames(self, motion_data: np.ndarray, motion_json: Dict) -> np.ndarray:
        """Add transition frames for initialization if specified."""
        if self.cfg.specify_init_values is None:
            return motion_data
            
        logger.info("Adding transition frames for specified initial values")
        head_tail_time_s = 2.0  # seconds
        frame_duration = float(motion_json["FrameDuration"])
        head_tail_frame_num = int(head_tail_time_s / frame_duration)
        
        first_frame = motion_data[0, :].copy()
        last_frame = motion_data[-1, :].copy()
        init_frame = first_frame.copy()
        
        # Apply initial values
        for key, value in self.cfg.specify_init_values.items():
            init_frame[self.trajectory_fields.index(key)] = value
        
        # Create transition frames
        head_transition_frames = []
        tail_transition_frames = []
        
        for idx in range(head_tail_frame_num):
            blend = float(idx / head_tail_frame_num)
            head_transition_frames.append(self._blend_frame_pose(init_frame, first_frame, blend))
            tail_transition_frames.append(self._blend_frame_pose(last_frame, init_frame, blend))
        
        # Combine all frames
        enhanced_data = np.vstack([
            np.array(head_transition_frames),
            motion_data,
            np.array(tail_transition_frames)
        ])
        
        logger.info(f"Added {2 * head_tail_frame_num} transition frames. "
                   f"New shape: {enhanced_data.shape}")
        
        return enhanced_data

    def _store_trajectory_metadata(self, motion_json: Dict, motion_tensor: torch.Tensor, 
                                 file_idx: int, frame_duration: float) -> None:
        """Store metadata for a single trajectory."""
        self.trajectory_idxs.append(file_idx)
        self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))
        self.trajectory_frame_durations.append(frame_duration)
        
        traj_len = (motion_tensor.shape[0]-1) * frame_duration
        self.trajectory_durations.append(traj_len)
        self.trajectory_frame_num.append(motion_tensor.shape[0])

        
        logger.info(f"Trajectory duration: {traj_len:.2f}s")

    def _normalize_trajectory_weights(self) -> None:
        """Normalize trajectory weights to sum to 1."""
        if self.trajectory_weights:
            total_weight = sum(self.trajectory_weights)
            self.trajectory_weights = [w / total_weight for w in self.trajectory_weights]
            

    def _preload_reference_motions(self) -> None:
        """Preload reference motion sequences for efficient sampling."""
        logger.info("Preloading reference motion sequences")
        
        # Calculate reference length
        if self.cfg.ref_length_s is None:
            self.cfg.ref_length_s = float(min(self.trajectory_durations))

        self.cfg.ref_length_s = min(float(min(self.trajectory_durations)), self.cfg.ref_length_s)
            
        self.clip_frame_num = int(self.cfg.ref_length_s / self.cfg.time_between_frames)
        logger.warn("Will depression augment_frame_num in the future, please use clip_frame_num")
        self.augment_frame_num = self.clip_frame_num 
        
        # Set trajectory number
        if self.cfg.clip_num is None:
            self.cfg.clip_num = len(self.trajectories)

        if self.cfg.trajectory_num is not None:
            self.cfg.clip_num = self.cfg.trajectory_num
            logger.warn("Depression trajectory_num in the future, please use clip_num")
            
        logger.info(f"Preloading {self.cfg.clip_num} trajectories with "
                   f"{self.clip_frame_num} frames each")
        
        # Sample trajectories and times
        traj_idxs = self.weighted_traj_idx_sample_batch(size=self.cfg.clip_num)

        
        # Preallocate tensor
        B, T = self.cfg.clip_num, self.clip_frame_num + self.cfg.amp_obs_frame_num
        D = self.get_frame_at_time(0, 0).shape[0] # feilds number
        
        self.preloaded_s = torch.empty((B, T, D), dtype=torch.float32, device=self.cfg.device)
        
        # Fill preloaded tensor
        for i, traj_idx in enumerate(traj_idxs):
            times = self.traj_time_sample_batch(traj_idx, size=T)
            for j, frame_time in enumerate(times):
                self.preloaded_s[i, j] = self.get_frame_at_time(traj_idx, frame_time)
        
        #self.preloaded_s = preloaded_s.requires_grad_(False)


        logger.info(f"Preloaded tensor shape: {self.preloaded_s.shape} "
                   f"on device: {self.preloaded_s.device}")



    def _init_amp_ref_obs(self) -> None:
        self.abs_frame_idx = self.start_idx
        self.amp_expert = self.preloaded_s[
                self.clip_idxs, 
                self.abs_frame_idx][:,self.style_field_index].repeat(1,2)

    def _setup_field_indices(self) -> None:
        """Setup indices for different field types."""
        # Validate field subsets
        self._validate_field_subsets()
        
        # Setup various field indices
        self._setup_velocity_indices()
        self._setup_root_indices()
        self._setup_goal_indices()
        self._setup_style_fields()
        
        # Create selected preloaded tensor
        self.selected_preloaded_s = self.preloaded_s[:, :, self.style_field_index]
        
        # Initialize sampling indices
        self._initialize_sampling_indices()

    def _validate_field_subsets(self) -> None:
        """Validate that specified field subsets exist in trajectory fields."""
        assert set(self.cfg.init_state_fields).issubset(set(self.trajectory_fields)), \
            f"init_state_fields {self.cfg.init_state_fields} not found in trajectory fields"
            
        assert set(self.cfg.style_fields).issubset(set(self.trajectory_fields)), \
            f"style_fields {self.cfg.style_fields} not found in trajectory fields"

    def _setup_velocity_indices(self) -> None:
        """Setup indices for velocity fields."""
        self.init_state_fields_index = [
            self.trajectory_fields.index(key) for key in self.cfg.init_state_fields
        ]
        
        # World frame velocities
        self.base_velocity_index_w = [
            self.trajectory_fields.index(key) 
            for key in ["root_vel_x_w", "root_vel_y_w", "root_ang_vel_z_w"]
        ]
        
        # Body frame velocities  
        self.base_velocity_index_b = [
            self.trajectory_fields.index(key)
            for key in ["root_vel_x_b", "root_vel_y_b", "root_ang_vel_z_b"]
        ]
        
        # Mixed frame velocities
        self.base_velocity_index = [
            self.trajectory_fields.index(key)
            for key in ["root_vel_x_b", "root_vel_y_b", "root_ang_vel_z_w"]
        ]

    def _setup_root_indices(self) -> None:
        """Setup indices for root position and orientation."""
        self.root_pos_index = [
            self.trajectory_fields.index(key) 
            for key in ["root_pos_x", "root_pos_y", "root_pos_z"]
        ]
        
        self.root_quat_index = [
            self.trajectory_fields.index(key)
            for key in ["root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
        ]
        
        # World frame linear and angular velocities
        self.root_lin_vel_index_w = [
            self.trajectory_fields.index(key)
            for key in ["root_vel_x_w", "root_vel_y_w", "root_vel_z_w"]
        ]
        self.root_ang_vel_index_w = [
            self.trajectory_fields.index(key)
            for key in ["root_ang_vel_x_w", "root_ang_vel_y_w", "root_ang_vel_z_w"]
        ]
        
        # Body frame linear and angular velocities
        self.root_lin_vel_index_b = [
            self.trajectory_fields.index(key)
            for key in ["root_vel_x_b", "root_vel_y_b", "root_vel_z_b"]
        ]
        self.root_ang_vel_index_b = [
            self.trajectory_fields.index(key)
            for key in ["root_ang_vel_x_b", "root_ang_vel_y_b", "root_ang_vel_z_b"]
        ]

    def _setup_goal_indices(self) -> None:
        """Setup indices for goal fields."""
        if self.cfg.style_goal_fields is not None:
            self.style_goal_index = torch.tensor([
                self.trajectory_fields.index(key) for key in self.cfg.style_goal_fields
            ]).to(self.cfg.device)
            
        if self.cfg.expressive_goal_fields is not None:
            self.expressive_goal_index = torch.tensor([
                self.trajectory_fields.index(key) for key in self.cfg.expressive_goal_fields
            ]).to(self.cfg.device)

    def _setup_style_fields(self) -> None:
        """Setup indices for style fields."""
        self.style_field_index = [
            self.trajectory_fields.index(key) for key in self.cfg.style_fields
        ]

    def _initialize_sampling_indices(self) -> None:
        """Initialize indices for trajectory sampling."""
        self.frame_idx = torch.zeros(self.cfg.clip_num, 
                                   device=self.cfg.device, 
                                   dtype=torch.long).requires_grad_(False)
        self.start_idx = torch.zeros(self.cfg.clip_num,
                                   device=self.cfg.device,
                                   dtype=torch.long).requires_grad_(False)
        self.clip_idxs = torch.arange(self.preloaded_s.shape[0],
                                    device=self.preloaded_s.device).requires_grad_(False)

    def _log_initialization_summary(self) -> None:
        """Log summary information after initialization."""
        # Convert to numpy arrays for efficient sampling
        self.trajectory_weights = np.array(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_durations = np.array(self.trajectory_durations)
        self.trajectory_frame_num = np.array(self.trajectory_frame_num)

        logger.info("Reference Motion Loader initialization complete")
        logger.info(f"Trajectory frame num: {self.trajectory_frame_num}")
        logger.info(f"Preloaded tensor dimensions: {len(self.preloaded_s)} clips, "
                   f"each with {self.preloaded_s[0].shape} frames")
        logger.info(f"Augment frame number: {self.clip_frame_num}, "
                   f"AMP observation frames: {self.cfg.amp_obs_frame_num}")
        logger.info(f"Total trajectories loaded: {len(self.trajectory_idxs)}")


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


    def _blend_frame_pose(self, frame1: np.ndarray, frame2: np.ndarray, blend: float) -> np.ndarray:
        """Blend between two frames."""
        # Implementation depends on your blending strategy
        return frame1 * (1 - blend) + frame2 * blend

    @property
    def num_motions(self) -> int:
        """Number of loaded motions."""
        return len(self.trajectories)



    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)



    def traj_time_sample_batch(self, traj_idx: int, size: int) -> np.ndarray:
        """
        Sample time points from a trajectory for batch processing.
        
        Args:
            traj_idx: Index of the trajectory to sample from
            size: Number of time samples to generate
            
        Returns:
            Array of sampled time points within the trajectory duration
            
        Raises:
            ValueError: If traj_idx is out of bounds or size is invalid
        """
        # Input validation
        if not 0 <= traj_idx < len(self.trajectories):
            raise ValueError(f"traj_idx {traj_idx} out of range [0, {len(self.trajectories)-1}]")
        
        if size <= 0:
            raise ValueError(f"Sample size must be positive, got {size}")
        
        # Get trajectory metadata
        traj_duration = self.trajectory_durations[traj_idx]
        frame_duration = self.trajectory_frame_durations[traj_idx]
        
        # Calculate safe time range (avoid sampling beyond valid frames)
        safe_time_margin = self.cfg.time_between_frames + frame_duration
        max_safe_time = traj_duration - safe_time_margin
        
        if max_safe_time <= 0:
            raise ValueError(f"Trajectory {traj_idx} too short ({traj_duration:.3f}s) "
                            f"for sampling with frame duration {frame_duration:.3f}s")
        
        if self.cfg.shuffle:
            # Random uniform sampling across the entire trajectory
            time_samples = np.random.uniform(low=0.0, high=max_safe_time, size=size)
        else:
            # Sequential sampling with optional random start
            if self.cfg.random_start:
                # Calculate maximum start time that allows complete sampling
                max_start_time = traj_duration - (size * self.cfg.time_between_frames)
                max_start_time = max(0.0, max_start_time - safe_time_margin)
                
                start_time = np.random.uniform(low=0.0, high=max_start_time)
            else:
                start_time = 0.0
            
            # Generate sequential time points
            time_samples = start_time + np.arange(size) * self.cfg.time_between_frames
            
            # Ensure samples don't exceed safe bounds
            time_samples = np.clip(time_samples, 0.0, max_safe_time)
        
        return time_samples


    def _lerp(self, val0, val1, blend):
        """Spherical linear interpolation."""
        return (1.0 - blend) * val0 + blend * val1


    def get_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        """Get frame at specific time from trajectory"""
        phase = (float(time) / self.trajectory_durations[traj_idx])  # percentage of the time on the trajectory
        n = self.trajectories[traj_idx].shape[0]-1  # frame number of traj_idx trajectory
        idx_low, idx_high = int(np.floor(phase * n)), int(np.ceil(phase * n))
        frame_0 = self.trajectories[traj_idx][idx_low]
        frame_1 = self.trajectories[traj_idx][idx_high]
        blend = phase * n - idx_low
        return self.blend_frame_pose(frame_0, frame_1, blend)


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
            blend_fd = self._lerp(frame0, frame1, blend)
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

            other_blend_fd = self._lerp(
                frame0[other_field_index], frame1[other_field_index], blend
            )

            blend_fd = frame0.clone()
            blend_fd[root_rot_index] = rot_blend_fd
            blend_fd[other_field_index] = other_blend_fd

            # blend_fd = torch.cat((other_blend_fd, rot_blend_fd), dim=-1, dtype=torch.float32, device=self.device)
        return blend_fd

    def step(self):
        """Generates a batch of AMP transitions."""
        # Define trajectory indices and frame positions
        self.abs_frame_idx = self.start_idx + self.frame_idx
        # I) AMP observation
        #try:
        amp_seq = [self.preloaded_s[self.clip_idxs, self.abs_frame_idx+i,:][:,self.style_field_index]  for i in range(self.cfg.amp_obs_frame_num)]
        #except Exception as e:
        #    import pdb;pdb.set_trace()

        # for the first step, the amp_ref, its current states and next states should be same
        amp_seq[1][self.frame_idx==0,:] = amp_seq[0][self.frame_idx==0,:]
        #import pdb;pdb.set_trace()
        
        self.amp_expert = torch.cat(amp_seq, dim=-1)
    
        # II) Goal (next frame data)
        self.next_frame_idx = self.abs_frame_idx + 1
    
        # Increment frame counter
        self.frame_idx += 1

    # all data
    @property
    def data(self):
        """Root position in world frame at current frame"""
        return self.preloaded_s[self.clip_idxs, self.abs_frame_idx]

    # Root state properties
    @property
    def root_pos_w(self):
        """Root position in world frame at current frame"""
        if not hasattr(self, 'clip_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_pos_w")
        return self.preloaded_s[self.clip_idxs, self.abs_frame_idx][:, self.root_pos_index]
    
    @property
    def root_quat_w(self):
        """Root quaternion in world frame at current frame"""
        if not hasattr(self, 'clip_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_quat_w")
        return self.preloaded_s[self.clip_idxs, self.abs_frame_idx][:, self.root_quat_index]
    
    @property
    def root_lin_vel_w(self):
        """Root linear velocity in world frame at current frame"""
        if not hasattr(self, 'clip_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_lin_vel_w")
        return self.preloaded_s[self.clip_idxs, self.abs_frame_idx][:, self.root_lin_vel_index_w]
    
    @property
    def root_ang_vel_w(self):
        """Root angular velocity in world frame at current frame"""
        if not hasattr(self, 'clip_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_ang_vel_w")
        return self.preloaded_s[self.clip_idxs, self.abs_frame_idx][:, self.root_ang_vel_index_w]
    
    @property
    def root_lin_vel_b(self):
        """Root linear velocity in body frame at current frame"""
        if not hasattr(self, 'clip_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_lin_vel_b")
        return self.preloaded_s[self.clip_idxs, self.abs_frame_idx][:, self.root_lin_vel_index_b]
    
    @property
    def root_ang_vel_b(self):
        """Root angular velocity in body frame at current frame"""
        if not hasattr(self, 'clip_idxs') or not hasattr(self, 'abs_frame_idx'):
            raise RuntimeError("Must call step() before accessing root_ang_vel_b")
        return self.preloaded_s[self.clip_idxs, self.abs_frame_idx][:, self.root_ang_vel_index_b]
    
    # Goal/Target properties (next frame data)
    @property
    def base_velocity_w(self):
        """Base velocity in world frame for next frame (goal)"""
        return self.preloaded_s[self.clip_idxs, self.next_frame_idx][:, self.base_velocity_index_w]
    
    @property
    def base_velocity_b(self):
        """Base velocity in body frame for next frame (goal)"""
        return self.preloaded_s[self.clip_idxs, self.next_frame_idx][:, self.base_velocity_index_b]

    @property
    def base_velocity(self):
        """Base velocity in body frame for next frame (goal)"""
        return self.preloaded_s[self.clip_idxs, self.next_frame_idx][:, self.base_velocity_index]
    
    @property
    def style_goal(self):
        """Style goal for next frame"""
        if self.cfg.style_goal_fields is None:
            return None
        return self.preloaded_s[self.clip_idxs, self.next_frame_idx][:, self.style_goal_index]
    
    @property
    def expressive_goal(self):
        """Expressive goal for next frame"""
        if self.cfg.expressive_goal_fields is None:
            return None
        return self.preloaded_s[self.clip_idxs, self.next_frame_idx][:, self.expressive_goal_index]


    def reset(self, env_ids: torch.Tensor = None):
        #max_start = self.preloaded_s.shape[1] - self.clip_frame_num - 1  # avoid -2+1 confusion
    
        if env_ids is None:
            self.frame_idx[:] = 0
            ##self.start_idx[:] = torch.randint(low=0, high=max_start + 1, size=(self.preloaded_s.shape[0],),device=self.start_idx.device)

            ## weighted sampling of traj indices
            #if not hasattr(self, "traj_weights"):
            #    self.traj_weights = torch.ones(self.preloaded_s.shape[0], device=self.preloaded_s.device)
            #
            #probs = self.traj_weights / self.traj_weights.sum()
            #self.clip_idxs[:] = torch.multinomial(
            #    probs, num_samples=self.preloaded_s.shape[0], replacement=True
            #)
        else:
            self.frame_idx[env_ids] = 0
            # NOTE, CHECKING THIS LATER
            #try:
            #    if env_ids.dtype == torch.bool:
            #        num_ids = env_ids.sum().item()  # how many True entries
            #    else:
            #        num_ids = env_ids.shape[0]

            #    self.start_idx[env_ids] = torch.randint(low=0, high=max_start + 1, size=(num_ids,), device=self.start_idx.device)
            #    import pdb;pdb.set_trace()
            #except RuntimeError as e:
            #    import pdb;pdb.set_trace()
            #
            ## weighted sampling only for selected envs
            #probs = self.traj_weights / self.traj_weights.sum()
            #self.clip_idxs[env_ids] = torch.multinomial(
            #    probs, num_samples=num_ids, replacement=True
            #)

        # Initial state
        self.init_states = self.preloaded_s[self.clip_idxs, self.start_idx, :][:, self.init_state_fields_index]

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

