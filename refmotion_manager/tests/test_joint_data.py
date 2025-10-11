import numpy as np
import matplotlib.pyplot as plt
import os, json, glob
import torch
from scipy.signal import savgol_filter
from refmotion_manager.motion_loader import RefMotionLoader
import seaborn as sns
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

class JointDataVisualizer:
    def __init__(self, ref_motion_loader, ref_motion_cfg, joint_limits=None):
        """
        Initialize visualizer with motion data and joint limits
        
        Args:
            ref_motion_loader: RefMotionLoader instance
            ref_motion_cfg: Motion configuration
            joint_limits: Dict with joint position and velocity limits
                         Format: {'joint_name': {'pos_limit': [min, max], 'vel_limit': max_vel}}
        """
        self.ref_motion = ref_motion_loader
        self.cfg = ref_motion_cfg
        self.trajectory_fields = ref_motion_loader.trajectory_fields
        self.joint_limits = joint_limits or {}
        
        # Create figures directory
        self.figs_dir = self._create_figs_directory()
        
        # Set seaborn style
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")

    def _create_figs_directory(self):
        """Create figures directory under data folder"""
        # Assuming motion files are in a data directory structure
        if self.cfg.motion_files:
            data_dir = os.path.dirname(self.cfg.motion_files[0])
            figs_dir = os.path.join(data_dir, "figs")
            os.makedirs(figs_dir, exist_ok=True)
            return figs_dir
        return "figs"

    def extract_joint_fields(self):
        """Extract all joint position and velocity fields"""
        joint_pos_fields = []
        joint_vel_fields = []
        
        for field in self.trajectory_fields:
            if 'joint' in field.lower() and 'pos' in field.lower():
                joint_pos_fields.append(field)
            elif 'joint' in field.lower() and 'vel' in field.lower():
                joint_vel_fields.append(field)
        
        # Fallback naming conventions
        if not joint_pos_fields:
            joint_pos_fields = [f for f in self.trajectory_fields if 'pos' in f.lower() and any(joint in f.lower() for joint in ['hip', 'knee', 'ankle', 'shoulder', 'elbow'])]
        if not joint_vel_fields:
            joint_vel_fields = [f for f in self.trajectory_fields if 'vel' in f.lower() and any(joint in f.lower() for joint in ['hip', 'knee', 'ankle', 'shoulder', 'elbow'])]
        
        return joint_pos_fields, joint_vel_fields

    def _get_joint_limits(self, joint_name):
        """Get joint limits for a specific joint"""
        if joint_name in self.joint_limits:
            return self.joint_limits[joint_name]
        else:
            # Default limits if not specified
            return {'pos_limit': [-3.14, 3.14], 'vel_limit': 10.0}

    def plot_all_joints_subplots(self, frame_start=0, frame_end=None, traj_idx=0):
        """
        Plot all joint positions and velocities in organized subplots with joint limits
        """
        joint_pos_fields, joint_vel_fields = self.extract_joint_fields()
        
        if frame_end is None:
            frame_end = len(self.ref_motion.trajectories[traj_idx])
        frame_end = min(frame_end, len(self.ref_motion.trajectories[traj_idx]))
        
        print(f"Found {len(joint_pos_fields)} joint position fields")
        print(f"Found {len(joint_vel_fields)} joint velocity fields")
        
        # Get data indices and extract data
        pos_indices = [self.trajectory_fields.index(field) for field in joint_pos_fields]
        vel_indices = [self.trajectory_fields.index(field) for field in joint_vel_fields]
        
        pos_data = self.ref_motion.trajectories[traj_idx][frame_start:frame_end, pos_indices].cpu().numpy()
        vel_data = self.ref_motion.trajectories[traj_idx][frame_start:frame_end, vel_indices].cpu().numpy()
        
        # Use correct frame duration from ref_motion
        frame_duration = self.ref_motion.trajectory_frame_durations[traj_idx]
        time = np.arange(frame_start, frame_end) * frame_duration
        
        # Create subplots for positions
        n_joints = len(joint_pos_fields)
        n_cols = 3  # Number of columns in subplot grid
        n_rows = (n_joints + n_cols - 1) // n_cols  # Calculate rows needed
        
        fig_pos, axes_pos = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), sharex=True)
        axes_pos = axes_pos.flatten() if n_joints > 1 else [axes_pos]
        
        # Plot joint positions
        for i, (ax, pos_field) in enumerate(zip(axes_pos, joint_pos_fields)):
            if i >= n_joints:
                ax.set_visible(False)
                continue
            
            joint_name = pos_field[:-8]
            pos_limits = self._get_joint_limits(joint_name)['pos_limit']
            
            
            # Create DataFrame for seaborn
            plot_df = pd.DataFrame({'Time': time, 'Position': pos_data[:, i]})
            
            # Use seaborn for plotting
            sns.lineplot(data=plot_df, x='Time', y='Position', ax=ax, linewidth=2)
            
            # Add position limits
            ax.axhline(y=pos_limits[0], color='r', linestyle='--', alpha=0.7, 
                      label=f'Min: {pos_limits[0]:.2f}')
            ax.axhline(y=pos_limits[1], color='r', linestyle='--', alpha=0.7, 
                      label=f'Max: {pos_limits[1]:.2f}')
            ax.fill_between(time, pos_limits[0], pos_limits[1], alpha=0.1, color='red')
            
            ax.set_title(pos_field, fontsize=12)
            ax.set_ylabel('Position (rad)', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_joints, len(axes_pos)):
            axes_pos[i].set_visible(False)
        
        fig_pos.suptitle('Joint Positions with Limits', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save position plot
        pos_save_path = os.path.join(self.figs_dir, "all_joints_positions_subplots.png")
        plt.savefig(pos_save_path, dpi=300, bbox_inches='tight')
        
        # Create subplots for velocities
        fig_vel, axes_vel = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), sharex=True)
        axes_vel = axes_vel.flatten() if n_joints > 1 else [axes_vel]
        
        # Plot joint velocities
        for i, (ax, vel_field) in enumerate(zip(axes_vel, joint_vel_fields)):
            if i >= n_joints:
                ax.set_visible(False)
                continue
                
            joint_name = vel_field[:-8]
            vel_limit = self._get_joint_limits(joint_name)['vel_limit']
            
            # Create DataFrame for seaborn
            plot_df = pd.DataFrame({'Time': time, 'Velocity': vel_data[:, i]})
            
            # Use seaborn for plotting
            sns.lineplot(data=plot_df, x='Time', y='Velocity', ax=ax, linewidth=2)
            
            # Add velocity limits
            ax.axhline(y=vel_limit, color='r', linestyle='--', alpha=0.7, 
                      label=f'Max: {vel_limit:.2f}')
            ax.axhline(y=-vel_limit, color='r', linestyle='--', alpha=0.7, 
                      label=f'Min: {-vel_limit:.2f}')
            ax.fill_between(time, -vel_limit, vel_limit, alpha=0.1, color='red')
            
            ax.set_title(vel_field, fontsize=12)
            ax.set_ylabel('Velocity (rad/s)', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_joints, len(axes_vel)):
            axes_vel[i].set_visible(False)
        
        fig_vel.suptitle('Joint Velocities with Limits', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save velocity plot
        vel_save_path = os.path.join(self.figs_dir, "all_joints_velocities_subplots.png")
        plt.savefig(vel_save_path, dpi=300, bbox_inches='tight')
        
        print(f"Saved position plot: {pos_save_path}")
        print(f"Saved velocity plot: {vel_save_path}")
        
        return pos_data, vel_data, joint_pos_fields, joint_vel_fields


    def plot_joint_statistics_seaborn(self, traj_idx=0):
        """Plot joint statistics using seaborn"""
        joint_pos_fields, joint_vel_fields = self.extract_joint_fields()
        
        pos_indices = [self.trajectory_fields.index(field) for field in joint_pos_fields]
        vel_indices = [self.trajectory_fields.index(field) for field in joint_vel_fields]
        
        pos_data = self.ref_motion.trajectories[traj_idx][:, pos_indices].cpu().numpy()
        vel_data = self.ref_motion.trajectories[traj_idx][:, vel_indices].cpu().numpy()
        
        # Create DataFrames for seaborn
        pos_stats = []
        vel_stats = []
        
        for i, field in enumerate(joint_pos_fields):
            joint_name = field[:-8]
            pos_limits = self._get_joint_limits(joint_name)['pos_limit']
            pos_stats.append({
                'Joint': field,
                'Mean': np.mean(pos_data[:, i]),
                'Std': np.std(pos_data[:, i]),
                'Min': np.min(pos_data[:, i]),
                'Max': np.max(pos_data[:, i]),
                'Range': np.ptp(pos_data[:, i]),
                'Limit_Min': pos_limits[0],
                'Limit_Max': pos_limits[1]
            })
            
        for i, field in enumerate(joint_vel_fields):
            joint_name = field[:-8]
            vel_limit = self._get_joint_limits(joint_name)['vel_limit']
            vel_stats.append({
                'Joint': field,
                'Mean': np.mean(vel_data[:, i]),
                'Std': np.std(vel_data[:, i]),
                'Min': np.min(vel_data[:, i]),
                'Max': np.max(vel_data[:, i]),
                'Range': np.ptp(vel_data[:, i]),
                'Limit_Max': vel_limit,
                'Limit_Min': -vel_limit
            })

        # Compact formatted output
        print("\n" + "="*100)
        print("JOINT STATISTICS SUMMARY")
        print("="*100)
        print(f"{'Joint':<20} {'Pos Min':<12} {'Pos Max':<12} {'Pos Min_Limit':<12}  {'Pos Max_limit':<12} | {'Vel Min':<12} {'Vel Max':<12} {'Vel Limit':<12} {'Status':<10}")
        print("-"*100)

        # Create a mapping for quick lookup
        pos_dict = {stat['Joint'].replace('_joint_dof_pos', ''): stat for stat in pos_stats}
        vel_dict = {stat['Joint'].replace('_joint_dof_vel', ''): stat for stat in vel_stats}

        all_joints = set(pos_dict.keys()) | set(vel_dict.keys())

        for joint in sorted(all_joints):
            pos_stat = pos_dict.get(joint)
            vel_stat = vel_dict.get(joint)

            pos_str = f"{pos_stat['Mean']:.2f}±{pos_stat['Std']:.2f}" if pos_stat else "N/A"
            pos_range = f"{pos_stat['Range']:.2f}" if pos_stat else "N/A"
            pos_max = f"{pos_stat['Max']:.2f}" if pos_stat else "N/A"
            pos_min = f"{pos_stat['Min']:.2f}" if pos_stat else "N/A"
            pos_max_limit = f"{pos_stat['Limit_Max']:.2f}" if pos_stat else "N/A"
            pos_min_limit = f"{pos_stat['Limit_Min']:.2f}" if pos_stat else "N/A"
            vel_str = f"{vel_stat['Mean']:.2f}±{vel_stat['Std']:.2f}" if vel_stat else "N/A"
            vel_range = f"{vel_stat['Range']:.2f}" if vel_stat else "N/A"
            vel_max = f"{vel_stat['Max']:.2f}" if vel_stat else "N/A"
            vel_min = f"{vel_stat['Min']:.2f}" if vel_stat else "N/A"
            vel_limit = f"{vel_stat['Limit_Max']:.2f}" if vel_stat else "N/A"

            # Check status
            status = "OK"
            if pos_stat and (pos_stat['Min'] < pos_stat['Limit_Min'] or pos_stat['Max'] > pos_stat['Limit_Max']):
                status = "POS LIMIT"
            elif vel_stat and (vel_stat['Min'] < vel_stat['Limit_Min'] or vel_stat['Max'] > vel_stat['Limit_Max']):
                status = "VEL LIMIT"

            print(f"{joint:<20} {pos_min:<12} {pos_max:<12} {pos_min_limit:<12} {pos_max_limit:<12} | {vel_min:<12} {vel_max:<12} {vel_limit:<12} {status:<10}")


        # Calculate trajectory plausibility
        plausibility_score, violation_details = calculate_trajectory_plausibility(pos_stats, vel_stats)
        rating, emoji = get_plausibility_rating(plausibility_score)

        # Print trajectory plausibility conclusion
        print("\n" + "="*80)
        print("TRAJECTORY PLAUSIBILITY ASSESSMENT")
        print("="*80)
        print(f"{emoji} Overall Plausibility Score: {plausibility_score:.1f}/100 - {rating}")
        print("\nViolation Details:")
        print("-" * 40)
        
        #pos_df = pd.DataFrame(pos_stats)
        #vel_df = pd.DataFrame(vel_stats)
        #
        ## Create statistical plots with limits
        #fig, ((ax3, ax4)) = plt.subplots(1, 2, figsize=(16, 10))

        ## Position ranges
        #sns.barplot(data=pos_df, x='Joint', y='Range', ax=ax3, palette='viridis')
        #ax3.set_title('Joint Position Ranges', fontsize=14)
        #ax3.set_ylabel('Range (rad)', fontsize=12)
        #ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)

        ## Velocity ranges
        #sns.barplot(data=vel_df, x='Joint', y='Range', ax=ax4, palette='rocket')
        #ax4.set_title('Joint Velocity Ranges', fontsize=14)
        #ax4.set_ylabel('Range (rad/s)', fontsize=12)
        #ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)

        #plt.tight_layout()

        ## Save statistics plot
        #stats_save_path = os.path.join(self.figs_dir, "joint_statistics.png")
        #plt.savefig(stats_save_path, dpi=300, bbox_inches='tight')


def calculate_trajectory_plausibility(pos_stats, vel_stats):
    """
    Calculate trajectory plausibility metric (0-100)
    Higher score = more physically plausible
    """
    total_score = 0
    max_score = 0
    violation_details = []
    
    # Position plausibility (50% weight)
    for stat in pos_stats:
        joint_name = stat['Joint'].replace('_joint_dof_pos', '')
        pos_range = stat['Limit_Max'] - stat['Limit_Min']
        
        # Check lower limit violation
        lower_violation = max(0, stat['Limit_Min'] - stat['Min'])
        upper_violation = max(0, stat['Max'] - stat['Limit_Max'])
        
        total_violation = lower_violation + upper_violation
        
        if total_violation > 0:
            # Exponential penalty for violations
            violation_ratio = total_violation / pos_range  # Normalize by 10% of range
            score = 1 - violation_ratio
        else:
            score = 1
        
        total_score += score
        max_score += 1
    
    # Velocity plausibility (50% weight)
    for stat in vel_stats:
        joint_name = stat['Joint'].replace('_joint_dof_vel', '')
        vel_limit = stat['Limit_Max']
        
        # Check velocity limit violations
        lower_violation = max(0, -stat['Limit_Min'] - (-stat['Min']))  # Handle negative limits
        upper_violation = max(0, stat['Max'] - stat['Limit_Max'])
        
        total_violation = lower_violation + upper_violation
        
        if total_violation > 0:
            # Exponential penalty for velocity violations
            violation_ratio = total_violation / vel_limit  # Normalize by 20% of limit
            score = 1 - violation_ratio
        else:
            score = 1
        
        total_score += score
        max_score += 1
    
    if max_score > 0:
        plausibility_score = (total_score / max_score) * 100
    else:
        plausibility_score = 100

    
    return plausibility_score, violation_details


def get_plausibility_rating(score):
    """Convert numerical score to qualitative rating"""
    if score >= 95:
        return "EXCELLENT", "🟢"
    elif score >= 85:
        return "GOOD", "🟡"
    elif score >= 70:
        return "FAIR", "🟠"
    elif score >= 50:
        return "POOR", "🔴"
    else:
        return "UNSAFE", "💀"


# Example joint limits (you should replace this with actual URDF parsing)
def get_default_joint_limits():
    """Example joint limits - replace with actual URDF parsing"""
    return {
        'left_hip_pitch_joint_dof': {'pos_limit': [-1.57, 1.57], 'vel_limit': 8.0},
        'right_hip_pitch_joint_dof': {'pos_limit': [-1.57, 1.57], 'vel_limit': 8.0},
        'left_knee_joint_dof': {'pos_limit': [0, 2.0], 'vel_limit': 10.0},
        'right_knee_joint_dof': {'pos_limit': [0, 2.0], 'vel_limit': 10.0},
        # Add more joints as needed
    }


def parse_urdf_limits(urdf_file_path):
    """Parse joint limits from URDF file"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    
    joint_limits = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        limit_elem = joint.find('limit')
        if limit_elem is not None:
            lower = float(limit_elem.get('lower', -3.14))
            upper = float(limit_elem.get('upper', 3.14))
            velocity = float(limit_elem.get('velocity', 10.0))
            joint_limits[name] = {
                'pos_limit': [lower, upper],
                'vel_limit': velocity
            }
    
    return joint_limits


@hydra.main(version_base=None, config_path="./../cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    #1) loading ref traj by refmotion_manager
    # load dataset for demonstration
    from refmotion_manager.tests.test_loader_cfg import ref_motion_cfg
    ref_motion_cfg.time_between_frames = 0.02

    dataset = cfg.get("dataset",None)
    motion_files = "./../"+os.path.join(dataset.folder, dataset.file)
    print(motion_files)
    ref_motion_cfg.motion_files= glob.glob(motion_files)
    ref_motion_cfg.clip_num = 1
    ref_motion = RefMotionLoader(ref_motion_cfg)

    # Get joint limits (replace with your URDF parsing)
    joint_limits = get_default_joint_limits()
    asset = cfg.get("asset",None)
    robot_urdf_path = "./../"+os.path.join(asset.root, asset.urdf)
    print(f"humanoid xlm is {robot_urdf_path}")

    joint_limits = parse_urdf_limits(robot_urdf_path)
    
    # Create visualizer with joint limits
    visualizer = JointDataVisualizer(ref_motion, ref_motion_cfg, joint_limits)
    
    # Generate all plots
    print("Generating joint subplots...")
    visualizer.plot_all_joints_subplots(frame_start=0, frame_end=100)
    
    print("Generating statistics plots...")
    visualizer.plot_joint_statistics_seaborn(traj_idx=0)
    #plt.show()
    
    print("All plots saved to 'figs' directory!")

# Usage example
if __name__ == "__main__":
    main()
    
