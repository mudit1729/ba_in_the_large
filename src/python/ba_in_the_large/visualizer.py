import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_residuals(initial_residuals, final_residuals):
    """Plot initial and final residuals."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(initial_residuals)
    plt.title('Initial Residuals')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(final_residuals)
    plt.title('Final Residuals')
    plt.grid(True)
    
    plt.tight_layout()
    return plt
    
def display_optimization_results(initial_params, final_params, elapsed_time, utils):
    """Display optimization results."""
    print("Optimization took {0:.0f} seconds".format(elapsed_time))
    
    print('Before:')
    print('cam0: {}'.format(utils.prettylist(initial_params[0:9])))
    print('cam1: {}'.format(utils.prettylist(initial_params[9:18])))
    
    print('After:')
    print('cam0: {}'.format(utils.prettylist(final_params[0:9])))
    print('cam1: {}'.format(utils.prettylist(final_params[9:18])))

def visualize_reconstruction(initial_params, final_params, n_cameras, n_points):
    """Visualize cameras and 3D points before and after optimization."""
    # Extract camera parameters and 3D points
    initial_camera_params = initial_params[:n_cameras * 9].reshape((n_cameras, 9))
    initial_points_3d = initial_params[n_cameras * 9:].reshape((n_points, 3))
    
    final_camera_params = final_params[:n_cameras * 9].reshape((n_cameras, 9))
    final_points_3d = final_params[n_cameras * 9:].reshape((n_points, 3))
    
    # Extract camera positions (translation part of camera parameters)
    initial_camera_positions = initial_camera_params[:, 3:6]
    final_camera_positions = final_camera_params[:, 3:6]
    
    # Create 3D plot with two subplots (before and after)
    fig = plt.figure(figsize=(15, 8))
    
    # Before optimization subplot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(initial_camera_positions[:, 0], initial_camera_positions[:, 1], initial_camera_positions[:, 2], 
              c='red', marker='o', s=50, label='Cameras')
    ax1.scatter(initial_points_3d[:, 0], initial_points_3d[:, 1], initial_points_3d[:, 2], 
              c='blue', marker='.', s=1, alpha=0.5, label='3D Points')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Before Optimization')
    ax1.legend()
    
    # After optimization subplot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(final_camera_positions[:, 0], final_camera_positions[:, 1], final_camera_positions[:, 2], 
              c='red', marker='o', s=50, label='Cameras')
    ax2.scatter(final_points_3d[:, 0], final_points_3d[:, 1], final_points_3d[:, 2], 
              c='blue', marker='.', s=1, alpha=0.5, label='3D Points')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('After Optimization')
    ax2.legend()
    
    # Set equal aspect ratio for both plots
    # Calculate ranges for consistent scaling
    all_points = np.vstack((initial_points_3d, final_points_3d))
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    # Apply same scaling to both plots
    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return plt