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
    
    # Extract camera rotation (rotation part of camera parameters)
    initial_camera_rotations = initial_camera_params[:, :3]
    final_camera_rotations = final_camera_params[:, :3]
    
    # Create 3D plot with two subplots (before and after)
    fig = plt.figure(figsize=(15, 10))
    
    # Add the visualization angle text above the plots
    fig.text(0.5, 0.95, 'Viewing Angle: Initially set to Camera 0 perspective', 
             ha='center', va='center', fontsize=12)
    
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
    
    # Set appropriate view for each plot
    def calculate_view_bounds(camera_positions, points_3d, camera_idx=0):
        """Calculate appropriate view bounds from a camera's perspective."""
        # Get the selected camera's position
        camera_pos = camera_positions[camera_idx]
        
        # Calculate distances from the camera to all 3D points
        distances = np.linalg.norm(points_3d - camera_pos, axis=1)
        
        # Find points that are within a reasonable distance (exclude outliers)
        # Use a percentile threshold to define "reasonable"
        threshold_distance = np.percentile(distances, 95)  # 95th percentile
        visible_points = points_3d[distances <= threshold_distance]
        
        if len(visible_points) < 10:  # If too few points, use all points
            visible_points = points_3d
            
        # Calculate the bounding box of visible points
        if len(visible_points) > 0:
            min_x, min_y, min_z = np.min(visible_points, axis=0)
            max_x, max_y, max_z = np.max(visible_points, axis=0)
            
            # Calculate center and range
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_z = (min_z + max_z) / 2
            
            # Calculate appropriate range for zoom level, adding a margin
            range_x = max(max_x - min_x, 1e-5) * 1.2  # Avoid zero range
            range_y = max(max_y - min_y, 1e-5) * 1.2
            range_z = max(max_z - min_z, 1e-5) * 1.2
            
            max_range = max(range_x, range_y, range_z) / 2
            
            return center_x, center_y, center_z, max_range
        else:
            # Fallback if no points are found
            return camera_pos[0], camera_pos[1], camera_pos[2], 10.0
    
    # Calculate view bounds for both initial and final views
    initial_center_x, initial_center_y, initial_center_z, initial_range = calculate_view_bounds(
        initial_camera_positions, initial_points_3d)
    
    final_center_x, final_center_y, final_center_z, final_range = calculate_view_bounds(
        final_camera_positions, final_points_3d)
    
    # Apply the calculated view bounds to each plot
    ax1.set_xlim(initial_center_x - initial_range, initial_center_x + initial_range)
    ax1.set_ylim(initial_center_y - initial_range, initial_center_y + initial_range)
    ax1.set_zlim(initial_center_z - initial_range, initial_center_z + initial_range)
    
    ax2.set_xlim(final_center_x - final_range, final_center_x + final_range)
    ax2.set_ylim(final_center_y - final_range, final_center_y + final_range)
    ax2.set_zlim(final_center_z - final_range, final_center_z + final_range)
    
    # Set the viewing angle to be from the first camera's perspective
    # We need to compute the viewing angles from the rotation matrix
    def get_view_angles_from_camera(camera_params, camera_idx=0):
        from scipy.spatial.transform import Rotation
        
        # Get the rotation vector for the selected camera
        rot_vec = camera_params[camera_idx, :3]
        
        # Convert to a 3x3 rotation matrix using Rodrigues' formula
        theta = np.linalg.norm(rot_vec)
        if theta > 0:
            axis = rot_vec / theta
            r = Rotation.from_rotvec(theta * axis)
            rotation_matrix = r.as_matrix()
            
            # The camera looks along the negative z-axis in camera coordinates
            # We need to transform this to determine the viewing direction
            view_direction = rotation_matrix.T.dot(np.array([0, 0, -1]))
            
            # Convert to elevation and azimuth angles for matplotlib's view_init
            # Elevation: angle from the xy-plane (in degrees)
            elevation = np.degrees(np.arcsin(view_direction[2]))
            # Azimuth: angle in the xy-plane from the x-axis (in degrees)
            azimuth = np.degrees(np.arctan2(view_direction[1], view_direction[0]))
            
            return elevation, azimuth
        else:
            # Default to a standard view if no rotation
            return 20, -60  # default matplotlib 3D view
        
    # Set the initial view to match the first camera
    try:
        initial_elev, initial_azim = get_view_angles_from_camera(initial_camera_params)
        ax1.view_init(elev=initial_elev, azim=initial_azim)
        ax2.view_init(elev=initial_elev, azim=initial_azim)  # Use same viewing angle for both
        
        # Update the viewing angle text
        fig.text(0.5, 0.95, f'Viewing Angle: Elev={initial_elev:.1f}°, Azim={initial_azim:.1f}° (Camera 0 perspective)', 
                ha='center', va='center', fontsize=12)
        
    except (ImportError, ValueError):
        # Fallback if we can't compute the viewing angle
        default_elev, default_azim = 20, -60
        ax1.view_init(elev=default_elev, azim=default_azim)
        ax2.view_init(elev=default_elev, azim=default_azim)
    
    # Add a synchronized view function to keep plots aligned during rotation
    def on_move(event):
        if event.inaxes == ax1:
            if hasattr(event, 'button') and event.button in [1, 3]:  # Check if it's a mouse drag
                # Get the current view angles from the first plot
                elev, azim = ax1.elev, ax1.azim
                # Apply the same view to the second plot
                ax2.view_init(elev=elev, azim=azim)
                # Update the viewing angle text
                fig.texts[0].set_text(f'Viewing Angle: Elev={elev:.1f}°, Azim={azim:.1f}°')
                # Redraw the figure
                fig.canvas.draw_idle()
        elif event.inaxes == ax2:
            if hasattr(event, 'button') and event.button in [1, 3]:  # Check if it's a mouse drag
                # Get the current view angles from the second plot
                elev, azim = ax2.elev, ax2.azim
                # Apply the same view to the first plot
                ax1.view_init(elev=elev, azim=azim)
                # Update the viewing angle text
                fig.texts[0].set_text(f'Viewing Angle: Elev={elev:.1f}°, Azim={azim:.1f}°')
                # Redraw the figure
                fig.canvas.draw_idle()
    
    # Connect the function to the mouse motion event
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title
    return fig