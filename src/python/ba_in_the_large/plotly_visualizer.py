import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import math

def generate_camera_triangles(camera_positions, camera_rotations, scale=0.1):
    """
    Generate 3D triangles to represent cameras.
    
    Args:
        camera_positions: Array of camera positions (N, 3)
        camera_rotations: Array of camera rotation vectors (N, 3)
        scale: Size of the camera triangle representation
        
    Returns:
        List of dictionaries with vertex coordinates for each camera
    """
    camera_meshes = []
    
    for i, (pos, rot_vec) in enumerate(zip(camera_positions, camera_rotations)):
        # Create camera triangle shape (pyramid)
        # Base vertices in camera reference frame
        theta = np.linalg.norm(rot_vec)
        if theta < 1e-10:
            # If rotation is negligible, use identity rotation
            R = np.eye(3)
        else:
            # Convert rotation vector to matrix using Rodrigues formula
            axis = rot_vec / theta
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        # Create a camera pyramid with apex at the camera position
        # and base facing the -z direction (camera's viewing direction)
        
        # Apex point
        apex = pos
        
        # Base vertices in camera coordinates
        base_pts = np.array([
            [-scale, -scale, 2*scale],  # bottom-left
            [scale, -scale, 2*scale],   # bottom-right
            [0, scale, 2*scale],        # top
        ])
        
        # Transform base points to world coordinates
        base_pts_world = []
        for pt in base_pts:
            # Rotate point
            pt_rotated = R.dot(pt)
            # Apply translation
            pt_world = pt_rotated + pos
            base_pts_world.append(pt_world)
        
        # Create a dictionary with vertices and color
        camera_mesh = {
            'index': i,
            'position': pos,
            'vertices': [apex] + base_pts_world,
            'color': f'rgba(255, 0, 0, 0.8)'  # Red with some transparency
        }
        
        camera_meshes.append(camera_mesh)
    
    return camera_meshes

def create_camera_mesh(camera_mesh):
    """Create mesh3d object for a camera from its vertices."""
    vertices = camera_mesh['vertices']
    
    # Create triangular faces for the pyramid
    # Each face connects the apex (index 0) with two adjacent base vertices
    i, j, k = [], [], []
    
    # Face 1: apex, base0, base1
    i.append(0); j.append(1); k.append(2)
    # Face 2: apex, base1, base2
    i.append(0); j.append(2); k.append(3)
    # Face 3: apex, base2, base0
    i.append(0); j.append(3); k.append(1)
    # Face 4: base0, base1, base2 (base triangle)
    i.append(1); j.append(2); k.append(3)
    
    # Extract x, y, z coordinates
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]
    
    # Create the 3D mesh
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=camera_mesh['color'],
        opacity=0.8,
        hoverinfo='text',
        text=f"Camera {camera_mesh['index']}",
        showlegend=False,
    )
    
    return mesh

def plot_residuals_plotly(initial_residuals, final_residuals):
    """Plot initial and final residuals using Plotly."""
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Initial Residuals", "Final Residuals"))
    
    fig.add_trace(
        go.Scatter(y=initial_residuals, mode='lines', name='Initial'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=final_residuals, mode='lines', name='Final'),
        row=2, col=1
    )
    
    fig.update_layout(
        title_text="Bundle Adjustment Residuals",
        height=800,
        showlegend=False,
    )
    
    return fig

def visualize_reconstruction_plotly(initial_params, final_params, n_cameras, n_points):
    """
    Visualize cameras and 3D points before and after optimization using Plotly.
    
    This uses GPU acceleration for smoother interactive 3D visualization.
    """
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
    
    # Generate camera triangles for both initial and final states
    initial_camera_meshes = generate_camera_triangles(initial_camera_positions, initial_camera_rotations)
    final_camera_meshes = generate_camera_triangles(final_camera_positions, final_camera_rotations)
    
    # Create subplot with shared view
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=["Before Optimization", "After Optimization"],
        horizontal_spacing=0.05,
    )
    
    # Add 3D point clouds
    # Before optimization
    fig.add_trace(
        go.Scatter3d(
            x=initial_points_3d[:, 0],
            y=initial_points_3d[:, 1],
            z=initial_points_3d[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color='blue',
                opacity=0.6
            ),
            name='3D Points (Before)',
        ),
        row=1, col=1
    )
    
    # After optimization
    fig.add_trace(
        go.Scatter3d(
            x=final_points_3d[:, 0],
            y=final_points_3d[:, 1],
            z=final_points_3d[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color='blue',
                opacity=0.6
            ),
            name='3D Points (After)',
        ),
        row=1, col=2
    )
    
    # Add camera meshes
    for cam_mesh in initial_camera_meshes:
        mesh = create_camera_mesh(cam_mesh)
        fig.add_trace(mesh, row=1, col=1)
    
    for cam_mesh in final_camera_meshes:
        mesh = create_camera_mesh(cam_mesh)
        fig.add_trace(mesh, row=1, col=2)
    
    # Update layout for better visualization
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.5, y=-1.5, z=1.5)
    )
    
    scene_settings = dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='data',
        camera=camera,
    )
    
    fig.update_layout(
        title_text="Bundle Adjustment 3D Reconstruction",
        scene=scene_settings,
        scene2=scene_settings,
        height=800,
        width=1400,
        template="plotly_dark",
    )
    
    # Add a note about controls for the user
    annotations = [dict(
        text="Controls: Click and drag to rotate, scroll to zoom, shift+click to pan",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.5, y=0,
        font=dict(size=12)
    )]
    
    fig.update_layout(annotations=annotations)
    
    # Add JavaScript for shared camera movement between subplots
    fig.update_layout(
        updatemenus=[{
            'buttons': [{
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                'label': 'Sync Views',
                'method': 'animate'
            }],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    return fig