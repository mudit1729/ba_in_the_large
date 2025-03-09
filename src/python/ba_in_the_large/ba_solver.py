import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import warnings
import time

# Import C++ implementation if available, otherwise use fallback mode
try:
    from .ba_cpp import solve_bundle_adjustment as solve_bundle_adjustment_ceres
    from .ba_cpp import compute_residuals as compute_residuals_ceres
    CERES_AVAILABLE = True
except ImportError:
    CERES_AVAILABLE = False
    warnings.warn("Ceres Solver C++ implementation not available. "
                 "Using SciPy implementation only. To use Ceres Solver, "
                 "make sure the ba_cpp extension is built.")

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def compute_residuals(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def solve_bundle_adjustment_scipy(camera_params, points_3d, camera_indices, point_indices, points_2d, verbose=2):
    """Solve bundle adjustment using SciPy's least_squares optimizer."""
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    res = least_squares(compute_residuals, x0, jac_sparsity=A, verbose=verbose, 
                        x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    
    return res


def solve_bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d, 
                           verbose=2, use_ceres=True):
    """Solve bundle adjustment using either Ceres Solver (C++) or SciPy.
    
    Args:
        camera_params: Camera parameters (n_cameras, 9)
        points_3d: 3D points (n_points, 3)
        camera_indices: Camera indices for each observation
        point_indices: Point indices for each observation
        points_2d: 2D observations (n_observations, 2)
        verbose: Verbosity level (0=silent, 1=minimal, 2=detailed)
        use_ceres: Whether to use Ceres Solver C++ implementation if available
        
    Returns:
        Result object with optimized parameters
    """
    # Use Ceres if requested and available
    if use_ceres and CERES_AVAILABLE:
        if verbose > 0:
            print("Using Ceres Solver C++ implementation...")
        
        start_time = time.time()
        
        # Call the C++ implementation
        result = solve_bundle_adjustment_ceres(
            camera_params, 
            points_3d, 
            camera_indices, 
            point_indices, 
            points_2d,
            verbose > 1  # Convert verbose level
        )
        
        elapsed_time = time.time() - start_time
        
        # Get results
        success = result["success"]
        camera_params_result = result["camera_params"]
        points_3d_result = result["points_3d"]
        residuals = result["residuals"]
        
        # Create a result object similar to SciPy's for compatibility
        class CeresResult:
            def __init__(self, success, x, fun, elapsed_time):
                self.success = success
                self.x = x
                self.fun = fun
                self.elapsed_time = elapsed_time
        
        # Combine parameters into a single array like the SciPy result
        x = np.hstack((camera_params_result.ravel(), points_3d_result.ravel()))
        
        # Create result object
        res = CeresResult(
            success=success,
            x=x,
            fun=residuals.ravel(),
            elapsed_time=elapsed_time
        )
        
        if verbose > 0:
            print(f"Ceres optimization {'succeeded' if success else 'failed'}")
            print(f"Optimization took {elapsed_time:.2f} seconds")
        
        return res
    else:
        # Fall back to SciPy implementation
        if use_ceres and not CERES_AVAILABLE:
            warnings.warn("Ceres Solver not available, falling back to SciPy implementation")
        
        if verbose > 0:
            print("Using SciPy implementation...")
        
        return solve_bundle_adjustment_scipy(
            camera_params, 
            points_3d, 
            camera_indices, 
            point_indices, 
            points_2d, 
            verbose
        )