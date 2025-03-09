from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from ba_in_the_large import (
    read_bal_data, 
    solve_bundle_adjustment, 
    plot_residuals, 
    display_optimization_results,
    prettylist
)

FILE_NAME = "problem-49-7776-pre.txt"

def main():
    # Read the BAL dataset
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
    
    # Print information
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    
    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    
    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    
    # Prepare initial parameters
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    
    # Compute initial residuals
    from ba_in_the_large.ba_solver import compute_residuals
    f0 = compute_residuals(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    
    # Run the bundle adjustment
    t0 = time.time()
    res = solve_bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d)
    t1 = time.time()
    
    # Display results
    from ba_in_the_large import utils
    display_optimization_results(x0, res.x, t1 - t0, utils)
    
    # Plot residuals
    plot = plot_residuals(f0, res.fun)
    plt.show()

if __name__ == "__main__":
    main()