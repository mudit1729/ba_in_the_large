from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from ba_in_the_large import (
    read_bal_data, 
    solve_bundle_adjustment, 
    plot_residuals,
    display_optimization_results,
    visualize_reconstruction,
    prettylist,
    PLOTLY_AVAILABLE
)

# Import Plotly visualizers if available
if PLOTLY_AVAILABLE:
    from ba_in_the_large import (
        visualize_reconstruction_plotly,
        plot_residuals_plotly
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bundle Adjustment in the Large')
    parser.add_argument('--file', type=str, default="problem-49-7776-pre.txt",
                        help='Path to the BAL dataset file')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the optimization results and 3D reconstruction')
    parser.add_argument('--solver', type=str, choices=['scipy', 'ceres', 'both'], default='scipy',
                        help='Solver to use: scipy (Python), ceres (C++), or both (for comparison)')
    parser.add_argument('--verbose', type=int, default=2, choices=[0, 1, 2],
                        help='Verbosity level: 0=silent, 1=minimal, 2=detailed')
    parser.add_argument('--engine', type=str, choices=['matplotlib', 'plotly'], default='plotly',
                        help='Visualization engine: matplotlib or plotly (with GPU acceleration)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file for Plotly visualization')
    args = parser.parse_args()
    
    # Check if Plotly is available when requested
    if args.engine == 'plotly' and not PLOTLY_AVAILABLE:
        print("Warning: Plotly visualization requested but not available. Falling back to matplotlib.")
        print("To use Plotly, install it with: pip install plotly")
        args.engine = 'matplotlib'
    
    # Read the BAL dataset
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(args.file)
    
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
    
    if args.solver == 'both':
        # Run both solvers and compare
        print("\n=== Running SciPy Solver (Python) ===")
        t0_scipy = time.time()
        res_scipy = solve_bundle_adjustment(
            camera_params, points_3d, camera_indices, point_indices, points_2d, 
            verbose=args.verbose, use_ceres=False
        )
        t1_scipy = time.time()
        
        print("\n=== Running Ceres Solver (C++) ===")
        t0_ceres = time.time()
        res_ceres = solve_bundle_adjustment(
            camera_params, points_3d, camera_indices, point_indices, points_2d, 
            verbose=args.verbose, use_ceres=True
        )
        t1_ceres = time.time()
        
        # Display comparison
        print("\n=== Solver Comparison ===")
        print("SciPy time: {:.2f} seconds".format(t1_scipy - t0_scipy))
        print("Ceres time: {:.2f} seconds".format(t1_ceres - t0_ceres))
        print("Speed improvement: {:.2f}x".format((t1_scipy - t0_scipy) / (t1_ceres - t0_ceres)))
        
        # Display results from Ceres (usually faster and more accurate)
        from ba_in_the_large import utils
        print("\n=== Final Results (Ceres) ===")
        display_optimization_results(x0, res_ceres.x, t1_ceres - t0_ceres, utils)
        
        # Use Ceres results for visualization
        res = res_ceres
    else:
        # Run the selected solver
        use_ceres = (args.solver == 'ceres')
        t0 = time.time()
        res = solve_bundle_adjustment(
            camera_params, points_3d, camera_indices, point_indices, points_2d, 
            verbose=args.verbose, use_ceres=use_ceres
        )
        t1 = time.time()
        
        # Display results
        from ba_in_the_large import utils
        display_optimization_results(x0, res.x, t1 - t0, utils)
    
    # Visualize if requested
    if args.visualize:
        # Get the base file name for output files
        base_filename = os.path.splitext(os.path.basename(args.file))[0]
        
        if args.engine == 'plotly':
            print("\nUsing Plotly for visualization with GPU acceleration")
            
            # Create output directory if not exists
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine output HTML file path
            output_file = args.output
            if output_file is None:
                output_file = os.path.join(output_dir, f"{base_filename}_visualization.html")
            
            # Create reconstruction visualization
            recon_fig = visualize_reconstruction_plotly(x0, res.x, n_cameras, n_points)
            
            # Create residuals visualization
            residual_fig = plot_residuals_plotly(f0, res.fun)
            
            # Save the visualization to an HTML file
            print(f"Saving visualization to {output_file}")
            recon_fig.write_html(output_file)
            
            # Save residuals to a separate file
            residuals_file = os.path.join(output_dir, f"{base_filename}_residuals.html")
            residual_fig.write_html(residuals_file)
            
            print("\nVisualization controls:")
            print("- Rotate: Click and drag with the mouse")
            print("- Zoom: Mouse wheel or trackpad")
            print("- Pan: Right-click and drag")
            print("- Reset view: Double-click")
            
            # Show the visualization in the browser
            recon_fig.show()
            residual_fig.show()
            
        else:
            # Use Matplotlib for visualization
            print("\nUsing Matplotlib for visualization")
            
            # Plot residuals
            residual_fig = plot_residuals(f0, res.fun)
            
            # Visualize 3D reconstruction before and after optimization
            reconstruction_fig = visualize_reconstruction(x0, res.x, n_cameras, n_points)
            
            print("\nVisualization controls:")
            print("- Rotate: Click and drag with the mouse")
            print("- Zoom: Use the mouse wheel")
            print("- Both plots rotate together to maintain the same viewing angle")
            print("- Current viewing angle is displayed above the plots")
            
            # Show all plots
            plt.show()

if __name__ == "__main__":
    main()