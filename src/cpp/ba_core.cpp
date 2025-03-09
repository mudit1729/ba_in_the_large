#include "ba_core.h"
#include <cmath>
#include <iostream>

namespace ba_in_the_large {

bool SolveBundleAdjustment(
    const int num_cameras,
    const int num_points,
    const int num_observations,
    const std::vector<int>& camera_indices,
    const std::vector<int>& point_indices,
    const std::vector<double>& observations,
    std::vector<double>& camera_params,
    std::vector<double>& points,
    bool verbose) {
    
    if (camera_indices.size() != point_indices.size() || 
        camera_indices.size() != num_observations ||
        observations.size() != 2 * num_observations ||
        camera_params.size() != num_cameras * CameraModel::kNumParams ||
        points.size() != num_points * 3) {
        std::cerr << "Invalid input dimensions in SolveBundleAdjustment" << std::endl;
        return false;
    }

    // Build the Ceres problem
    ceres::Problem problem;

    // Add residual blocks for each observation
    for (int i = 0; i < num_observations; ++i) {
        const int camera_idx = camera_indices[i];
        const int point_idx = point_indices[i];
        const double observed_x = observations[2 * i];
        const double observed_y = observations[2 * i + 1];

        // Create a cost function based on the model
        ceres::CostFunction* cost_function = ReprojectionError::Create(observed_x, observed_y);

        // Add residual block to the problem
        problem.AddResidualBlock(
            cost_function,
            nullptr, // No robust loss function
            &camera_params[camera_idx * CameraModel::kNumParams],
            &points[point_idx * 3]);
    }

    // Configure the solver
    ceres::Solver::Options options;
    
    // Use sparse Cholesky solver (similar to the SciPy Jacobian sparse solver)
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    
    // Use sparse Jacobian (like in the Python implementation)
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    
    // Set verbosity level
    options.minimizer_progress_to_stdout = verbose;
    
    // Similar convergence criteria to SciPy's ftol=1e-4
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-10;  // Stricter than SciPy's default
    options.parameter_tolerance = 1e-8;  // Stricter than SciPy's default
    
    // Set maximum iterations (default in Ceres is 50)
    options.max_num_iterations = 100;
    
    // Run the solver
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // Print summary if verbose
    if (verbose) {
        std::cout << summary.BriefReport() << std::endl;
    }
    
    // Return true if optimization was successful
    return summary.IsSolutionUsable();
}

void ComputeResiduals(
    const std::vector<double>& camera_params,
    const std::vector<double>& points,
    const std::vector<int>& camera_indices,
    const std::vector<int>& point_indices,
    const std::vector<double>& observations,
    std::vector<double>& residuals) {
    
    const int num_observations = camera_indices.size();
    residuals.resize(2 * num_observations);
    
    for (int i = 0; i < num_observations; ++i) {
        const int camera_idx = camera_indices[i];
        const int point_idx = point_indices[i];
        const double observed_x = observations[2 * i];
        const double observed_y = observations[2 * i + 1];
        
        // Get camera parameters and 3D point
        const double* camera = &camera_params[camera_idx * CameraModel::kNumParams];
        const double* point = &points[point_idx * 3];
        double* res = &residuals[2 * i];
        
        // Compute residuals using the same model as in the cost function
        ReprojectionError reprojection_error(observed_x, observed_y);
        reprojection_error(camera, point, res);
    }
}

}  // namespace ba_in_the_large