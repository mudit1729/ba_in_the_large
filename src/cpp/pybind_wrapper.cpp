#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "ba_core.h"

namespace py = pybind11;

// Helper function to convert numpy arrays to std::vector
template <typename T>
std::vector<T> numpy_to_vector(py::array_t<T> array) {
    py::buffer_info buffer = array.request();
    T* data = static_cast<T*>(buffer.ptr);
    return std::vector<T>(data, data + buffer.size);
}

// Wrapper for the C++ bundle adjustment solver
py::dict solve_bundle_adjustment_ceres(
    py::array_t<double> camera_params_array,
    py::array_t<double> points_3d_array,
    py::array_t<int> camera_indices_array,
    py::array_t<int> point_indices_array,
    py::array_t<double> points_2d_array,
    bool verbose = true) {
    
    // Convert numpy arrays to std::vector
    std::vector<double> camera_params = numpy_to_vector(camera_params_array);
    std::vector<double> points_3d = numpy_to_vector(points_3d_array);
    std::vector<int> camera_indices = numpy_to_vector(camera_indices_array);
    std::vector<int> point_indices = numpy_to_vector(point_indices_array);
    
    // Convert 2D points to a flattened vector
    py::buffer_info points_2d_buffer = points_2d_array.request();
    if (points_2d_buffer.ndim != 2 || points_2d_buffer.shape[1] != 2) {
        throw std::runtime_error("points_2d must be a Nx2 array");
    }
    
    const int num_observations = points_2d_buffer.shape[0];
    std::vector<double> observations(2 * num_observations);
    double* points_2d_data = static_cast<double*>(points_2d_buffer.ptr);
    
    for (int i = 0; i < num_observations; ++i) {
        observations[2 * i] = points_2d_data[i * 2];
        observations[2 * i + 1] = points_2d_data[i * 2 + 1];
    }
    
    // Get dimensions
    const int num_cameras = camera_params.size() / ba_in_the_large::CameraModel::kNumParams;
    const int num_points = points_3d.size() / 3;
    
    // Make copies of parameters for optimization
    std::vector<double> camera_params_optimized = camera_params;
    std::vector<double> points_3d_optimized = points_3d;
    
    // Solve the bundle adjustment problem
    bool success = ba_in_the_large::SolveBundleAdjustment(
        num_cameras,
        num_points,
        num_observations,
        camera_indices,
        point_indices,
        observations,
        camera_params_optimized,
        points_3d_optimized,
        verbose
    );
    
    // Compute residuals after optimization
    std::vector<double> residuals;
    ba_in_the_large::ComputeResiduals(
        camera_params_optimized,
        points_3d_optimized,
        camera_indices,
        point_indices,
        observations,
        residuals
    );
    
    // Convert optimized parameters back to numpy arrays
    py::array_t<double> camera_params_result = py::array_t<double>(camera_params_optimized.size());
    py::buffer_info camera_buffer = camera_params_result.request();
    double* camera_ptr = static_cast<double*>(camera_buffer.ptr);
    std::copy(camera_params_optimized.begin(), camera_params_optimized.end(), camera_ptr);
    
    py::array_t<double> points_3d_result = py::array_t<double>(points_3d_optimized.size());
    py::buffer_info points_buffer = points_3d_result.request();
    double* points_ptr = static_cast<double*>(points_buffer.ptr);
    std::copy(points_3d_optimized.begin(), points_3d_optimized.end(), points_ptr);
    
    py::array_t<double> residuals_result = py::array_t<double>(residuals.size());
    py::buffer_info residuals_buffer = residuals_result.request();
    double* residuals_ptr = static_cast<double*>(residuals_buffer.ptr);
    std::copy(residuals.begin(), residuals.end(), residuals_ptr);
    
    // Reshape arrays to match input format
    camera_params_result = camera_params_result.reshape({num_cameras, ba_in_the_large::CameraModel::kNumParams});
    points_3d_result = points_3d_result.reshape({num_points, 3});
    residuals_result = residuals_result.reshape({num_observations, 2});
    
    // Return results as a Python dictionary
    py::dict result;
    result["success"] = success;
    result["camera_params"] = camera_params_result;
    result["points_3d"] = points_3d_result;
    result["residuals"] = residuals_result;
    
    return result;
}

// Compute residuals only (for testing/validation)
py::array_t<double> compute_residuals_ceres(
    py::array_t<double> camera_params_array,
    py::array_t<double> points_3d_array,
    py::array_t<int> camera_indices_array,
    py::array_t<int> point_indices_array,
    py::array_t<double> points_2d_array) {
    
    // Convert numpy arrays to std::vector
    std::vector<double> camera_params = numpy_to_vector(camera_params_array);
    std::vector<double> points_3d = numpy_to_vector(points_3d_array);
    std::vector<int> camera_indices = numpy_to_vector(camera_indices_array);
    std::vector<int> point_indices = numpy_to_vector(point_indices_array);
    
    // Convert 2D points to a flattened vector
    py::buffer_info points_2d_buffer = points_2d_array.request();
    if (points_2d_buffer.ndim != 2 || points_2d_buffer.shape[1] != 2) {
        throw std::runtime_error("points_2d must be a Nx2 array");
    }
    
    const int num_observations = points_2d_buffer.shape[0];
    std::vector<double> observations(2 * num_observations);
    double* points_2d_data = static_cast<double*>(points_2d_buffer.ptr);
    
    for (int i = 0; i < num_observations; ++i) {
        observations[2 * i] = points_2d_data[i * 2];
        observations[2 * i + 1] = points_2d_data[i * 2 + 1];
    }
    
    // Compute residuals
    std::vector<double> residuals;
    ba_in_the_large::ComputeResiduals(
        camera_params,
        points_3d,
        camera_indices,
        point_indices,
        observations,
        residuals
    );
    
    // Convert residuals to numpy array
    py::array_t<double> residuals_result = py::array_t<double>(residuals.size());
    py::buffer_info residuals_buffer = residuals_result.request();
    double* residuals_ptr = static_cast<double*>(residuals_buffer.ptr);
    std::copy(residuals.begin(), residuals.end(), residuals_ptr);
    
    // Reshape to match 2D format
    residuals_result = residuals_result.reshape({num_observations, 2});
    
    return residuals_result;
}

PYBIND11_MODULE(ba_cpp, m) {
    m.doc() = "C++ implementation of bundle adjustment using Ceres Solver";
    
    m.def("solve_bundle_adjustment", &solve_bundle_adjustment_ceres, 
          py::arg("camera_params"),
          py::arg("points_3d"),
          py::arg("camera_indices"),
          py::arg("point_indices"),
          py::arg("points_2d"),
          py::arg("verbose") = true,
          "Solve bundle adjustment problem using Ceres Solver");
    
    m.def("compute_residuals", &compute_residuals_ceres,
          py::arg("camera_params"),
          py::arg("points_3d"),
          py::arg("camera_indices"),
          py::arg("point_indices"),
          py::arg("points_2d"),
          "Compute residuals for bundle adjustment problem");
}