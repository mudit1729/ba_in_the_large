#pragma once

#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace ba_in_the_large {

// Camera model for bundle adjustment
struct CameraModel {
    // Camera parameters: 
    // 3 for rotation (angle-axis), 
    // 3 for translation, 
    // 1 for focal length, 
    // 2 for radial distortion (k1, k2)
    static constexpr int kNumParams = 9;
};

// Reprojection error for bundle adjustment
// Uses the same camera model as the Python implementation
class ReprojectionError {
public:
    ReprojectionError(const double observed_x, const double observed_y)
        : observed_x_(observed_x), observed_y_(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // Camera parameters (same as Python implementation)
        const T* rotation = camera;
        const T* translation = camera + 3;
        const T& f = camera[6];
        const T& k1 = camera[7];
        const T& k2 = camera[8];

        // Rotate point using angle-axis rotation
        T p[3];
        ceres::AngleAxisRotatePoint(rotation, point, p);

        // Apply translation
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        // Project to image coordinates
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply radial distortion
        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + k1 * r2 + k2 * r2 * r2;
        
        // Apply focal length
        xp = f * distortion * xp;
        yp = f * distortion * yp;

        // Compute residuals
        residuals[0] = xp - T(observed_x_);
        residuals[1] = yp - T(observed_y_);

        return true;
    }

    // Factory to hide the construction of the CostFunction object
    static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, CameraModel::kNumParams, 3>(
            new ReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x_;
    double observed_y_;
};

// Function to solve bundle adjustment problem using Ceres
bool SolveBundleAdjustment(
    const int num_cameras,
    const int num_points,
    const int num_observations,
    const std::vector<int>& camera_indices,
    const std::vector<int>& point_indices,
    const std::vector<double>& observations,
    std::vector<double>& camera_params,
    std::vector<double>& points,
    bool verbose = true);

// Function to compute residuals (for comparison with Python)
void ComputeResiduals(
    const std::vector<double>& camera_params,
    const std::vector<double>& points,
    const std::vector<int>& camera_indices,
    const std::vector<int>& point_indices,
    const std::vector<double>& observations,
    std::vector<double>& residuals);

}  // namespace ba_in_the_large