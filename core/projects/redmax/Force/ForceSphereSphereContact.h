#pragma once

#include "Common.h"
#include "Force/Force.h"
#include "ContactMetrics.h"

namespace redmax {

class BodySphere;

class ForceSphereSphereContact : public Force {
public:
    const BodySphere* _sphere1;
    const BodySphere* _sphere2;
    dtype _kn;
    dtype _damping;
    dtype _base_kn;
    dtype _base_damping;
    dtype _scale;
    dtype _smoothing;
    dtype _velocity_smoothing;

    ForceSphereSphereContact(
        Simulation* sim,
        const BodySphere* sphere1,
        const BodySphere* sphere2,
        dtype kn,
        dtype damping,
        dtype smoothing = 0.02,
        dtype velocity_smoothing = 0.05);

    void computeForce(VectorX& fm, VectorX& fr, bool verbose = false);
    void computeForceWithDerivative(
        VectorX& fm, VectorX& fr,
        MatrixX& Km, MatrixX& Dm,
        MatrixX& Kr, MatrixX& Dr,
        bool verbose = false);
    void computeForceWithDerivative(
        VectorX& fm, VectorX& fr,
        MatrixX& Km, MatrixX& Dm,
        MatrixX& Kr, MatrixX& Dr,
        MatrixX& dfm_dp, MatrixX& dfr_dp,
        bool verbose = false);

    void test_derivatives_runtime();
    void set_scale(dtype scale);

    ContactPairMetric get_contact_pair_metric(
        const MatrixX& J,
        bool derivatives,
        dtype force_threshold) const;

private:
    bool evaluate(
        Vector3& force,
        Matrix3* df_dx,
        Matrix3* df_dv,
        bool verbose) const;
};

}
