#include "Force/ForceSphereSphereContact.h"

#include "Body/BodySphere.h"
#include "Simulation.h"
#include "Utils.h"

namespace redmax {

namespace {

dtype sphere_contact_sigmoid(dtype x) {
    if (x >= 0.) {
        const dtype z = std::exp(-x);
        return 1. / (1. + z);
    }
    const dtype z = std::exp(x);
    return z / (1. + z);
}

dtype sphere_contact_softplus(dtype x) {
    return std::max(x, 0.) + std::log1p(std::exp(-std::abs(x)));
}

struct SphereNormalResponse {
    dtype magnitude;
    dtype derivative_gap;
    dtype derivative_velocity;
    dtype penetration;
    dtype penetration_derivative_gap;
};

SphereNormalResponse sphere_normal_response(
    dtype gap,
    dtype normal_velocity,
    dtype kn,
    dtype damping,
    dtype smoothing,
    dtype velocity_smoothing) {
    const dtype eps = std::max(smoothing, dtype(1e-8));
    const dtype veps = std::max(velocity_smoothing, dtype(1e-8));
    constexpr dtype cutoff_units = 8.;
    if (gap >= cutoff_units * eps) {
        return {0., 0., 0., 0., 0.};
    }
    const dtype x = -gap / eps;
    const dtype cutoff_x = -cutoff_units;
    const dtype cutoff_activation = sphere_contact_sigmoid(cutoff_x);
    const dtype raw_activation = sphere_contact_sigmoid(x);
    const dtype activation = raw_activation - cutoff_activation;
    const dtype penetration = eps * (
        sphere_contact_softplus(x)
        - sphere_contact_softplus(cutoff_x)
        - cutoff_activation * (x - cutoff_x));
    const dtype approach_activation =
        sphere_contact_sigmoid(-normal_velocity / veps);
    const dtype approach_speed =
        veps * sphere_contact_softplus(-normal_velocity / veps);
    return {
        kn * penetration + damping * activation * approach_speed,
        -kn * activation
            - damping * raw_activation * (1. - raw_activation)
                * approach_speed / eps,
        -damping * activation * approach_activation,
        penetration,
        -activation,
    };
}

}

ForceSphereSphereContact::ForceSphereSphereContact(
    Simulation* sim,
    const BodySphere* sphere1,
    const BodySphere* sphere2,
    dtype kn,
    dtype damping,
    dtype smoothing,
    dtype velocity_smoothing)
    : Force(sim),
      _sphere1(sphere1),
      _sphere2(sphere2),
      _kn(kn),
      _damping(damping),
      _base_kn(kn),
      _base_damping(damping),
      _scale(1.),
      _smoothing(smoothing),
      _velocity_smoothing(velocity_smoothing) {}

void ForceSphereSphereContact::set_scale(dtype scale) {
    if (!std::isfinite(scale) || scale < 0.) {
        throw std::invalid_argument("contact scale must be finite and nonnegative");
    }
    _scale = scale;
    _kn = _base_kn * scale;
    _damping = _base_damping * scale;
}

bool ForceSphereSphereContact::evaluate(
    Vector3& force,
    Matrix3* df_dx,
    Matrix3* df_dv,
    bool verbose) const {
    const Vector3 p1 = _sphere1->_E_0i.topRightCorner(3, 1);
    const Vector3 p2 = _sphere2->_E_0i.topRightCorner(3, 1);
    const Vector3 x = p1 - p2;
    const dtype center_distance = x.norm();
    const dtype radius_sum = _sphere1->_radius + _sphere2->_radius;

    force.setZero();
    if (df_dx != nullptr) {
        df_dx->setZero();
    }
    if (df_dv != nullptr) {
        df_dv->setZero();
    }

    if (center_distance - radius_sum >= 8. * _smoothing) {
        return false;
    }
    if (center_distance <= constants::eps) {
        throw_error(
            "Sphere-sphere contact is undefined for coincident centers: " +
            _sphere1->_name + " and " + _sphere2->_name
        );
    }

    const Matrix3 R1 = _sphere1->_E_0i.topLeftCorner(3, 3);
    const Matrix3 R2 = _sphere2->_E_0i.topLeftCorner(3, 3);
    const Vector3 velocity1 = R1 * _sphere1->_phi.tail(3);
    const Vector3 velocity2 = R2 * _sphere2->_phi.tail(3);
    const Vector3 relative_velocity = velocity1 - velocity2;
    const Vector3 normal = x / center_distance;
    const dtype gap = center_distance - radius_sum;
    const dtype normal_velocity = normal.dot(relative_velocity);
    const SphereNormalResponse response = sphere_normal_response(
        gap,
        normal_velocity,
        _kn,
        _damping,
        _smoothing,
        _velocity_smoothing);

    force = response.magnitude * normal;

    if (df_dx != nullptr || df_dv != nullptr) {
        const Matrix3 normal_jacobian =
            (Matrix3::Identity() - normal * normal.transpose()) / center_distance;
        if (df_dx != nullptr) {
            const RowVector3 dm_dx =
                response.derivative_gap * normal.transpose()
                + response.derivative_velocity
                    * relative_velocity.transpose() * normal_jacobian;
            *df_dx =
                normal * dm_dx + response.magnitude * normal_jacobian;
        }
        if (df_dv != nullptr) {
            *df_dv =
                response.derivative_velocity * normal * normal.transpose();
        }
    }

    if (verbose) {
        std::cerr << "Sphere-Sphere Contact: body1=" << _sphere1->_name
                  << ", body2=" << _sphere2->_name
                  << ", gap=" << gap
                  << ", normal_velocity=" << normal_velocity
                  << ", force=" << force.transpose() << std::endl;
    }
    return true;
}

void ForceSphereSphereContact::computeForce(
    VectorX& fm, VectorX& fr, bool verbose) {
    Vector3 world_force;
    if (!evaluate(world_force, nullptr, nullptr, verbose)) {
        return;
    }

    const Matrix3 R1 = _sphere1->_E_0i.topLeftCorner(3, 3);
    const Matrix3 R2 = _sphere2->_E_0i.topLeftCorner(3, 3);
    fm.segment(_sphere1->_index[0] + 3, 3) += R1.transpose() * world_force;
    fm.segment(_sphere2->_index[0] + 3, 3) -= R2.transpose() * world_force;
}

void ForceSphereSphereContact::computeForceWithDerivative(
    VectorX& fm, VectorX& fr,
    MatrixX& Km, MatrixX& Dm,
    MatrixX& Kr, MatrixX& Dr,
    bool verbose) {
    Vector3 world_force;
    Matrix3 df_dx;
    Matrix3 df_dv;
    if (!evaluate(world_force, &df_dx, &df_dv, verbose)) {
        return;
    }

    const Matrix3 R1 = _sphere1->_E_0i.topLeftCorner(3, 3);
    const Matrix3 R2 = _sphere2->_E_0i.topLeftCorner(3, 3);
    const Vector3 local_velocity1 = _sphere1->_phi.tail(3);
    const Vector3 local_velocity2 = _sphere2->_phi.tail(3);
    const int index1 = _sphere1->_index[0];
    const int index2 = _sphere2->_index[0];

    const Vector3 local_force1 = R1.transpose() * world_force;
    const Vector3 local_force2 = -R2.transpose() * world_force;
    fm.segment(index1 + 3, 3) += local_force1;
    fm.segment(index2 + 3, 3) += local_force2;

    Matrix36 dx_dq1 = Matrix36::Zero();
    Matrix36 dx_dq2 = Matrix36::Zero();
    dx_dq1.rightCols(3) = R1;
    dx_dq2.rightCols(3) = -R2;

    Matrix36 dv_dq1 = Matrix36::Zero();
    Matrix36 dv_dq2 = Matrix36::Zero();
    dv_dq1.leftCols(3) = -R1 * math::skew(local_velocity1);
    dv_dq2.leftCols(3) = R2 * math::skew(local_velocity2);

    Matrix36 dv_dphi1 = Matrix36::Zero();
    Matrix36 dv_dphi2 = Matrix36::Zero();
    dv_dphi1.rightCols(3) = R1;
    dv_dphi2.rightCols(3) = -R2;

    const Matrix36 df_dq1 = df_dx * dx_dq1 + df_dv * dv_dq1;
    const Matrix36 df_dq2 = df_dx * dx_dq2 + df_dv * dv_dq2;
    const Matrix36 df_dphi1 = df_dv * dv_dphi1;
    const Matrix36 df_dphi2 = df_dv * dv_dphi2;

    Km.block(index1 + 3, index1, 3, 6) += R1.transpose() * df_dq1;
    Km.block(index1 + 3, index1, 3, 3) += math::skew(local_force1);
    Km.block(index1 + 3, index2, 3, 6) += R1.transpose() * df_dq2;
    Km.block(index2 + 3, index1, 3, 6) -= R2.transpose() * df_dq1;
    Km.block(index2 + 3, index2, 3, 6) -= R2.transpose() * df_dq2;
    Km.block(index2 + 3, index2, 3, 3) += math::skew(local_force2);

    Dm.block(index1 + 3, index1, 3, 6) += R1.transpose() * df_dphi1;
    Dm.block(index1 + 3, index2, 3, 6) += R1.transpose() * df_dphi2;
    Dm.block(index2 + 3, index1, 3, 6) -= R2.transpose() * df_dphi1;
    Dm.block(index2 + 3, index2, 3, 6) -= R2.transpose() * df_dphi2;
}

void ForceSphereSphereContact::computeForceWithDerivative(
    VectorX& fm, VectorX& fr,
    MatrixX& Km, MatrixX& Dm,
    MatrixX& Kr, MatrixX& Dr,
    MatrixX& dfm_dp, MatrixX& dfr_dp,
    bool verbose) {
    computeForceWithDerivative(fm, fr, Km, Dm, Kr, Dr, verbose);
}

ContactPairMetric ForceSphereSphereContact::get_contact_pair_metric(
    const MatrixX& J,
    bool derivatives,
    dtype force_threshold) const {
    const int ndof_q = _sim->_ndof_r;
    const int ndof_p = _sim->_ndof_p;
    const Vector3 p1 = _sphere1->_E_0i.topRightCorner(3, 1);
    const Vector3 p2 = _sphere2->_E_0i.topRightCorner(3, 1);
    const Vector3 delta = p1 - p2;
    const dtype center_distance = delta.norm();
    if (center_distance <= constants::eps) {
        throw_error(
            "Sphere-sphere contact metric is undefined for coincident centers: "
            + _sphere1->_name + " and " + _sphere2->_name);
    }
    const Vector3 normal = delta / center_distance;
    const dtype gap =
        center_distance - _sphere1->_radius - _sphere2->_radius;
    const Matrix3 R1 = _sphere1->_E_0i.topLeftCorner(3, 3);
    const Matrix3 R2 = _sphere2->_E_0i.topLeftCorner(3, 3);
    const Vector3 velocity1 = R1 * _sphere1->_phi.tail(3);
    const Vector3 velocity2 = R2 * _sphere2->_phi.tail(3);
    const dtype normal_velocity =
        normal.dot(velocity1 - velocity2);
    const SphereNormalResponse response = sphere_normal_response(
        gap,
        normal_velocity,
        _kn,
        _damping,
        _smoothing,
        _velocity_smoothing);
    const dtype penetration = response.penetration;

    ContactPointMetric point;
    point.gap = gap;
    point.world_position = p1 - normal * _sphere1->_radius;
    point.normal = normal;
    point.elastic_force = _kn * penetration;
    Vector3 total_force;
    evaluate(total_force, nullptr, nullptr, false);
    point.total_normal_force = total_force.norm();
    point.dgap_dq = RowVectorX::Zero(ndof_q);
    point.dgap_dp = RowVectorX::Zero(ndof_p);
    point.delastic_dq = RowVectorX::Zero(ndof_q);
    point.delastic_dp = RowVectorX::Zero(ndof_p);

    if (derivatives) {
        RowVector6 dgap_body1, dgap_body2;
        RowVectorX dgap_dp1, dgap_dp2;
        body_point_gap_derivatives(
            _sphere1,
            Vector3::Zero(),
            -1,
            normal.transpose(),
            dgap_body1,
            dgap_dp1);
        body_point_gap_derivatives(
            _sphere2,
            Vector3::Zero(),
            -1,
            -normal.transpose(),
            dgap_body2,
            dgap_dp2);
        RowVectorX dgap_dmax = RowVectorX::Zero(_sim->_ndof_m);
        dgap_dmax.segment(_sphere1->_index[0], 6) = dgap_body1;
        dgap_dmax.segment(_sphere2->_index[0], 6) = dgap_body2;
        point.dgap_dq = dgap_dmax * J;
        point.dgap_dp = dgap_dp1 + dgap_dp2;
        const dtype delastic_dd =
            _kn * response.penetration_derivative_gap;
        point.delastic_dq = delastic_dd * point.dgap_dq;
        point.delastic_dp = delastic_dd * point.dgap_dp;
    }

    return aggregate_contact_pair_metric(
        "sphere_sphere",
        _sphere1->_name,
        _sphere2->_name,
        std::vector<ContactPointMetric>{point},
        _smoothing,
        force_threshold,
        ndof_q,
        ndof_p);
}

void ForceSphereSphereContact::test_derivatives_runtime() {
    VectorX fm = VectorX::Zero(_sim->_ndof_m);
    VectorX fr = VectorX::Zero(_sim->_ndof_r);
    MatrixX Km = MatrixX::Zero(_sim->_ndof_m, _sim->_ndof_m);
    MatrixX Dm = MatrixX::Zero(_sim->_ndof_m, _sim->_ndof_m);
    MatrixX Kr = MatrixX::Zero(_sim->_ndof_r, _sim->_ndof_r);
    MatrixX Dr = MatrixX::Zero(_sim->_ndof_r, _sim->_ndof_r);
    computeForceWithDerivative(fm, fr, Km, Dm, Kr, Dr);

    const Matrix4 E1 = _sphere1->_E_0i;
    const Matrix4 E2 = _sphere2->_E_0i;
    const Vector6 phi1 = _sphere1->_phi;
    const Vector6 phi2 = _sphere2->_phi;
    const dtype eps = 1e-7;
    MatrixX Km_fd = MatrixX::Zero(_sim->_ndof_m, 12);
    MatrixX Dm_fd = MatrixX::Zero(_sim->_ndof_m, 12);

    for (int i = 0; i < 6; ++i) {
        Vector6 dq = Vector6::Zero();
        dq[i] = eps;
        const_cast<BodySphere*>(_sphere1)->_E_0i = E1 * math::exp(dq);
        VectorX fm_pos = VectorX::Zero(_sim->_ndof_m);
        computeForce(fm_pos, fr);
        Km_fd.col(i) = (fm_pos - fm) / eps;
    }
    const_cast<BodySphere*>(_sphere1)->_E_0i = E1;
    for (int i = 0; i < 6; ++i) {
        Vector6 dq = Vector6::Zero();
        dq[i] = eps;
        const_cast<BodySphere*>(_sphere2)->_E_0i = E2 * math::exp(dq);
        VectorX fm_pos = VectorX::Zero(_sim->_ndof_m);
        computeForce(fm_pos, fr);
        Km_fd.col(6 + i) = (fm_pos - fm) / eps;
    }
    const_cast<BodySphere*>(_sphere2)->_E_0i = E2;

    for (int i = 0; i < 6; ++i) {
        Vector6 phi = phi1;
        phi[i] += eps;
        const_cast<BodySphere*>(_sphere1)->_phi = phi;
        VectorX fm_pos = VectorX::Zero(_sim->_ndof_m);
        computeForce(fm_pos, fr);
        Dm_fd.col(i) = (fm_pos - fm) / eps;
    }
    const_cast<BodySphere*>(_sphere1)->_phi = phi1;
    for (int i = 0; i < 6; ++i) {
        Vector6 phi = phi2;
        phi[i] += eps;
        const_cast<BodySphere*>(_sphere2)->_phi = phi;
        VectorX fm_pos = VectorX::Zero(_sim->_ndof_m);
        computeForce(fm_pos, fr);
        Dm_fd.col(6 + i) = (fm_pos - fm) / eps;
    }
    const_cast<BodySphere*>(_sphere2)->_phi = phi2;

    MatrixX Km_local(_sim->_ndof_m, 12);
    Km_local.leftCols(6) = Km.middleCols(_sphere1->_index[0], 6);
    Km_local.rightCols(6) = Km.middleCols(_sphere2->_index[0], 6);
    MatrixX Dm_local(_sim->_ndof_m, 12);
    Dm_local.leftCols(6) = Dm.middleCols(_sphere1->_index[0], 6);
    Dm_local.rightCols(6) = Dm.middleCols(_sphere2->_index[0], 6);
    const dtype km_error = (Km_local - Km_fd).norm();
    const dtype dm_error = (Dm_local - Dm_fd).norm();
    std::cerr << "Sphere-Sphere Contact derivative check: Km error=" << km_error
              << ", Dm error=" << dm_error << std::endl;
    if (km_error > 1e-4 || dm_error > 1e-4) {
        throw_error("Sphere-sphere contact derivative check failed");
    }
}

}
