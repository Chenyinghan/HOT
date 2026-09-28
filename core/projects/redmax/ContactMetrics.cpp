#include "ContactMetrics.h"

#include "Body/Body.h"
#include "Joint/Joint.h"
#include "Simulation.h"

#include <algorithm>

namespace redmax {

dtype contact_metric_sigmoid(dtype value) {
    if (value >= 0.) {
        const dtype z = std::exp(-value);
        return 1. / (1. + z);
    }
    const dtype z = std::exp(value);
    return z / (1. + z);
}

dtype contact_metric_softplus(dtype value) {
    return std::max(value, dtype(0.)) + std::log1p(std::exp(-std::abs(value)));
}

ContactPairMetric aggregate_contact_pair_metric(
    const std::string& contact_type,
    const std::string& body1,
    const std::string& body2,
    const std::vector<ContactPointMetric>& points,
    dtype smoothing,
    dtype force_threshold,
    int ndof_q,
    int ndof_p) {
    ContactPairMetric metric;
    metric.contact_type = contact_type;
    metric.body1 = body1;
    metric.body2 = body2;
    metric.sample_count = int(points.size());
    metric.dactivation_dq = VectorX::Zero(ndof_q);
    metric.dactivation_dp = VectorX::Zero(ndof_p);
    metric.delastic_force_dq = VectorX::Zero(ndof_q);
    metric.delastic_force_dp = VectorX::Zero(ndof_p);

    if (points.empty()) {
        return metric;
    }

    const dtype eps = std::max(smoothing, dtype(1e-8));
    dtype min_gap = std::numeric_limits<dtype>::infinity();
    for (const auto& point : points) {
        min_gap = std::min(min_gap, point.gap);
        metric.elastic_normal_force += point.elastic_force;
        metric.total_normal_force += point.total_normal_force;
        if (point.delastic_dq.size() == ndof_q) {
            metric.delastic_force_dq += point.delastic_dq.transpose();
        }
        if (point.delastic_dp.size() == ndof_p) {
            metric.delastic_force_dp += point.delastic_dp.transpose();
        }
        if (point.gap <= 0.) {
            if (point.contact_point_id >= 0) {
                metric.contact_point_ids.push_back(point.contact_point_id);
            }
            metric.world_positions.push_back(point.world_position);
            metric.normals.push_back(point.normal);
        }
    }

    dtype exp_sum = 0.;
    std::vector<dtype> weights(points.size(), 0.);
    for (size_t i = 0; i < points.size(); ++i) {
        weights[i] = std::exp(-(points[i].gap - min_gap) / eps);
        exp_sum += weights[i];
    }
    exp_sum = std::max(exp_sum, std::numeric_limits<dtype>::min());
    const dtype soft_min_gap =
        min_gap - eps * std::log(exp_sum / dtype(points.size()));
    metric.activation = contact_metric_sigmoid(-soft_min_gap / eps);

    RowVectorX dsoftmin_dq = RowVectorX::Zero(ndof_q);
    RowVectorX dsoftmin_dp = RowVectorX::Zero(ndof_p);
    for (size_t i = 0; i < points.size(); ++i) {
        const dtype weight = weights[i] / exp_sum;
        if (points[i].dgap_dq.size() == ndof_q) {
            dsoftmin_dq += weight * points[i].dgap_dq;
        }
        if (points[i].dgap_dp.size() == ndof_p) {
            dsoftmin_dp += weight * points[i].dgap_dp;
        }
    }
    const dtype dactivation_dgap =
        -metric.activation * (1. - metric.activation) / eps;
    metric.dactivation_dq = (dactivation_dgap * dsoftmin_dq).transpose();
    metric.dactivation_dp = (dactivation_dgap * dsoftmin_dp).transpose();

    metric.min_gap = min_gap;
    metric.max_penetration = std::max(dtype(0.), -min_gap);
    metric.in_activation_band = min_gap < 8. * eps;
    metric.geometrically_touching = min_gap <= 0.;
    metric.force_active = metric.total_normal_force > force_threshold;
    return metric;
}

void body_point_gap_derivatives(
    const Body* body,
    const Vector3& local_point,
    int contact_point_id,
    const RowVector3& dgap_dxw,
    RowVector6& dgap_dbody,
    RowVectorX& dgap_dp) {
    const Matrix3 R = body->_E_0i.topLeftCorner(3, 3);
    Matrix36 dxw_dbody;
    dxw_dbody.leftCols(3) = -R * math::skew(local_point);
    dxw_dbody.rightCols(3) = R;
    dgap_dbody = dgap_dxw * dxw_dbody;

    const Simulation* sim = body->_sim;
    dgap_dp = RowVectorX::Zero(sim->_ndof_p);

    for (auto ancestor = body->_joint; ancestor != nullptr; ancestor = ancestor->_parent) {
        if (!ancestor->_design_params_1._active) {
            continue;
        }
        for (int k = 0; k < ancestor->_design_params_1._ndof; ++k) {
            const int idx = ancestor->_design_params_1._param_index(k);
            const Matrix4& dE = body->_dE0i_dp1(idx);
            const Vector3 dxw =
                dE.topLeftCorner(3, 3) * local_point
                + dE.topRightCorner(3, 1);
            dgap_dp(idx) = dgap_dxw.dot(dxw);
        }
    }

    if (body->_design_params_2._active) {
        for (int k = 0; k < body->_design_params_2._ndof; ++k) {
            const int idx =
                sim->_ndof_p1 + body->_design_params_2._param_index(k);
            const Matrix4& dE = body->_dE0i_dp2(k);
            const Vector3 dxw =
                dE.topLeftCorner(3, 3) * local_point
                + dE.topRightCorner(3, 1);
            dgap_dp(idx) = dgap_dxw.dot(dxw);
        }
    }

    if (body->_design_params_3._active && contact_point_id >= 0) {
        const int local_offset = contact_point_id * 3;
        if (local_offset + 2 < body->_design_params_3._ndof) {
            for (int axis = 0; axis < 3; ++axis) {
                const int idx =
                    sim->_ndof_p1 + sim->_ndof_p2
                    + body->_design_params_3._param_index(local_offset + axis);
                dgap_dp(idx) = dgap_dxw.dot(R.col(axis));
            }
        }
    }
}

bool contact_pair_matches(
    const std::vector<std::pair<std::string, std::string>>& filters,
    const std::string& body1,
    const std::string& body2) {
    if (filters.empty()) {
        return true;
    }
    for (const auto& filter : filters) {
        const bool direct =
            (filter.first == "*" || filter.first == body1)
            && (filter.second == "*" || filter.second == body2);
        const bool reverse =
            (filter.first == "*" || filter.first == body2)
            && (filter.second == "*" || filter.second == body1);
        if (direct || reverse) {
            return true;
        }
    }
    return false;
}

std::string contact_pair_key(
    const std::string& contact_type,
    const std::string& body1,
    const std::string& body2) {
    return contact_type + "\n" + body1 + "\n" + body2;
}

}
