#pragma once

#include "Common.h"

namespace redmax {

class Body;

struct ContactPointMetric {
    int contact_point_id = -1;
    dtype gap = std::numeric_limits<dtype>::infinity();
    dtype elastic_force = 0.;
    dtype total_normal_force = 0.;
    Vector3 world_position = Vector3::Zero();
    Vector3 normal = Vector3::Zero();
    RowVectorX dgap_dq;
    RowVectorX dgap_dp;
    RowVectorX delastic_dq;
    RowVectorX delastic_dp;
};

struct ContactPairMetric {
    std::string contact_type;
    std::string body1;
    std::string body2;
    int sample_count = 0;
    dtype min_gap = std::numeric_limits<dtype>::infinity();
    dtype max_penetration = 0.;
    dtype activation = 0.;
    dtype elastic_normal_force = 0.;
    dtype total_normal_force = 0.;
    bool in_activation_band = false;
    bool geometrically_touching = false;
    bool force_active = false;
    std::vector<int> contact_point_ids;
    std::vector<Vector3> world_positions;
    std::vector<Vector3> normals;
    VectorX dactivation_dq;
    VectorX dactivation_dp;
    VectorX delastic_force_dq;
    VectorX delastic_force_dp;
};

struct ContactPairSummary {
    std::string contact_type;
    std::string body1;
    std::string body2;
    int observed_substeps = 0;
    int active_substeps = 0;
    int first_active_step = -1;
    int last_active_step = -1;
    dtype min_gap = std::numeric_limits<dtype>::infinity();
    dtype max_penetration = 0.;
    dtype peak_elastic_normal_force = 0.;
    dtype peak_total_normal_force = 0.;
    dtype normal_impulse = 0.;
    bool ever_geometrically_touching = false;
    std::vector<int> contact_point_ids;
};

dtype contact_metric_sigmoid(dtype value);
dtype contact_metric_softplus(dtype value);

ContactPairMetric aggregate_contact_pair_metric(
    const std::string& contact_type,
    const std::string& body1,
    const std::string& body2,
    const std::vector<ContactPointMetric>& points,
    dtype smoothing,
    dtype force_threshold,
    int ndof_q,
    int ndof_p);

void body_point_gap_derivatives(
    const Body* body,
    const Vector3& local_point,
    int contact_point_id,
    const RowVector3& dgap_dxw,
    RowVector6& dgap_dbody,
    RowVectorX& dgap_dp);

bool contact_pair_matches(
    const std::vector<std::pair<std::string, std::string>>& filters,
    const std::string& body1,
    const std::string& body2);

std::string contact_pair_key(
    const std::string& contact_type,
    const std::string& body1,
    const std::string& body2);

}
