#include "Actuator/ActuatorMotor.h"
#include "Body/Body.h"
#include "Joint/Joint.h"
#include "Robot.h"
#include "Simulation.h"

namespace redmax {

namespace {

void collect_fixed_subtree_bodies(Joint* joint, std::vector<Body*>& bodies) {
    if (joint->_body != nullptr) {
        bodies.push_back(joint->_body);
    }
    for (Joint* child : joint->_children) {
        if (child->_ndof == 0) {
            collect_fixed_subtree_bodies(child, bodies);
        }
    }
}

}

ActuatorMotor::ActuatorMotor(std::string name, Joint* joint, ControlMode control_mode, VectorX ctrl_min, VectorX ctrl_max, VectorX ctrl_P, VectorX ctrl_D)
    : Actuator(name, joint->_ndof, ctrl_min, ctrl_max) {
    
    _joint = joint;
    _control_mode = control_mode;
    _ctrl_P = ctrl_P;
    _ctrl_D = ctrl_D;
}

ActuatorMotor::ActuatorMotor(std::string name, Joint* joint, ControlMode control_mode, dtype ctrl_min, dtype ctrl_max, dtype ctrl_P, dtype ctrl_D)
    : Actuator(name, joint->_ndof, ctrl_min, ctrl_max) {
    
    _joint = joint;
    _control_mode = control_mode;
    _ctrl_P = VectorX::Constant(_ndof, ctrl_P);
    _ctrl_D = VectorX::Constant(_ndof, ctrl_D);
}

void ActuatorMotor::update_states(const VectorX& dofs, const VectorX& dofs_vel) {
    for (int i = 0;i < _ndof;i++) {
        _dofs[i] = dofs[_joint->_index[i]];
        _dofs_vel[i] = dofs_vel[_joint->_index[i]];
    }
}

bool ActuatorMotor::uses_decoupled_wrench_control() const {
    return _control_mode == ControlMode::FORCE
        && _joint->_ndof == 6
        && _joint->uses_decoupled_wrench_control();
}

VectorX ActuatorMotor::bounded_force_command() const {
    VectorX u = _u.cwiseMin(VectorX::Ones(_joint->_ndof))
                    .cwiseMax(-VectorX::Ones(_joint->_ndof));
    return map_value(
        u,
        -VectorX::Ones(_joint->_ndof),
        VectorX::Ones(_joint->_ndof),
        _ctrl_min,
        _ctrl_max
    ).cwiseMin(_ctrl_max).cwiseMax(_ctrl_min);
}

Matrix6 ActuatorMotor::compute_decoupled_wrench_map(
    const MatrixX& J,
    const JacobianMatrixVector* dJ_dq,
    std::vector<Matrix6>* state_derivatives,
    const SparseJacobianMatrixVector* dJ_dp1,
    const SparseJacobianMatrixVector* dJ_dp2,
    std::vector<Matrix6>* design_derivatives) const {
    Simulation* sim = _joint->_sim;
    if (J.rows() != sim->_ndof_m || J.cols() != sim->_ndof_r) {
        throw_error(
            "Decoupled free3d wrench map requires the current joint Jacobian");
    }
    if (state_derivatives != nullptr) {
        if (dJ_dq == nullptr
            || dJ_dq->_data.size()
                != static_cast<std::size_t>(sim->_ndof_r)) {
            throw_error(
                "Decoupled free3d state derivatives require dJ/dq");
        }
        state_derivatives->assign(sim->_ndof_r, Matrix6::Zero());
    }
    std::vector<dtype> design_mass_derivatives;
    if (design_derivatives != nullptr) {
        if (dJ_dp1 == nullptr || dJ_dp2 == nullptr) {
            throw_error(
                "Decoupled free3d design derivatives require dJ/dp");
        }
        if (dJ_dp1->_data.size()
                != static_cast<std::size_t>(sim->_ndof_p1)
            || dJ_dp2->_data.size()
                != static_cast<std::size_t>(sim->_ndof_p2)) {
            throw_error(
                "Decoupled free3d received inconsistent dJ/dp dimensions");
        }
        design_derivatives->assign(sim->_ndof_p, Matrix6::Zero());
        design_mass_derivatives.assign(sim->_ndof_p, 0.);
    }

    std::vector<Body*> bodies;
    collect_fixed_subtree_bodies(_joint, bodies);
    if (bodies.empty()) {
        throw_error(
            "Decoupled free3d joint has no rigid body in its fixed subtree");
    }

    const int q0 = _joint->_index[0];
    Matrix6 mass_matrix = Matrix6::Zero();
    dtype total_mass = 0.;
    for (Body* body : bodies) {
        const int m0 = body->_index[0];
        const Matrix6 J_body = J.block<6, 6>(m0, q0);
        const Matrix6 inertia = body->_Inertia.asDiagonal();
        mass_matrix.noalias() += J_body.transpose() * inertia * J_body;
        total_mass += body->_mass;
        if (state_derivatives != nullptr) {
            for (int k = 0; k < sim->_ndof_r; ++k) {
                const Matrix6 dJ_body =
                    dJ_dq->_data[k].block<6, 6>(m0, q0);
                (*state_derivatives)[k].noalias() +=
                    dJ_body.transpose() * inertia * J_body
                    + J_body.transpose() * inertia * dJ_body;
            }
        }
        if (design_derivatives != nullptr) {
            for (int k = 0; k < sim->_ndof_p1; ++k) {
                Matrix6 dJ_body = Matrix6::Zero();
                dJ_body = dJ_dp1->_data[k].block(m0, q0, 6, 6);
                (*design_derivatives)[k].noalias() +=
                    dJ_body.transpose() * inertia * J_body
                    + J_body.transpose() * inertia * dJ_body;
            }
            for (int k = 0; k < sim->_ndof_p2; ++k) {
                Matrix6 dJ_body = Matrix6::Zero();
                dJ_body = dJ_dp2->_data[k].block(m0, q0, 6, 6);
                const int index = sim->_ndof_p1 + k;
                (*design_derivatives)[index].noalias() +=
                    dJ_body.transpose() * inertia * J_body
                    + J_body.transpose() * inertia * dJ_body;
            }
            if (body->_design_params_4._active) {
                const int p4_offset =
                    sim->_ndof_p1 + sim->_ndof_p2 + sim->_ndof_p3;
                const int p4_index =
                    body->_design_params_4._param_index[0];

                Matrix6 d_inertia = Matrix6::Zero();
                d_inertia.bottomRightCorner<3, 3>().setIdentity();
                const int mass_index = p4_offset + p4_index;
                (*design_derivatives)[mass_index].noalias() +=
                    J_body.transpose() * d_inertia * J_body;
                design_mass_derivatives[mass_index] += 1.;

                for (int k = 0; k < 3; ++k) {
                    d_inertia.setZero();
                    d_inertia(k, k) = 1.;
                    (*design_derivatives)[mass_index + 1 + k].noalias() +=
                        J_body.transpose() * d_inertia * J_body;
                }
            }
        }
    }
    if (!std::isfinite(total_mass) || total_mass <= 1e-12) {
        throw_error(
            "Decoupled free3d fixed subtree must have positive finite mass");
    }

    const Matrix3 rotational_mass = mass_matrix.bottomRightCorner<3, 3>();
    Eigen::LDLT<Matrix3> rotational_solver(rotational_mass);
    if (rotational_solver.info() != Eigen::Success
        || !rotational_solver.isPositive()) {
        throw_error(
            "Decoupled free3d fixed subtree has singular rotational inertia");
    }

    Matrix6 acceleration_map = Matrix6::Zero();
    acceleration_map.topLeftCorner<3, 3>() =
        Matrix3::Identity() / total_mass;
    acceleration_map.bottomRightCorner<3, 3>() =
        rotational_solver.solve(Matrix3::Identity());
    const Matrix6 wrench_map = mass_matrix * acceleration_map;

    const Matrix3 rotational_inverse =
        acceleration_map.bottomRightCorner<3, 3>();
    auto convert_mass_derivative =
        [&](const Matrix6& dM, dtype dmass) {
            Matrix6 dA = Matrix6::Zero();
            dA.topLeftCorner<3, 3>() =
                -Matrix3::Identity() * dmass
                / (total_mass * total_mass);
            dA.bottomRightCorner<3, 3>() =
                -rotational_inverse
                * dM.bottomRightCorner<3, 3>()
                * rotational_inverse;
            Matrix6 derivative =
                dM * acceleration_map + mass_matrix * dA;
            return derivative;
        };

    if (state_derivatives != nullptr) {
        for (int k = 0; k < sim->_ndof_r; ++k) {
            (*state_derivatives)[k] =
                convert_mass_derivative((*state_derivatives)[k], 0.);
        }
    }
    if (design_derivatives != nullptr) {
        for (int k = 0; k < sim->_ndof_p; ++k) {
            (*design_derivatives)[k] = convert_mass_derivative(
                (*design_derivatives)[k],
                design_mass_derivatives[k]);
        }
    }
    return wrench_map;
}

void ActuatorMotor::add_decoupled_force(VectorX& fr) {
    _fr = bounded_force_command();
    Simulation* sim = _joint->_sim;
    MatrixX J = MatrixX::Zero(sim->_ndof_m, sim->_ndof_r);
    MatrixX Jdot = MatrixX::Zero(sim->_ndof_m, sim->_ndof_r);
    sim->_robot->computeJointJacobian(J, Jdot);
    _decoupled_wrench_map =
        compute_decoupled_wrench_map(J);
    _decoupled_generalized_force = _decoupled_wrench_map * _fr;
    fr.segment<6>(_joint->_index[0]) += _decoupled_generalized_force;
}

void ActuatorMotor::computeForce(VectorX& fm, VectorX& fr) {
    if (_control_mode == ControlMode::FORCE) {
        if (uses_decoupled_wrench_control()) {
            add_decoupled_force(fr);
            return;
        }
        _fr = bounded_force_command();
        fr.segment(_joint->_index[0], _joint->_ndof) += _fr;
    } else {
        _pos_error = _u - _dofs;
        _vel_error = - _dofs_vel;
        _fr = (_ctrl_P.cwiseProduct(_pos_error) + _ctrl_D.cwiseProduct(_vel_error)).cwiseMin(_ctrl_max).cwiseMax(_ctrl_min);
        fr.segment(_joint->_index[0], _joint->_ndof) += _fr;
    }
}

void ActuatorMotor::computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr) {
    if (uses_decoupled_wrench_control()) {
        _fr = bounded_force_command();
        Simulation* sim = _joint->_sim;
        MatrixX J = MatrixX::Zero(sim->_ndof_m, sim->_ndof_r);
        MatrixX Jdot = MatrixX::Zero(sim->_ndof_m, sim->_ndof_r);
        JacobianMatrixVector dJ_dq(
            sim->_ndof_m, sim->_ndof_r, sim->_ndof_r);
        JacobianMatrixVector dJdot_dq(
            sim->_ndof_m, sim->_ndof_r, sim->_ndof_r);
        sim->_robot->computeJointJacobianWithDerivative(
            J, Jdot, dJ_dq, dJdot_dq);
        std::vector<Matrix6> derivatives;
        _decoupled_wrench_map =
            compute_decoupled_wrench_map(
                J,
                &dJ_dq,
                &derivatives);
        _decoupled_generalized_force =
            _decoupled_wrench_map * _fr;
        const int q0 = _joint->_index[0];
        fr.segment<6>(q0) += _decoupled_generalized_force;
        for (int k = 0; k < _joint->_sim->_ndof_r; ++k) {
            Kr.block<6, 1>(q0, k).noalias() +=
                derivatives[k] * _fr;
        }
        return;
    }
    computeForce(fm, fr);
}

void ActuatorMotor::computeForceWithDerivative(
    VectorX& fm, VectorX& fr,
    MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr,
    MatrixX& dfr_dp,
    const MatrixX& J,
    const JacobianMatrixVector& dJ_dq,
    const SparseJacobianMatrixVector& dJ_dp1,
    const SparseJacobianMatrixVector& dJ_dp2) {
    if (!uses_decoupled_wrench_control()) {
        computeForceWithDerivative(fm, fr, Km, Dm, Kr, Dr);
        return;
    }

    _fr = bounded_force_command();
    std::vector<Matrix6> state_derivatives;
    std::vector<Matrix6> design_derivatives;
    _decoupled_wrench_map = compute_decoupled_wrench_map(
        J,
        &dJ_dq,
        &state_derivatives,
        &dJ_dp1,
        &dJ_dp2,
        &design_derivatives);
    _decoupled_generalized_force = _decoupled_wrench_map * _fr;
    const int q0 = _joint->_index[0];
    fr.segment<6>(q0) += _decoupled_generalized_force;
    for (int k = 0; k < _joint->_sim->_ndof_r; ++k) {
        Kr.block<6, 1>(q0, k).noalias() +=
            state_derivatives[k] * _fr;
    }
    for (int k = 0; k < _joint->_sim->_ndof_p; ++k) {
        dfr_dp.block<6, 1>(q0, k).noalias() +=
            design_derivatives[k] * _fr;
    }
}

void ActuatorMotor::compute_dfdu(MatrixX& dfm_du, MatrixX& dfr_du) {
    if (_control_mode == ControlMode::FORCE) {
        if (uses_decoupled_wrench_control()) {
            for (int i = 0; i < _joint->_ndof; ++i) {
                if (_u[i] >= -1. && _u[i] <= 1.) {
                    dfr_du.block<6, 1>(
                        _joint->_index[0], _index[i])
                        += _decoupled_wrench_map.col(i)
                        * ((_ctrl_max[i] - _ctrl_min[i]) / 2.);
                }
            }
            return;
        }
        for (int i = 0;i < _joint->_ndof;i++) {
            if (_u[i] >= -1. && _u[i] <= 1.) {
                dfr_du(_joint->_index[i], _index[i]) += (_ctrl_max[i] - _ctrl_min[i]) / 2.;
            }
        }
    } else {
        for (int i = 0;i < _joint->_ndof;i++) {
            dtype f = _ctrl_P[i] * _pos_error[i] + _ctrl_D[i] * _vel_error[i];
            if (f >= _ctrl_min[i] && f <= _ctrl_max[i])
                dfr_du(_joint->_index[i], _index[i]) += _ctrl_P[i];
        }
    }
}

void ActuatorMotor::compute_extra_derivatives(MatrixX& dfm_dqprev, MatrixX& dfm_dqdotprev, MatrixX& dfr_dqprev, MatrixX& dfr_dqdotprev) {
    if (_control_mode == ControlMode::POS) {
        for (int i = 0;i < _joint->_ndof;i++) {
            dtype f = _ctrl_P[i] * _pos_error[i] + _ctrl_D[i] * _vel_error[i];
            if (f >= _ctrl_min[i] && f <= _ctrl_max[i]) {
                dfr_dqprev(_joint->_index[i], _index[i]) -= _ctrl_P[i];
                dfr_dqdotprev(_joint->_index[i], _index[i]) -= _ctrl_D[i];
            }
        }
    }
}
}
