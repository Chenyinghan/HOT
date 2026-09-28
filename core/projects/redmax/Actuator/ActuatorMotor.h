#pragma once
#include "Actuator/Actuator.h"

namespace redmax {

class ActuatorMotor : public Actuator {
public:
    enum ControlMode {
        FORCE, // input to FORCE mode is [-1, 1] mapping to [ctrl_min, ctrl_max]
        POS // input to POS mode is target_dof, the torque is P * (target_dof - current_dof) + D * (target_dof_vel - current_dof_vel)
    };

    ControlMode _control_mode; // the control mode
    VectorX _ctrl_P, _ctrl_D;
    // stored temporary variables
    VectorX _pos_error, _vel_error;
    Matrix6 _decoupled_wrench_map = Matrix6::Identity();
    Vector6 _decoupled_generalized_force = Vector6::Zero();

    ActuatorMotor(std::string name, Joint* joint, ControlMode control_mode, 
                    VectorX ctrl_min, VectorX ctrl_max, 
                    VectorX ctrl_P = VectorX::Zero(1), VectorX ctrl_D = VectorX::Zero(1));
    ActuatorMotor(std::string name, Joint* joint, ControlMode control_mode,
                    dtype ctrl_min, dtype ctrl_max,
                    dtype ctrl_P = 0., dtype ctrl_D = 0.);

    void update_states(const VectorX& dofs, const VectorX& dofs_vel);

    void computeForce(VectorX& fm, VectorX& fr);
    void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr);
    void computeForceWithDerivative(
        VectorX& fm, VectorX& fr,
        MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr,
        MatrixX& dfr_dp,
        const MatrixX& J,
        const JacobianMatrixVector& dJ_dq,
        const SparseJacobianMatrixVector& dJ_dp1,
        const SparseJacobianMatrixVector& dJ_dp2);

    void compute_dfdu(MatrixX& dfm_du, MatrixX& dfr_du);
    void compute_extra_derivatives(MatrixX& dfm_dqprev, MatrixX& dfm_dqdotprev, MatrixX& dfr_dqprev, MatrixX& dfr_dqdotprev);

private:
    bool uses_decoupled_wrench_control() const;
    VectorX bounded_force_command() const;
    Matrix6 compute_decoupled_wrench_map(
        const MatrixX& J,
        const JacobianMatrixVector* dJ_dq = nullptr,
        std::vector<Matrix6>* state_derivatives = nullptr,
        const SparseJacobianMatrixVector* dJ_dp1 = nullptr,
        const SparseJacobianMatrixVector* dJ_dp2 = nullptr,
        std::vector<Matrix6>* design_derivatives = nullptr) const;
    void add_decoupled_force(VectorX& fr);
};

}
