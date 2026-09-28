#pragma once

#include "Joint/JointFree3DExp.h"

namespace redmax {

class JointFree3DExpDecoupled : public JointFree3DExp {
public:
    JointFree3DExpDecoupled(
        Simulation *sim,
        int id,
        Joint *parent,
        Matrix3 R_pj_0,
        Vector3 p_pj_0,
        Joint::Frame frame = Joint::Frame::LOCAL)
        : JointFree3DExp(sim, id, parent, R_pj_0, p_pj_0, frame) {}

    bool uses_decoupled_wrench_control() const override { return true; }
};

}
