#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "Simulation.h"
#include "SimEnvGenerator.h"

using namespace redmax;

namespace py = pybind11;

Simulation* make_sim(std::string env_name, std::string integrator = "BDF2") {
    Simulation* sim = nullptr;
    if (env_name == "SinglePendulum-Test") {
        sim = SimEnvGenerator::createSinglePendulumTest(integrator);
    } else if (env_name == "Prismatic-Test") {
        sim = SimEnvGenerator::createPrismaticTest(integrator);
    } else if (env_name == "Free2D-Test") {
        sim = SimEnvGenerator::createFree2DTest(integrator);
    } else if (env_name == "GroundContact-Test") {
        sim = SimEnvGenerator::createGroundContactTest(integrator);
    } else if (env_name == "BoxContact-Test") {
        sim = SimEnvGenerator::createBoxContactTest(integrator);
    } else if (env_name == "TorqueFingerFlick-Demo") {
        sim = SimEnvGenerator::createTorqueFingerFlickDemo(integrator);
    } else if (env_name == "TorqueFinger-Demo") {
        sim = SimEnvGenerator::createTorqueFingerDemo(integrator);
    } else {
        return NULL;
    }
    return sim;
}

PYBIND11_MODULE(redmax_py, m) {
    py::class_<ContactPairMetric>(m, "ContactPairMetric")
        .def_readonly("contact_type", &ContactPairMetric::contact_type)
        .def_readonly("body1", &ContactPairMetric::body1)
        .def_readonly("body2", &ContactPairMetric::body2)
        .def_readonly("sample_count", &ContactPairMetric::sample_count)
        .def_readonly("min_gap", &ContactPairMetric::min_gap)
        .def_readonly("max_penetration", &ContactPairMetric::max_penetration)
        .def_readonly("activation", &ContactPairMetric::activation)
        .def_readonly(
            "elastic_normal_force",
            &ContactPairMetric::elastic_normal_force)
        .def_readonly(
            "total_normal_force",
            &ContactPairMetric::total_normal_force)
        .def_readonly(
            "in_activation_band",
            &ContactPairMetric::in_activation_band)
        .def_readonly(
            "geometrically_touching",
            &ContactPairMetric::geometrically_touching)
        .def_readonly("force_active", &ContactPairMetric::force_active)
        .def_readonly(
            "contact_point_ids",
            &ContactPairMetric::contact_point_ids)
        .def_readonly("world_positions", &ContactPairMetric::world_positions)
        .def_readonly("normals", &ContactPairMetric::normals)
        .def_readonly("dactivation_dq", &ContactPairMetric::dactivation_dq)
        .def_readonly("dactivation_dp", &ContactPairMetric::dactivation_dp)
        .def_readonly(
            "delastic_force_dq",
            &ContactPairMetric::delastic_force_dq)
        .def_readonly(
            "delastic_force_dp",
            &ContactPairMetric::delastic_force_dp);

    py::class_<ContactPairSummary>(m, "ContactPairSummary")
        .def_readonly("contact_type", &ContactPairSummary::contact_type)
        .def_readonly("body1", &ContactPairSummary::body1)
        .def_readonly("body2", &ContactPairSummary::body2)
        .def_readonly(
            "observed_substeps",
            &ContactPairSummary::observed_substeps)
        .def_readonly("active_substeps", &ContactPairSummary::active_substeps)
        .def_readonly(
            "first_active_step",
            &ContactPairSummary::first_active_step)
        .def_readonly(
            "last_active_step",
            &ContactPairSummary::last_active_step)
        .def_readonly("min_gap", &ContactPairSummary::min_gap)
        .def_readonly(
            "max_penetration",
            &ContactPairSummary::max_penetration)
        .def_readonly(
            "peak_elastic_normal_force",
            &ContactPairSummary::peak_elastic_normal_force)
        .def_readonly(
            "peak_total_normal_force",
            &ContactPairSummary::peak_total_normal_force)
        .def_readonly("normal_impulse", &ContactPairSummary::normal_impulse)
        .def_readonly(
            "ever_geometrically_touching",
            &ContactPairSummary::ever_geometrically_touching)
        .def_readonly(
            "contact_point_ids",
            &ContactPairSummary::contact_point_ids);


    // simulation options
    py::class_<Simulation::Options>(m, "Options")
        .def(py::init<Vector3, dtype, string>(), 
                py::arg("gravity") = -980. * Vector3::UnitZ(), 
                py::arg("h") = 0.02, 
                py::arg("integrator") = "BDF1")
                
        .def_readwrite("gravity", &Simulation::Options::_gravity)
        .def_readwrite("h", &Simulation::Options::_h)
        .def_readwrite("integrator", &Simulation::Options::_integrator);
    
    // viewer options
    py::class_<Simulation::ViewerOptions>(m, "ViewerOptions")
        .def(py::init())

        .def_readwrite("fps", &Simulation::ViewerOptions::_fps)
        .def_readwrite("speed", &Simulation::ViewerOptions::_speed)
        .def_readwrite("camera_pos", &Simulation::ViewerOptions::_camera_pos)
        .def_readwrite("camera_up", &Simulation::ViewerOptions::_camera_up)
        .def_readwrite("camera_lookat", &Simulation::ViewerOptions::_camera_lookat)
        .def_readwrite("background_color_rgba", &Simulation::ViewerOptions::_background_color_rgba)
        .def_readwrite("ground", &Simulation::ViewerOptions::_ground)
        .def_readwrite("E_g", &Simulation::ViewerOptions::_E_g)
        .def_readwrite("record", &Simulation::ViewerOptions::_record)
        .def_readwrite("record_folder", &Simulation::ViewerOptions::_record_folder)
        .def_readwrite("loop", &Simulation::ViewerOptions::_loop)
        .def_readwrite("infinite", &Simulation::ViewerOptions::_infinite);

    // backward data
    py::class_<BackwardInfo>(m, "BackwardInfo")
        .def_readwrite("df_dq0", &BackwardInfo::_df_dq0)
        .def_readwrite("df_dqdot0", &BackwardInfo::_df_dqdot0)
        .def_readwrite("df_dp", &BackwardInfo::_df_dp)
        .def_readwrite("df_du", &BackwardInfo::_df_du)
        .def_readwrite("df_dq", &BackwardInfo::_df_dq)
        .def_readwrite("df_dvar", &BackwardInfo::_df_dvar)
        .def_readwrite("df_dtactile", &BackwardInfo::_df_dtactile)
        .def_readwrite("q_his", &BackwardInfo::_q_his)

        .def("set_flags", &BackwardInfo::set_flags,
                "set the flags for backward", 
                py::arg("flag_q0"), py::arg("flag_qdot0"), py::arg("flag_p"), py::arg("flag_u"));

    // backward results
    py::class_<BackwardResults>(m, "BackwardResults")
        .def_readonly("df_dq0", &BackwardResults::_df_dq0)
        .def_readonly("df_dqdot0", &BackwardResults::_df_dqdot0)
        .def_readonly("df_dp", &BackwardResults::_df_dp)
        .def_readonly("df_du", &BackwardResults::_df_du);

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<std::string, bool>(), 
                py::arg("xml_file_path"), 
                py::arg("verbose") = false)

        .def_readonly("options", &Simulation::_options)
        .def_readonly("viewer_options", &Simulation::_viewer_options)

        .def_readonly("ndof_r", &Simulation::_ndof_r)
        .def_readonly("ndof_m", &Simulation::_ndof_m)
        .def_readonly("ndof_p", &Simulation::_ndof_p)
        .def_readonly("ndof_u", &Simulation::_ndof_u)
        .def_readonly("ndof_var", &Simulation::_ndof_var)
        .def_readonly("ndof_tactile", &Simulation::_ndof_tactile)

        .def("init", &Simulation::init,
                "initialize the simulation")
        .def("get_q_init", &Simulation::get_q_init,
                "get init q for simulation")
        .def("get_qdot_init", &Simulation::get_qdot_init,
                "get init qdot for simulation")
        .def("set_state_init", &Simulation::set_state_init, 
                "set init state (q, qdot) for simulation", 
                py::arg("q_init"), py::arg("qdot_init"))
        .def("set_q_init", &Simulation::set_q_init, 
                "set init q for simulation",
                py::arg("q_init"))
        .def("set_qdot_init", &Simulation::set_qdot_init, 
                "set init qdot for simulation",
                py::arg("qdot_init"))
        .def_property_readonly("q_init", &Simulation::get_q_init)
        .def_property_readonly("qdot_init", &Simulation::get_qdot_init)

        .def("get_q", &Simulation::get_q,
                "get q")
        .def("get_qdot", &Simulation::get_qdot,
                "get qdot")
        .def("set_state", &Simulation::set_state,
                "set current generalized coordinates and velocities",
                py::arg("q"), py::arg("qdot"))
        .def("set_q", &Simulation::set_q,
                "set current generalized coordinates",
                py::arg("q"))
        .def("set_qdot", &Simulation::set_qdot,
                "set current generalized velocities",
                py::arg("qdot"))
        .def("get_variables", &Simulation::get_variables,
                "get variables")
        .def("get_design_params", &Simulation::get_design_params,
                "get design parameters")
        .def_property_readonly("q", &Simulation::get_q)
        .def_property_readonly("qdot", &Simulation::get_qdot)
        .def_property_readonly("variables", &Simulation::get_variables)
        .def_property_readonly("design_params", &Simulation::get_design_params)

        .def("set_design_params", &Simulation::set_design_params,
                "set design parameters",
                py::arg("design_params"))

        .def("set_u", &Simulation::set_u,
                "set u",
                py::arg("u"))
        .def("get_ctrl_range", &Simulation::get_ctrl_range, 
                "get the range of the ctrl",
                py::arg("ctrl_min"), py::arg("ctrl_max"))
        .def("get_ctrl_force", &Simulation::get_ctrl_force,
                "get the force of the control motors")

        .def("print_ctrl_info", &Simulation::print_ctrl_info)
        .def("print_design_params_info", &Simulation::print_design_params_info)

        .def("get_tactile_depth", &Simulation::get_tactile_depth,
                "get the tactile depth given the tactile name, shape (ndof_tactile(name), )",
                py::arg("name"))
        .def("get_tactile_normal_force", &Simulation::get_tactile_normal_force,
                "get the tactile normal force given the tactile name, shape (ndof_tactile(name), )",
                py::arg("name"))
        .def("get_tactile_shear_force", &Simulation::get_tactile_shear_force,
                "get the tactile shear force given the tactile name, shape (ndof_tactile(name), 2)",
                py::arg("name"))
        .def("get_tactile_force", &Simulation::get_tactile_force,
                "get the tactile force given the tactile name, shape (ndof_tactile(name), 3)",
                py::arg("name"))
        .def("get_tactile_force_vector", &Simulation::get_tactile_force_vector,
                "get the tactile force of all tactile sensors, shape (ndof_tactile(name) * 3, )")
        .def("get_tactile_image_pos", &Simulation::get_tactile_image_pos,
                "get the tactile image positions given the tactile name, shape (ndof_tactile(name), 2)",
                py::arg("name"))
        .def("get_tactile_flow_images", &Simulation::get_tactile_flow_images,
                "get the tactile flow images of all sensors, shape (ndof_tactile, rows, cols, 3)")
        .def("get_tactile_sensor_pos", &Simulation::get_tactile_sensor_pos,
                "get the position of each marker in tactile sensor local frame, shape (ndof_tactile(name), 2)",
                py::arg("name"))

        .def("set_contact_scale", &Simulation::set_contact_scale,
            "set the contact scale for continuation method",
            py::arg("scale"))
        .def(
            "get_contact_pair_metrics",
            &Simulation::get_contact_pair_metrics,
            "return current contact-pair metrics at the accepted state",
            py::arg("filters") =
                std::vector<std::pair<std::string, std::string>>(),
            py::arg("derivatives") = false,
            py::arg("ground_metric_smoothing") = 0.02,
            py::arg("force_threshold") = 1e-6)
        .def(
            "set_contact_tracking",
            &Simulation::set_contact_tracking,
            "enable accepted-substep contact tracking",
            py::arg("enabled"),
            py::arg("filters") =
                std::vector<std::pair<std::string, std::string>>(),
            py::arg("ground_metric_smoothing") = 0.02,
            py::arg("force_threshold") = 1e-6)
        .def(
            "clear_contact_history",
            &Simulation::clear_contact_history,
            "clear accepted-substep contact history")
        .def(
            "get_contact_summary",
            &Simulation::get_contact_summary,
            "return accepted-substep contact summaries")
        .def("get_solver_diagnostics", &Simulation::get_solver_diagnostics,
            "return implicit solver diagnostics for the current rollout")
            
        .def("set_rendering_mesh_vertices", &Simulation::set_rendering_mesh_vertices,
            "set the rendering mesh vertices for abstract body",
            py::arg("Vs"))
        .def("set_rendering_mesh", &Simulation::set_rendering_mesh,
            "set the rendering mesh for abstract body",
            py::arg("Vs"), py::arg("Fs"))
        
        .def("update_virtual_object", &Simulation::update_virtual_object,
                "update the properties of the virtual objects by name",
                py::arg("name"), py::arg("data"))

        .def("update_contact_parameters", &Simulation::update_contact_parameters,
                "update the parameters of general primitive contact",
                py::arg("body1"), py::arg("body2"), py::arg("kn"), py::arg("kt"), py::arg("mu"), py::arg("damping"))
        .def("update_tactile_parameters", &Simulation::update_tactile_parameters,
                "update the parameters of tactile sensor",
                py::arg("name"), py::arg("kn"), py::arg("kt"), py::arg("mu"), py::arg("damping"))
        .def("update_body_density", &Simulation::update_body_density,
                "update the density of body by its name",
                py::arg("body_name"), py::arg("density"))
        .def("update_body_color", &Simulation::update_body_color,
                "update the color of body by its name",
                py::arg("body_name"), py::arg("color"))
        .def("update_body_size", &Simulation::update_body_size,
                "update the size of body by its name (only support cuboid now)",
                py::arg("body_name"), py::arg("body_size"))
        .def("update_joint_damping", &Simulation::update_joint_damping,
                "update the damping of joint by its name",
                py::arg("joint_name"), py::arg("damping"))
        .def("update_tactile_sensor_pos", &Simulation::update_tactile_sensor_pos,
                "update the marker position of tactile sensor",
                py::arg("name"), py::arg("new_pos"))
        .def("update_joint_location", &Simulation::update_joint_location,
                "update the joint_location of joint by its name",
                py::arg("joint_name"), py::arg("joint_location"))
        .def("update_endeffector_position", &Simulation::update_endeffector_position,
                "update the position of the endeffector (in joint frame)",
                py::arg("endeffector_name"), py::arg("position"))

        .def_readonly("backward_info", &Simulation::_backward_info)
        .def_readonly("backward_results", &Simulation::_backward_results)

        .def("reset", &Simulation::reset,
                "reset the simulation.",
                py::arg("backward_flag") = false, py::arg("backward_design_params_flag") = false)
                
        .def("clearBackwardCache", &Simulation::clearBackwardCache,
                "clear the backward cache.")
        .def("saveBackwardCache", &Simulation::saveBackwardCache,
                "save the backward info in to cache.")
        .def("popBackwardCache", &Simulation::popBackwardCache,
                "pop up the backward info at the end of the cache.")
        .def("backwardCacheSize", &Simulation::backwardCacheSize,
                "get the size of the backward cache")

        .def("forward", &Simulation::forward,
                "step forward the simulation.",
                py::arg("num_steps"), py::arg("verbose") = false, py::arg("test_derivatives") = false, py::arg("save_last_frame_var_only") = false)
        .def("backward", &Simulation::backward,
                "backward differentiation.")
        .def("backward_steps", &Simulation::backward_steps,
                "backward differentiation for a specified number of steps",
                py::arg("num_backward_steps"))
                
        .def("replay", &Simulation::replay,
                "replay (render) the simulation.")

        .def("export_replay", &Simulation::export_replay,
                "export the body transformations in each step.",
                py::arg("folder"))

        .def("print_time_report", &Simulation::print_time_report,
                "print time report of the simulation.");

    m.def("make_sim", &make_sim, "initialize a simulation instance", py::arg("env_name"), py::arg("integrator") = "BDF2");
}
