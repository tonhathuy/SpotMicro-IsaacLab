import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

SPOTMICRO_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.environ['HOME'] + "/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spotmicro_quadruped/spot_micro/spot_micro.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, # added this line
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.719),
        joint_pos={
            ".*_shoulder": 0.0,
            ".*_leg": 0.5,
            ".*_foot": -1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_leg", ".*_foot", ".*_shoulder"],
            stiffness    = 3.4 / 0.35,        # ≈ 10 N·m/rad  → có thể nâng tới ~40 khi robot nặng
            damping      = 0.05 * 40.0,       # ≈ 2 N·m·s/rad  (5 % k‑p) – bắt đầu ít, tăng dần
            effort_limit = 3.5,               # N·m  (ngang stall‑torque 6 V)
            velocity_limit = 8.0              # rad/s (> 7.5 để không kìm tốc độ tối đa)
        ),
        # "base_legs": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_leg", ".*_foot", ".*_shoulder"],   # regex khớp tên hinge
        #     stiffness=20.0,                    # k_p  (N·m/rad)
        #     damping=0.5,                       # k_d  (N·m·s/rad)
        #     effort_limit=3.4,                  # τ_max
        #     velocity_limit=8.0,                # |ω|
        #     armature=0.0,                      # giữ mặc định
        # )
    },
)