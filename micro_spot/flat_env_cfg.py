# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.micro_spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_tasks.manager_based.locomotion.velocity.config.micro_spot.robot_spot_ean import QUAD_EAN  # isort: skip

hxy = [
    "front_left_shoulder",
    "front_right_shoulder",
    "rear_left_shoulder",
    "rear_right_shoulder",
    "front_left_leg",
    "front_right_leg",
    "rear_left_leg",
    "rear_right_leg"
]


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
import math


@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    #base_velocity = mdp.UniformVelocityCommandCfg(
        #asset_name="robot",
        #resampling_time_range=(10.0, 10.0),
        #rel_standing_envs=0.1,
        #rel_heading_envs=0.0,
        #heading_command=True,
        #debug_vis=True,
        #ranges=mdp.UniformVelocityCommandCfg.Ranges(
            #lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
        #),
    #)

    
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=.5,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-2.0, 2.0), ang_vel_z=(0.,0.), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (12., 12.),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


#@configclass
#class SpotRewardsCfg:
    ## -- task
    #air_time = RewardTermCfg(
        #func=spot_mdp.air_time_reward,
        #weight=5.0,
        #params={
            #"mode_time": 0.3,
            #"velocity_threshold": 0.5,
            #"asset_cfg": SceneEntityCfg("robot"),
            #"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
        #},
    #)
    #base_angular_velocity = RewardTermCfg(
        #func=spot_mdp.base_angular_velocity_reward,
        #weight=10.0,
        #params={"std": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    #)
    #base_linear_velocity = RewardTermCfg(
        #func=spot_mdp.base_linear_velocity_reward,
        #weight=10.0,
        #params={"std": 0.5, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    #)
    #foot_clearance = RewardTermCfg(
        #func=spot_mdp.foot_clearance_reward,
        #weight=0.5,
        #params={
            #"std": 0.05,
            #"tanh_mult": 2.0,
            #"target_height": 0.1,
            #"asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_link"),
        #},
    #)
    #gait = RewardTermCfg(
        #func=spot_mdp.GaitReward,
        #weight=10.0,
        #params={
            #"std": 0.1,
            #"max_err": 0.2,
            #"velocity_threshold": 0.5,
            #"synced_feet_pair_names": (("front_left_toe_link", "rear_right_toe_link"), ("front_right_toe_link", "rear_left_toe_link")),
            #"asset_cfg": SceneEntityCfg("robot"),
            #"sensor_cfg": SceneEntityCfg("contact_forces"),
        #},
    #)

    ## -- penalties
    #action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    #air_time_variance = RewardTermCfg(
        #func=spot_mdp.air_time_variance_penalty,
        #weight=-1.0,
        #params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link")},
    #)
    #base_motion = RewardTermCfg(
        #func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    #)
    #base_orientation = RewardTermCfg(
        #func=spot_mdp.base_orientation_penalty, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot")}
    #)
    #foot_slip = RewardTermCfg(
        #func=spot_mdp.foot_slip_penalty,
        #weight=-0.5,
        #params={
            #"asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_link"),
            #"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            #"threshold": 1.0,
        #},
    #)
    #joint_acc = RewardTermCfg(
        #func=spot_mdp.joint_acceleration_penalty,
        #weight=-5.0e-5,
        #params={"asset_cfg": SceneEntityCfg("robot", joint_names=hxy)},
    #)
    #joint_pos = RewardTermCfg(
        #func=spot_mdp.joint_position_penalty,
        #weight=-1.5,
        #params={
            #"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            #"stand_still_scale": 10.0,
            #"velocity_threshold": 0.5,
        #},
    #)
    #joint_torques = RewardTermCfg(
        #func=spot_mdp.joint_torques_penalty,
        #weight=-2.0e-4,
        #params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    #)
    #joint_vel = RewardTermCfg(
        #func=spot_mdp.joint_velocity_penalty,
        #weight=-5.0e-3,
        #params={"asset_cfg": SceneEntityCfg("robot", joint_names=hxy)},
    #)

    #low_height = RewardTermCfg(
        #func=spot_mdp.low_height_penalty,
        #weight=-5.0,
        #params={"min_height": 0.3, "scale": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    #)

@configclass
class SpotRewardsCfg:
    # -- task rewards
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=10.0,  # Tăng từ 5.0 để khuyến khích chu kỳ dáng đi năng động
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=15.0,  # Tăng từ 5.0 để ưu tiên theo dõi lệnh xoay
        params={"std": 1.5, "asset_cfg": SceneEntityCfg("robot")},  # Giảm std để tăng độ nhạy
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=15.0,  # Tăng từ 5.0 để ưu tiên theo dõi lệnh di chuyển
        params={"std": 0.5, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},  # Giảm std
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=1.0,  # Tăng từ 0.5 để khuyến khích nâng chân cao
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.12,  # Tăng từ 0.1 để yêu cầu nâng chân cao hơn
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_link"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=12.0,  # Tăng từ 10.0 để ưu tiên dáng đi đồng bộ
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("front_left_toe_link", "rear_right_toe_link"), ("front_right_toe_link", "rear_left_toe_link")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(
        func=spot_mdp.action_smoothness_penalty,
        weight=-0.5,  # Giảm từ -1.0 để cho phép chuyển động năng động hơn
    )
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-0.5,  # Giảm từ -1.0 để tránh phạt quá mạnh
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-6.0,  # Tăng từ -3.0 để ngăn nghiêng hoặc hạ thấp
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-2.0e-5,  # Giảm từ -1.0e-4 để cho phép chuyển động nhanh hơn
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.5,  # Tăng từ -0.7 để ưu tiên tư thế đứng
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 1.0,  # Tăng từ 5.0 để phạt mạnh khi không ở tư thế đứng
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-1.0e-4,  # Giảm từ -5.0e-4 để giảm ưu tiên tiết kiệm năng lượng
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-2.0e-3,  # Giảm từ -1.0e-2 để cho phép vận tốc khớp cao hơn
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    low_height = RewardTermCfg(
        func=spot_mdp.low_height_penalty,
        weight=-8.0,  # Phạt mạnh khi hạ thấp trọng tâm
        params={"min_height": 0.35, "scale": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    no_significant_movement = RewardTermCfg(
        func=spot_mdp.no_significant_movement_penalty,
        weight=-50.0,  # Phạt mạnh khi không di chuyển đáng kể
        params={"min_vel": 0.3, "scale": 5.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    wrong_direction = RewardTermCfg(
        func=spot_mdp.wrong_direction_penalty,
        weight=-15.0,  # Trọng số âm để phạt, điều chỉnh dựa trên mức độ ưu tiên
        params={
            "std": 0.5,  # Độ lệch chuẩn, điều chỉnh để kiểm soát độ nhạy của phạt
            "min_cmd_vel": 0.1,  # Ngưỡng vận tốc lệnh tối thiểu
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*leg_link", ".*foot_link"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class SpotFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Spot-d
        self.scene.robot = QUAD_EAN.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        # no height scan
        self.scene.height_scanner = None


class SpotFlatEnvCfg_PLAY(SpotFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
