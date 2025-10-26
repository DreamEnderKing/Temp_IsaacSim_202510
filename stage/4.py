from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

from time import time, sleep
import sys
import queue

import numpy as np
import torch
import quaternion
from isaacsim.core.api import World
import isaacsim.core.api.objects as obj
import isaacsim.core.prims as prims
from isaacsim.core.api.materials import OmniPBR, OmniGlass, PhysicsMaterial
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

curPath = "/root/workspace/main/temp/Temp_IsaacSim_202510"

class Task_4(BaseTask):
    def __init__(
        self, 
        world: World,
        name: str,
        source_pos: np.ndarray,
        target_pos: np.ndarray,
        size: float = 0.02,
        tolerance: float = 0.01,
        slow_rate: int = 1,
        obstacle: bool = False,
        logger = None,
        offset = None
    ):
        self.world = world
        self._name = name
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.size = size
        self.tolerance = tolerance
        self.slow_rate = slow_rate
        self.obstacle = obstacle
        self.logger = logger

        self.target_prim = None
        self.robot_prim = None
        self.robot_solver = None
        self.visual_ball = None

        self.sub = "move_source"
        self.action = queue.Queue()
        self.first_call = True
        self.temp_hold = {
            "start": None,
            "gripper": 0.0,
            "steady": False
        }

        self.__curobo_init__()

    def __curobo_init__(self):
        self.tensor_args = TensorDeviceType()
        config = load_yaml(join_path(curPath, "assets/franka.yml"))["robot_cfg"]["kinematics"]
        generator = CudaRobotGeneratorConfig(
            external_asset_path=join_path(curPath, "assets"),
            external_robot_configs_path=join_path(curPath, "assets"),
            load_link_names_with_mesh=True,
            tensor_args=self.tensor_args, 
            **config
        )
        self.robot_cfg = RobotConfig(CudaRobotModelConfig.from_config(generator), self.tensor_args)
        self.fk_model = CudaRobotModel(self.robot_cfg.kinematics)
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(my_world.stage)
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            max_attempts=5,
        )

    def print(self, msg):
        if self.logger:
            self.logger.write(msg + "\n")
        else:
            print(msg)

    def set_up_scene(self, scene):
        # add ground plane
        add_reference_to_stage(
            usd_path="assets/ground_plane.usd",
            prim_path="/World/GroundPlane"
        )
        ground_material = PhysicsMaterial(
            prim_path="/World/Physics/GroundPlane",
            static_friction=0,
            dynamic_friction=0,
            restitution=1.0
        )
        ground = obj.GroundPlane(
            prim_path="/World/GroundPlane",
            name="GroundPlane",
            z_position=0,
            physics_material=ground_material
        )
        scene.add(ground)
        self.print("Added ground plane.")

        # add desktop
        center_x = (self.source_pos[0] + self.target_pos[0]) / 2
        desk_physics = PhysicsMaterial(
            prim_path="/World/Physics/Desktop",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0
        )
        desk = obj.FixedCuboid(
            prim_path="/World/Desktop",
            name="Desktop",
            translation=np.array([center_x, 0, 0.1]),
            scale=np.array([0.6, 0.6, 0.2]),
            physics_material=desk_physics,
            visual_material=OmniPBR(prim_path="/World/Looks/Desktop", color=np.array([0.8, 0.8, 0.0]))
        )
        scene.add(desk)

        ob_physics = PhysicsMaterial(
            prim_path="/World/Physics/Ob",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0            
        )
        ob1 = obj.FixedCuboid(
            prim_path="/World/Ob1",
            name="Ob1",
            translation=np.array([center_x, 0, 0.25]),
            scale=np.array([0.02, 0.6, 0.10]),
            physics_material=ob_physics,
            visual_material=OmniPBR(prim_path="/World/Looks/Ob1", color=np.array([0.0, 0.0, 1.0]))
        )
        scene.add(ob1)

        # add target
        target_physics = PhysicsMaterial(
            prim_path="/World/Physics/Target",
            static_friction=10.0,
            dynamic_friction=10.0,
            restitution=0.0
        )
        target = obj.DynamicCuboid(
            prim_path="/World/Target",
            name="Target",
            translation=self.source_pos,
            size=self.size,
            physics_material=target_physics,
            visual_material=OmniPBR(prim_path="/World/Looks/Target", color=np.array([1.0, 0.0, 0.0]))
        )
        target.set_collision_approximation('boundingCube')
        prims.RigidPrim(prim_paths_expr='/World/Target', masses=[0.1]).disable_gravities()
        scene.add(target)
        self.print("Added target at position: {}".format(self.source_pos))
        self.target_prim = target

        # add franka
        # from isaacsim.storage.native import get_assets_root_path
        add_reference_to_stage(
            usd_path="assets/franka.usd", 
            prim_path="/World/Franka"
        )
        franka = Robot(
            prim_path="/World/Franka",
            name="franka",
            position=np.zeros(shape=(3, )),
            orientation=np.array([1, 0, 0, 0])
        )
        franka.set_joints_default_state(
            positions=np.array([0, 0, 0, -1, 0, 1, 0, 0.04, 0.04]),
            velocities=np.zeros(shape=(9,)),
            efforts=np.zeros(shape=(9,))
        )
        scene.add(franka)
        self.print("Added franka robot.")
        self.robot_prim = franka
        self.center_distance = prims.SingleXFormPrim(prim_path="/World/Franka/panda_hand/tool_center").get_local_pose()[0][2]

        obstacles = self.usd_helper.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path="/World/Franka",
            ignore_substring=[
                "/World/Franka",
                "/World/Target",
                "/World/GroundPlane",
                "/curobo",
            ],
        ).get_collision_check_world()
        print(obstacles.objects)
        if self.obstacle:
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                obstacles,
                interpolation_dt=1.0/60.0,
                # collision_checker_type=CollisionCheckerType.MESH
            )
        else:
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                None,
                interpolation_dt=1.0/60.0,
                # collision_checker_type=CollisionCheckerType.MESH
            )
        self.robot_solver = MotionGen(motion_gen_config)
        self.robot_solver.warmup()

        # add visual ball to end effector
        visual_ball_material = OmniGlass(
            prim_path="/World/Looks/VisualBall",
            color=np.array([0.8, 0.8, 1.0]),
            ior=1.5
        )
        self.visual_ball = obj.VisualSphere(
            prim_path="/World/VisualBall",
            name="test_visual_ball",
            radius=0.03,
            color=np.array([1.0, 0.0, 0.0]),
            visual_material=visual_ball_material
        )
        scene.add(self.visual_ball)

    def get_observations(self):
        return {
            "target_position": self.target_prim.get_world_pose()[0],
            "end_position": prims.SingleXFormPrim(prim_path="/World/Franka/panda_hand/tool_center").get_world_pose()[0],
            "gripper_hold_position": self.temp_hold["gripper"] if self.temp_hold["steady"] else 0.0
        }
    
    def ik_solve(self, target: np.ndarray):
        loss = np.inf
        result = None
        start_state = JointState(
            position=self.tensor_args.to_device(self.robot_prim.get_joint_positions()[:7]),
            velocity=self.tensor_args.to_device(np.zeros(7)),
            acceleration=self.tensor_args.to_device(np.zeros(7)),
            jerk=self.tensor_args.to_device(np.zeros(7)),
            joint_names=self.robot_prim.dof_names[:7]
        )
        start_state = start_state.get_ordered_joint_state(self.robot_solver.kinematics.joint_names)
        start_state = start_state.unsqueeze(0)  # add batch dimension
        for theta in [np.pi]:
            _target = target + np.array([-np.sin(theta)*self.center_distance, 0, -np.cos(theta)*self.center_distance])
            goal_pose = Pose(
                position=self.tensor_args.to_device(target),
                quaternion=self.tensor_args.to_device(np.array([np.cos(theta/2), 0, np.sin(theta/2), 0]))
            )
            result = self.robot_solver.plan_single(start_state, goal_pose, self.plan_config)
            if result.success.item():
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = self.robot_solver.get_full_js(cmd_plan)
                for i in range(len(cmd_plan.position)):
                    cmd = cmd_plan[i]
                    self.action.put(ArticulationAction(
                        joint_positions=cmd.position.cpu().numpy(),
                        joint_velocities=cmd.velocity.cpu().numpy(),
                        joint_indices=np.array(range(7))
                    ))
                return True
            else:
                self.print("IK attempt failed at theta = {:.4f}".format(theta))
                return False

    def sub_move_source(self, index: int, sim_time: float):
        if np.linalg.norm(self.get_observations()["end_position"] - self.source_pos) < self.size * self.tolerance:
            self.print("Reached source position.")
            self.sub = "hold"
            self.temp_hold["start"] = sim_time
            self.first_call = True
            self.visual_ball.set_visibility(False)
        elif self.first_call and self.action.empty():
            success = self.ik_solve(self.source_pos)
            if success:
                self.print("IK solution found for source position.")
            else:
                self.print("No IK solution found for source position.")
                self.sub = "error"
            self.first_call = False

    def sub_hold(self, index: int, sim_time: float):
        if self.temp_hold["steady"]:
            if sim_time - self.temp_hold["start"] > 1.0 / self.slow_rate:
                self.print("Target held steady.")
                self.sub = "move_target"
                self.first_call = True
        elif self.first_call:
            if self.action.empty():
                self.action.put(ArticulationAction(
                    joint_positions=[0.0]*2,
                    joint_indices=[7, 8]
                ))
                self.first_call = False
        else:
            gripper = self.robot_prim.get_joint_positions()[7]
            if abs(gripper - self.temp_hold["gripper"]) < self.size * self.tolerance * 0.01:
                if sim_time - self.temp_hold["start"] > 2.0 / self.slow_rate:
                    self.temp_hold["steady"] = True
                    self.temp_hold["start"] = sim_time
                    self.print("Gripper steady at {:.4f}".format(gripper))
            else:
                self.temp_hold["gripper"] = gripper
                self.temp_hold["start"] = sim_time

    def sub_move_target(self, index: int, sim_time: float):
        if np.linalg.norm(self.get_observations()["target_position"] - self.target_pos) < self.size * self.tolerance * 100:
            self.print("Target moved to target position.")
            self.sub = "loose"
            self.first_call = True
        elif self.first_call and self.action.empty():
            self.print("Gripper force: {:.4f}".format(self.robot_prim.get_measured_joint_efforts()[7]) )
            success = self.ik_solve(self.target_pos)
            if success:
                self.print("IK solution found for target position.")
            else:
                self.print("No IK solution found for target position.")
                self.sub = "error"
            self.first_call = False

    def sub_loose(self, index: int, sim_time: float):
        if self.robot_prim.get_joint_positions()[7] > 0.035:
            self.print("Gripper loosened.")
            self.sub = "done"
            self.first_call = True
        elif self.first_call and self.action.empty():
            self.action.put(ArticulationAction(
                joint_positions=[0.04, 0.04],
                joint_indices=[7, 8]
            ))

    def sub_done(self, index: int, sim_time: float):
        if self.first_call:
            self.print("Task completed successfully.")
            self.first_call = False

    def sub_error(self, index: int, sim_time: float):
        if self.first_call:
            self.print("An error occurred.")
            self.first_call = False

    def pre_step(self, index: int, sim_time: float):
        self.visual_ball.set_world_pose(
            self.get_observations()["end_position"],
            np.array([1,0,0,0])
        )

        if index % (self.slow_rate * 50) != 0 and self.obstacle:
            obstacles = self.usd_helper.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path="/World/Franka",
                ignore_substring=[
                    "/World/Franka",
                    "/World/Target",
                    "/World/VisualBall",
                    "/World/GroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            self.robot_solver.update_world(obstacles)

        getattr(self, "sub_" + self.sub)(index, sim_time)
        if not self.action.empty() and index % self.slow_rate == 0:
            self.robot_prim.apply_action(self.action.get())

if __name__ == "__main__":
    with open('log/log.txt', 'w') as log:
        # slow down for better visualization
        # err...... maybe slower for better simulation accuracy
        slow_rate = 10
        my_world = World(
            physics_dt=1.0/60.0/slow_rate,
            rendering_dt=1.0/60.0/slow_rate,
            stage_units_in_meters=1.0
        )
        my_world.add_task(Task_4(
            world=my_world,
            name="task",
            source_pos=np.array([0.4, 0.0, 0.22]),
            target_pos=np.array([0.8, 0.0, 0.22]),
            size=0.04,
            tolerance=0.01,
            slow_rate=slow_rate,
            obstacle=False,
            logger=None
        ))
        my_world.reset()

    start_time = time()
    while simulation_app.is_running() and time() - start_time < 3600:
        my_world.step(render=True)
    
    simulation_app.close()