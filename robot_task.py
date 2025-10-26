import numpy as np
from typing import Optional

from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.api.materials import OmniGlass
import isaacsim.core.api.objects as obj
from isaacsim.core.api.robots import Robot
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver

# from curobo.util.usd_helper import CuroboUSDHelper

class RobotTask(BaseTask):
    def __init__(
        self,
        name: str,
        target_prim: GeometryPrim,
        tolerance: float,
        robot_prim: Robot,
        offset: Optional[np.ndarray] = None,
        logger=None,
        slow_rate: int = 1
    ):
        super().__init__(name=name, offset=offset)
        self.target_prim = target_prim
        self.tolerance = tolerance
        self.target_position = np.zeros(3)

        self.robot_prim = robot_prim
        self.robot_controller = robot_prim.get_articulation_controller()
        self.robot_solver = KinematicsSolver(robot_prim)
        # 0: ready, 1:running
        self.state = 0
        self.last_target = np.zeros(3)
        self.last_stamp = 0.0
        self.last_stay = 0.0
        self.logger = logger
        self.slow_rate = slow_rate

        self.visual_ball = None
        # self.usd_helper = CuroboUSDHelper()

    def print(self, msg):
        if self.logger:
            self.logger.write(msg + "\n")
        else:
            print(msg)

    def get_observations(self):
        # works when robot in (0,0,0) and no rotation
        target_position, _ = self.target_prim.get_world_pose()
        robot_position, _ = self.robot_prim.get_world_pose()
        end_position, _ = self.robot_solver.compute_end_effector_pose()
        return {
            "target_position": target_position,
            "robot_position": robot_position,
            "end_position": end_position
        }
    
    def get_params(self):
        return {
            "target_prim": self.target_prim,
            "tolerance": self.tolerance,
            "robot_prim": self.robot_prim,
        }
    
    def set_params(self, params):
        if "target_prim" in params:
            self.target_prim = params["target_prim"]
            self.target_position = self.target_prim.get_world_poses()[0][0]
        if "tolerance" in params:
            self.tolerance = params["tolerance"]
        if "robot_prim" in params:
            self.robot_prim = params["robot_prim"]
            self.robot_controller = self.robot_prim.get_articulation_controller()
            self.robot_solver = KinematicsSolver(self.robot_prim)
    
    def calculate_metrics(self):
        return {
            "distance_to_target": np.linalg.norm(
                self.get_observations()["end_position"] - self.get_observations()["target_position"]
            )
        }
    
    def is_done(self):
        return np.linalg.norm(self.calculate_metrics()["distance_to_target"]) < 0.05

    def ik_solve(self):
        loss = np.inf
        result = None
        for theta in np.linspace(0, np.pi, 7):
            action, success = self.robot_solver.compute_inverse_kinematics(
                target_position=self.target_position,
                target_orientation=np.array([np.cos(theta/2), 0, np.sin(theta/2), 0]),
                position_tolerance=self.tolerance,
                orientation_tolerance=np.pi/8
            )
            action_loss = np.linalg.norm(self.robot_prim.get_joint_positions()[:7] - action.joint_positions)
            if success and action_loss < loss:
                loss = action_loss
                result = action
        return result

    def pre_step(self, index: int, sim_time: float):
        # print contact forces
        # self.print(self.target_prim.get_contact_force_data())

        # visualize target
        self.visual_ball.set_world_pose(
            np.array(self.get_observations()["end_position"]),
            np.array([1,0,0,0])
        )

        # target position update
        if self.state == 0 and np.linalg.norm(self.target_position - self.get_observations()["target_position"]) > 0.1:
            # wait until target stays still for 1 second
            if np.linalg.norm(self.get_observations()["target_position"] - self.last_target) < 0.01:
                self.last_stay += sim_time - self.last_stamp
            else:
                self.last_stay = 0.0
                self.last_target = self.get_observations()["target_position"]
            if self.last_stay > 0.5 / self.slow_rate:
                self.target_position = self.get_observations()["target_position"]
                self.last_target = self.target_position
                self.print("Target moved, start to reach the target.")
                action = self.ik_solve()
                if action:
                    self.robot_controller.apply_action(action)
                    self.state = 1
                    self.action_start_time = sim_time
                    self.print("Action applied, start moving.")
                else:
                    self.print("IK did not converge to a solution. No action is being taken.")
                self.last_stay = 0.0
        elif self.state == 1:
            if self.is_done():
                self.print("Reached the target position successfully.")
                self.state = 0
            elif sim_time - self.action_start_time > 3.0 / self.slow_rate:
                self.print("Failed to reach the target position within the time limit.")
                self.state = 0
                self.last_stay = 3.0 / self.slow_rate
        self.last_stamp = sim_time

    def set_up_scene(self, scene):
        visual_ball_material = OmniGlass(
            prim_path="/World/Looks/VisualBall",
            color=np.array([0.0, 1.0, 0.0]),
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

    def post_reset(self):
        kps, kds = self.robot_prim._articulation_controller.get_gains()
        # self.robot_prim._articulation_controller.set_gains(
        #     kps=kps*0.0000001,
        #     kds=kds*0.00001
        # )
        print("RobotTask post reset: lowered joint gains for better IK tracking.")
        print("Current Kps:", self.robot_prim._articulation_controller.get_gains()[0])
        print("Current Kds:", self.robot_prim._articulation_controller.get_gains()[1])