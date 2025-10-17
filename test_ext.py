import numpy as np
import isaacsim.core.api.objects as obj
import isaacsim.core.prims as prims
import isaacsim.core.utils.viewports as view
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver

#view.set_camera_view(eye=[1.5, 0.5, 1.5], target=[0.0, -0.5, 0.5])
# print(prims.__file__)

assets_root_path = get_assets_root_path()

# Add Ground Plane
# obj.GroundPlane(prim_path='/World/GroundPlane', name='GroundPlane', z_position=0.0)

# Add Wooden Ball
add_reference_to_stage(
    usd_path='/root/workspace/main/export/WoodBall.usd',
    prim_path='/World/Ball'
)
ball = prims.GeometryPrim(prim_paths_expr='/World/Ball', collisions=[True])
ball.set_collision_approximations(['boundingSphere'])
ball_radius = 0.02
ball.set_world_poses(np.array([[0.4,0,ball_radius+0.3]]), np.array([[1,0,0,0]]))
ball.set_local_scales(np.array([[ball_radius,ball_radius,ball_radius]]))

ball = prims.RigidPrim(prim_paths_expr='/World/Ball')
ball.set_masses([0.1])
ball.disable_gravities()
# ball.apply_forces_and_torques_at_pos(np.array([[-1, 0, 0]]), np.array([[0, 0, 1]]))

# Add Franka
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")  # add robot to stage

franka = Robot(prim_path="/World/Franka")

# print('positions:', franka.get_joint_positions())
# reset: default gripper open, arm straight up
franka.set_default_state(
    position=np.zeros(shape=(3, )),
    orientation=np.array([0.5, 0, 0, 0]),  # xyzw
)
franka.set_joints_default_state(
    positions=np.array([0, 0, 0, -1, 0, 1, 0, 0.04, 0.04]),
    velocities=np.zeros(shape=(franka.num_dof,)),
    efforts=np.zeros(shape=(franka.num_dof,))
)
franka.set_joint_positions(np.zeros(9))
franka.post_reset()
if not franka.handles_initialized:
    franka.initialize()
with open('/root/workspace/main/export/franka_dof.txt', 'w') as f:
    # 写入表头
    f.write("DOF Name       |Type        |Limits|Lower    |Upper    |Drive Mode |Max Vel    |Max Effort |Stiffness  |Damping    \n")
    f.write("-" * 120 + "\n")
    
    for cn in range(franka.num_dof):
        props = franka.dof_properties[cn]
        
        # 类型映射
        type_map = {0: "unknown", 1: "rotation", 2: "translation"}
        drive_map = {1: "force", 2: "acceleration"}
        
        f.write("{:<15}|{:<12}|{:<6}|{:<9.3f}|{:<9.3f}|{:<11}|{:<10.3f}|{:<10.3f}|{:<10.3f}|{:<10.3f}\n".format(
            franka.dof_names[cn],
            type_map.get(props[0], "invalid"),
            "Yes" if props[1] else "No",
            props[2], props[3],  # lower, upper
            drive_map.get(props[4], "unknown"),
            props[5], props[6], props[7], props[8]  # velocity, effort, stiffness, damping
        ))

controller = franka.get_articulation_controller()
solver = KinematicsSolver(franka)

target_positions, _ = ball.get_world_poses()
print("Target position:", target_positions)

action, success = solver.compute_inverse_kinematics(
    target_position=target_positions[0],
    target_orientation=np.array([0, 0, 1, 0])
)
if success:
    controller.apply_action(action)
else:
    print("IK did not converge to a solution.  No action is being taken.")

gripper_positions = franka.get_joint_positions()
gripper_positions[-2:] = np.array([0.0, 0.0])  # close gripper
controller.apply_action(ArticulationAction(joint_positions=gripper_positions))