from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

from time import time, sleep
import sys

import numpy as np
from isaacsim.core.api import World
import isaacsim.core.api.objects as obj
import isaacsim.core.prims as prims
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

from robot_task import RobotTask

def main(world, log, ball_x, ball_radius):
    # preparing the scene
    assets_root_path = get_assets_root_path()
    log.write("Assets root path: {}\n".format(assets_root_path))

    # set camera view
    set_camera_view(
        eye=[ball_x/2, 2, 0.5], target=[ball_x/2, -2, 0.5], camera_prim_path="/OmniverseKit_Persp"
    )
    # add ground plane
    world.scene.add_default_ground_plane()

    # add wooden ball
    add_reference_to_stage(
        usd_path='/root/workspace/main/export/WoodBall.usd',
        prim_path='/World/Ball'
    )
    # ball = prims.RigidPrim(prim_paths_expr='/World/Ball', masses=[0.1])
    # ball.enable_gravities()

    ball = prims.GeometryPrim(prim_paths_expr='/World/Ball', collisions=[True])
    ball.set_collision_approximations(['boundingSphere'])
    ball.set_world_poses(np.array([[ball_x,0,ball_radius+0.3]]), np.array([[1,0,0,0]]))
    ball.set_local_scales(np.array([[ball_radius,ball_radius,ball_radius]]))

    # add franka
    asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
    franka = Robot(prim_path="/World/Franka", name="franka")
    franka.set_default_state(
        position=np.zeros(shape=(3, )),
        orientation=np.array([0.5, 0, 0, 0]),  # xyzw
    )
    franka.set_joints_default_state(
        positions=np.array([0, 0, 0, -1, 0, 1, 0, 0.04, 0.04]),
        velocities=np.zeros(shape=(9,)),
        efforts=np.zeros(shape=(9,))
    )

    # initialize the world
    world.reset()
    if not franka.handles_initialized:
        franka.initialize()
    franka.post_reset()

    world.add_task(RobotTask(
        name="reach_ball_task",
        target_prim=ball,
        tolerance=ball_radius*4,
        robot_prim=franka
    ))

if __name__ == "__main__":
    with open('/root/workspace/main/export/log/log.txt', 'w') as log:
        my_world = World(stage_units_in_meters=1.0)
        main(my_world, log, ball_x=0.6, ball_radius=0.02)

    start_time = time()
    while simulation_app.is_running() and time() - start_time < 3600:
        my_world.step(render=True)
    
    simulation_app.close()