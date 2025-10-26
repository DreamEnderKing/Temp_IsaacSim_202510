from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

from time import time, sleep
import sys

import numpy as np
from isaacsim.core.api import World
import isaacsim.core.api.objects as obj
import isaacsim.core.prims as prims
from isaacsim.core.api.materials import OmniPBR, OmniGlass, PhysicsMaterial
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

from robot_task import RobotTask

def main(world, log, ball_x, ball_radius, slow_rate):
    # preparing the scene
    assets_root_path = get_assets_root_path()
    log.write("Assets root path: {}\n".format(assets_root_path))

    # set camera view
    set_camera_view(
        eye=[ball_x/2, 2, 0.5], target=[ball_x/2, -2, 0.5], camera_prim_path="/OmniverseKit_Persp"
    )
    # add ground plane
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
    world.scene.add(ground)

    # add wooden ball
    add_reference_to_stage(
        usd_path='WoodBall.usd',
        prim_path='/World/Ball'
    )
    prims.RigidPrim(prim_paths_expr='/World/Ball', masses=[0.1]).enable_gravities()

    ball = prims.SingleGeometryPrim(
        prim_path='/World/Ball',
        name='WoodBall',
        collision=True,
        translation=np.array([ball_x,0,ball_radius+0.3]),
        orientation=np.array([1,0,0,0]),
        scale=np.array([ball_radius,ball_radius,ball_radius])
    )
    ball.set_collision_approximation('boundingSphere')
    world.scene.add(ball)

    # add franka
    asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
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
    world.scene.add(franka)

    world.add_task(RobotTask(
        name="reach_ball_task",
        target_prim=ball,
        tolerance=ball_radius*4,
        robot_prim=franka,
        slow_rate=slow_rate
    ))

if __name__ == "__main__":
    with open('log/log.txt', 'w') as log:
        # slow down by 100x for better visualization
        # err...... maybe slower for better simulation accuracy
        slow_rate = 100
        my_world = World(
            physics_dt=1.0/60.0/slow_rate,
            rendering_dt=1.0/60.0/slow_rate,
            stage_units_in_meters=1.0
        )
        main(my_world, log, ball_x=0.6, ball_radius=0.02, slow_rate=slow_rate)
        my_world.reset()

    start_time = time()
    while simulation_app.is_running() and time() - start_time < 3600:
        my_world.step(render=True)
    
    simulation_app.close()