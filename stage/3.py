from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

from time import time, sleep
import sys

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import OmniPBR, OmniGlass, PhysicsMaterial
from isaacsim.core.api.sensors.rigid_contact_view import RigidContactView
import isaacsim.core.api.objects as obj
import isaacsim.core.prims as prims
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

def main(world, log):
    # preparing the scene
    assets_root_path = get_assets_root_path()
    log.write("Assets root path: {}\n".format(assets_root_path))
    friction = 0.0

    # set camera view
    set_camera_view(
        eye=[2.5, 1, 5], target=[2.5, 1, 0], camera_prim_path="/OmniverseKit_Persp"
    )

    # add ground plane
    add_reference_to_stage(
        usd_path="assets/ground_plane.usd",
        prim_path="/World/GroundPlane"
    )
    ground_material = PhysicsMaterial(
        prim_path="/World/Physics/GroundPlane",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=1.0
    )
    ground = obj.GroundPlane(
        prim_path="/World/GroundPlane",
        name="GroundPlane",
        z_position=0,
        physics_material=ground_material
    )
    prims.SingleXFormPrim(
        prim_path="/World/GroundPlane/SphereLight"
    ).set_world_pose(np.array([2.5, 1, 2]), np.array([1,0,0,0]))
    world.scene.add(ground)

    # add box
    box_visual = OmniPBR(
        prim_path="/World/Looks/Box",
        color=np.array([0.3, 0.0, 1.0])
    )
    box_physics = PhysicsMaterial(
        prim_path="/World/Physics/Box",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=1.0
    )
    box_F = obj.FixedCuboid(
        prim_path="/World/Box/Box_F",
        name="Box_F",
        size=1,
        scale=[0.2, 2, 4],
        position=[-0.1, 1, 2],
        visual_material=box_visual,
        physics_material=box_physics
    )
    box_B = obj.FixedCuboid(
        prim_path="/World/Box/Box_B",
        name="Box_B",
        size=1,
        scale=[0.2, 2, 4],
        position=[5.1, 1, 2],
        visual_material=box_visual,
        physics_material=box_physics
    )
    box_L = obj.FixedCuboid(
        prim_path="/World/Box/Box_L",
        name="Box_L",
        size=1,
        scale=[5.4, 0.2, 4],
        position=[2.5, -0.1, 2],
        visual_material=box_visual,
        physics_material=box_physics
    )
    box_R = obj.FixedCuboid(
        prim_path="/World/Box/Box_R",
        name="Box_R",
        size=1,
        scale=[5.4, 0.2, 4],
        position=[2.5, 2.1, 2],
        visual_material=box_visual,
        physics_material=box_physics
    )
    world.scene.add(box_F)
    world.scene.add(box_B)
    world.scene.add(box_L)
    world.scene.add(box_R)

    # add ball
    ball_visual = OmniPBR(
        prim_path="/World/Looks/Ball",
        color=np.array([0.0, 0.5, 0.0])
    )
    ball_physics = PhysicsMaterial(
        prim_path="/World/Physics/Ball",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=1.0
    )
    ball = obj.DynamicSphere(
        prim_path="/World/Ball",
        name="Ball",
        radius=0.1,
        position=[2.5, 1, 2.1],
        mass=0.5,
        linear_velocity=[1, 0, 0],
        visual_material=ball_visual,
        physics_material=ball_physics
    )
    prims.RigidPrim(
        prim_paths_expr='/World/Ball'
    ).enable_gravities()
    ball.set_collision_approximation('boundingSphere')
    world.scene.add(ball)

    ball_view = RigidContactView(
        prim_paths_expr="/World/Ball",
        filter_paths_expr=[
            "/World/Box/Box_F",
            "/World/Box/Box_B",
            "/World/Box/Box_L",
            "/World/Box/Box_R",
            "/World/GroundPlane"
        ],
        name="Ball_View",
        max_contact_count=6
    )

    world.reset()
    ball_view.initialize(world.physics_sim_view)

    return {
        "ball_view": ball_view
    }

def pre_stage(log, params):
    ball = params["ball_view"]
    forces = ball.get_contact_force_data()
    # print('Forces: {0}'.format(forces))
    # if all(forces > 0):
    #     print('Forces: {0}'.format(forces))


if __name__ == "__main__":
    with open('/root/workspace/main/temp/Temp_IsaacSim_202510/log/log.txt', 'w') as log:
        my_world = World(stage_units_in_meters=1.0)
        args = main(my_world, log)

    start_time = time()
    while simulation_app.is_running() and time() - start_time < 3600:
        pre_stage(log, args)
        my_world.step(render=True)
    
    simulation_app.close()