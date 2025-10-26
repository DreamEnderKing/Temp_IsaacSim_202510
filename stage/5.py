from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import numpy as np

from isaacsim.core.api import World
from isaacsim.core.api.materials import OmniPBR
import isaacsim.core.api.objects as obj
from isaacsim.core.utils.stage import add_reference_to_stage

from curobo.util_file import join_path, load_yaml

world = World(stage_units_in_meters=1.0)

add_reference_to_stage(
    usd_path="assets/ground_plane.usd",
    prim_path="/World/GroundPlane"
)

add_reference_to_stage(
    usd_path="assets/franka.usd", 
    prim_path="/World/Franka"
)

config = load_yaml("/root/workspace/main/temp/Temp_IsaacSim_202510/assets/franka_desc/franka_mesh.yml")
material = OmniPBR(prim_path="/World/Materials/Material_Sphere", color=np.array([0,1,0]))
for name, params in config["collision_spheres"].items():
    i = 0
    for param in params:
        s = obj.VisualSphere(
            prim_path=f"/World/Franka/{name}/sphere_{i}",
            radius=param["radius"],
            visual_material=material
        )
        s.set_local_pose(param["center"], [1,0,0,0])
        i += 1

while simulation_app.is_running():
    world.step(render=True)