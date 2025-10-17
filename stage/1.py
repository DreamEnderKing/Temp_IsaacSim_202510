import numpy as np
import isaacsim.core.api.objects as obj
import isaacsim.core.prims as prims

obj.DynamicCuboid(
    prim_path="/cube1",
    name="cube1",
    position=np.array([0, 0.5, 0.5]),
    size=0.3,
    color=np.array([0, 255, 255]),
    mass=1.0,
)

trans = np.array([[1.0, 0.0, 1.0]])
orient = np.array([[np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]])
scale = np.array([[1.0, 1.0, 2.0]])

visual_cube_xform = prims.XFormPrim(prim_paths_expr='/cube1')
visual_cube_xform.set_world_poses(trans, orient)
visual_cube_xform.set_local_scales(scale)