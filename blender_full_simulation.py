# ──────────────────────────────────────────────────────────────────────────────
# Path setup & imports
# ──────────────────────────────────────────────────────────────────────────────
import sys
import os

EXTRA_PY_PATHS = [r"C:\Users\Lab\AppData\Roaming\Python\Python311\Scripts" + "\\..\\site-packages",
                  r"D:\projects\ClProjects\SMPL-Anthropometry"]
for p in EXTRA_PY_PATHS:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

for name in list(sys.modules.keys()):
    if name.startswith("gutils"):
        print(name)
        del sys.modules[name]

from mathutils import Quaternion, Matrix, Vector
import bpy
from gutils.smpl_torch_utils import *
from gutils.blender_utils import *
import numpy as np
import torch
from smplx import SMPL
import trimesh
import math

# ---------- Config ----------
OBJ_BODY_PATH = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"
SCALE_FACTOR = 0.01  # scales your input OBJ vertices before fitting
SMPL_MODEL_PATH = r"D:\projects\ClProjects\SMPL_Model"  # Download from SMPL website
SMPL_GENDER = "FEMALE"  # "MALE" | "FEMALE" | "NEUTRAL"
OUTPUT_FBX = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m2.fbx"
FBX_TEMPLATE_PATH_N_PREFIX = r"C:\Users\Lab\Downloads\template_smpl_"
OBJ_GARMENT_PATH  = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_garment_modified_rot_scaled.obj"

DEVICE = "cpu"  # 'cpu' is fine for this

# Main workflow
if __name__ == "__main__":
    # Load mesh
    print('Starting SMPL fitting and FBX export...')
    mesh = load_mesh_obj(OBJ_BODY_PATH, True, scale_factor=SCALE_FACTOR)

    # Fit SMPL
    model, params = fit_smpl_to_obj(
        mesh, SMPL_MODEL_PATH,
        gender=SMPL_GENDER, device=DEVICE
    )

    # Clear Blender scene
    clear_scene()

    # Load Template FBX
    fbx_temp_obj = import_fbx(FBX_TEMPLATE_PATH_N_PREFIX + SMPL_GENDER.lower() + ".fbx")


    shape_key_values = compute_smpl_shape_key_values(params['betas'], params['body_pose'],
                                                     params['global_orient'], params['transl'], device=DEVICE)

    body_mesh_obj, armature_obj = select_smpl_mesh(fbx_temp_obj)
    set_blender_shape_keys(body_mesh_obj, shape_key_values)

    # Set bone poses
    set_smpl_pose_on_armature(armature_obj, params['global_orient'], params['body_pose'], DEVICE)

    # Apply translation (to armature or mesh; here to armature)
    # armature_obj.location = Vector(params['transl'].cpu().numpy())

    bpy.context.view_layer.update()

    make_rigid(body_mesh_obj)

    garment_mesh_obj = import_obj_to_blender(OBJ_GARMENT_PATH)

    rot_obj(garment_mesh_obj, 'X', 90)

    tx,ty,tz = params['transl'].numpy()
    garment_mesh_obj.location = Vector((tx, tz, -ty))

    make_garment(garment_mesh_obj)

    print("Done!")

