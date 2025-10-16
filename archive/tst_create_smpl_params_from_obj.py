import importlib.util
import os

import os, sys

from matplotlib.scale import scale_factory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gutils.smpl_torch_utils import *

# Main workflow

if __name__ == "__main__":
    # obj_path = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"; scale_factor = 0.01 # This model is in CM
    # obj_path = r"C:\Users\Lab\Downloads\clothes_images\liran_focal120_mesh_0_0.obj"; scale_factor = 1
    # obj_path = r"C:\Users\Lab\Downloads\clothes_images\model_l1_mesh_0_0.obj"; scale_factor = 1
    # obj_path = r"C:\Users\Lab\Downloads\clothes_images\mesh_w_clothes_from_opensite\model_xs1_mesh.obj"; scale_factor = 1.6/1.95
    # obj_path = r"C:\Users\Lab\Downloads\clothes_images\mesh_w_clothes_from_opensite\white_mesh (2).obj"; scale_factor = 1#1.8/1.96
    # obj_path = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\BCNet\recs\model_xs1a_smpl.obj"; scale_factor = 1.0
    obj_path = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\BCNet\recs\liran_smpl.obj"; scale_factor = 1.0

    smpl_model_path = r"D:\projects\ClProjects\SMPL_Model"  # Download from SMPL website

    # Load and verify mesh
    mesh = load_mesh_obj(obj_path, True,scale_factor)


    # Fit SMPL
    model, params = fit_smpl_to_obj(
        mesh, smpl_model_path,
        gender='female', device='cpu', flag_debug = True, flag_fit_mesh_to_smpl=True )

    print("Here")

    # Save parameters
    # np.savez(obj_path.replace('.obj', '_params.npz'), **params)

    # Save with controls
    # fbx_path = obj_path.replace('.obj', '_with_rig.fbx')
    # script_path = save_fbx_with_smpl_controls(
    #     fitted_vertices, faces, params, fbx_path,
    #     smpl_addon_path=r"C:\Users\Lab\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons\smpl_blender_addon"  # Update this!
    # )

