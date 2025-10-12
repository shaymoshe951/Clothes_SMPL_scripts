import importlib.util
import os

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gutils.smpl_torch_utils import *


# ALLOWED_ERROR_TRESHOLD_IN_M = 0.015  # 2 cm
# MAX_NUMBER_OF_FIT_TRAINING_ITERS = 3000
# def fit_smpl_to_obj(obj_path, smpl_model_path, gender='female', device='cpu', scale_factor=0.01):
#     """Fit SMPL parameters to an OBJ mesh with optional scaling"""
#
#     mesh = trimesh.load(obj_path)
#     print(f"Original mesh bounds: {mesh.bounds}")
#     mesh.vertices *= scale_factor
#     print(f"Scaled mesh bounds (scale={scale_factor}): {mesh.bounds}")
#
#     target_vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
#
#     smpl = SMPL(model_path=smpl_model_path, gender=gender).to(device)
#
#     betas = torch.zeros(10, requires_grad=True, device=device)
#     body_pose = torch.zeros(69, requires_grad=True, device=device)
#     global_orient = torch.zeros(3, requires_grad=True, device=device)
#     transl = torch.zeros(3, requires_grad=True, device=device)
#
#     optimizer = torch.optim.Adam([betas, body_pose, global_orient, transl], lr=0.1)
#     # Scheduler: Reduce on validation loss plateau
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.4, patience=10, verbose=True, threshold=0.001, cooldown=0, min_lr=2e-3
#     )
#     print("Fitting SMPL parameters...")
#     sqr_loss_vector = []
#     max_err_vector = []
#     for i in range(MAX_NUMBER_OF_FIT_TRAINING_ITERS):
#         optimizer.zero_grad()
#
#         output = smpl(betas=betas.unsqueeze(0),
#                       body_pose=body_pose.unsqueeze(0),
#                       global_orient=global_orient.unsqueeze(0),
#                       transl=transl.unsqueeze(0))
#
#         loss = torch.nn.functional.mse_loss(output.vertices[0], target_vertices)
#         loss.backward()
#         optimizer.step()
#         sqr_loss_vector.append(loss.sqrt().item())
#         max_error = (output.vertices[0] - target_vertices).abs().max().item()
#         max_err_vector.append(max_error)
#
#         # Step the scheduler (triggers reduction if plateau)
#         scheduler.step(loss)
#
#         if i % 100 == 0:
#             print(f"Iteration {i}, Loss: {loss.item():.6f}")
#             print("max abs error:",max_error)
#
#         if max_error < ALLOWED_ERROR_TRESHOLD_IN_M:
#             print(f"Converged at iteration {i}, max error: {max_error:.6f}, loss: {loss.item():.6f}")
#             break
#
#     import matplotlib.pyplot as plt
#     epochs = list(range(1, len(sqr_loss_vector) + 1))  # x-axis: epochs 1 to 10
#
#     # If losses are a torch tensor, convert: loss_vector = losses.cpu().numpy().tolist()
#
#     # Create the plot
#     plt.figure(figsize=(8, 5))  # Optional: Set figure size
#     plt.plot(epochs, max_err_vector, marker='o', linewidth=2, markersize=4, color='g', label='Max Error')
#     plt.plot(epochs, sqr_loss_vector, marker='_', linewidth=2, markersize=4, color='b', label='Training Loss')
#     plt.yscale('log')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Over Epochs')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#     params = {
#         'betas': betas.detach().cpu().numpy(),
#         'body_pose': body_pose.detach().cpu().numpy(),
#         'global_orient': global_orient.detach().cpu().numpy(),
#         'transl': transl.detach().cpu().numpy(),
#         'gender': gender,
#         'scale_factor': scale_factor
#     }
#
#     return params, output.vertices[0].detach().cpu().numpy(), mesh.faces
#
#
# def save_fbx_with_smpl_controls(vertices, faces, params, fbx_path, smpl_addon_path=None):
#     """Save FBX with SMPL rig and controls"""
#
#     # If no addon path provided, try common locations
#     if smpl_addon_path is None:
#         possible_paths = [
#             os.path.expanduser("~/.config/blender/*/scripts/addons/smpl"),
#             "C:/Users/*/AppData/Roaming/Blender Foundation/Blender/*/scripts/addons/smpl",
#         ]
#         smpl_addon_path = "YOUR_SMPL_ADDON_PATH"  # User needs to specify
#
#     blender_script = f"""
# import bpy
# import sys
# import json
#
# # Add SMPL addon path if provided
# smpl_addon = r'{smpl_addon_path}'
# if smpl_addon != 'YOUR_SMPL_ADDON_PATH':
#     sys.path.append(smpl_addon)
#
# # Clear scene
# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete()
#
# # Try to use SMPL addon
# try:
#     bpy.ops.scene.smpl_add_gender(gender='{params['gender'].upper()}')
#     smpl_obj = bpy.context.object
#
#     # Set shape parameters
#     betas = {params['betas'].tolist()}
#     for i, beta in enumerate(betas[:10]):
#         smpl_obj[f'Shape{{i:03d}}'] = float(beta)
#
#     # Update mesh to reflect shape changes
#     bpy.ops.object.smpl_set_betas()
#
#     print("SMPL rig created with controls!")
#
# except:
#     print("Creating manual rig (install SMPL addon for full controls)")
#
#     # Create basic mesh with armature
#     vertices = {vertices.tolist()}
#     faces = {faces.tolist()}
#
#     mesh = bpy.data.meshes.new('SMPL_Body')
#     obj = bpy.data.objects.new('SMPL_Body', mesh)
#     bpy.context.collection.objects.link(obj)
#     mesh.from_pydata(vertices, [], faces)
#     mesh.update()
#
#     # Store parameters as custom properties
#     obj['smpl_betas'] = json.dumps({params['betas'].tolist()})
#     obj['smpl_body_pose'] = json.dumps({params['body_pose'].tolist()})
#     obj['smpl_gender'] = '{params['gender']}'
#
#     # Create simple armature
#     bpy.ops.object.armature_add()
#     armature = bpy.context.object
#     armature.name = "SMPL_Rig"
#     obj.parent = armature
#
# # Export FBX
# bpy.ops.export_scene.fbx(
#     filepath=r'{fbx_path}',
#     use_selection=False,
#     use_armature_deform_only=True,
#     add_leaf_bones=False,
#     embed_textures=False
# )
#
# print("FBX saved!")
# """
#
#     script_path = fbx_path.replace('.fbx', '_export.py')
#     with open(script_path, 'w') as f:
#         f.write(blender_script)
#
#     print(f"\n{'=' * 60}")
#     print("IMPORTANT: For full SMPL controls in Blender:")
#     print("1. Download SMPL Blender add-on from https://smpl.is.tue.mpg.de/")
#     print("2. Install it in Blender: Edit > Preferences > Add-ons > Install")
#     print("3. Update 'smpl_addon_path' in the script")
#     print(f"{'=' * 60}\n")
#     print(f"Blender script: {script_path}")
#     print(f"Run with: blender --background --python {script_path}")
#
#     return script_path
#

# Main workflow

if __name__ == "__main__":
    obj_path = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"; scale_factor = 0.01 # This model is in CM
    # obj_path = r"C:\Users\Lab\Downloads\clothes_images\liran_focal120_mesh_0_0.obj"; scale_factor = 1
    # obj_path = r"C:\Users\Lab\Downloads\clothes_images\model_l1_mesh_0_0.obj"; scale_factor = 1
    smpl_model_path = r"D:\projects\ClProjects\SMPL_Model"  # Download from SMPL website

    # Load and verify mesh
    mesh = load_mesh_obj(obj_path, True,scale_factor)


    # Fit SMPL
    params, fitted_vertices, faces = fit_smpl_to_obj(
        mesh, smpl_model_path,
        gender='female', device='cpu', flag_debug = True )

    # Save parameters
    # np.savez(obj_path.replace('.obj', '_params.npz'), **params)

    # Save with controls
    # fbx_path = obj_path.replace('.obj', '_with_rig.fbx')
    # script_path = save_fbx_with_smpl_controls(
    #     fitted_vertices, faces, params, fbx_path,
    #     smpl_addon_path=r"C:\Users\Lab\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons\smpl_blender_addon"  # Update this!
    # )

