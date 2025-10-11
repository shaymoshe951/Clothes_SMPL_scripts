import numpy as np
import torch
from smplx import SMPL
import trimesh
import os


def fit_smpl_to_obj(obj_path, smpl_model_path, gender='female', device='cpu', scale_factor=0.01):
    """Fit SMPL parameters to an OBJ mesh with optional scaling"""

    mesh = trimesh.load(obj_path)
    print(f"Original mesh bounds: {mesh.bounds}")
    mesh.vertices *= scale_factor
    print(f"Scaled mesh bounds (scale={scale_factor}): {mesh.bounds}")

    target_vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)

    smpl = SMPL(model_path=smpl_model_path, gender=gender).to(device)

    betas = torch.zeros(10, requires_grad=True, device=device)
    body_pose = torch.zeros(69, requires_grad=True, device=device)
    global_orient = torch.zeros(3, requires_grad=True, device=device)
    transl = torch.zeros(3, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([betas, body_pose, global_orient, transl], lr=0.01)

    print("Fitting SMPL parameters...")
    for i in range(1000):
        optimizer.zero_grad()

        output = smpl(betas=betas.unsqueeze(0),
                      body_pose=body_pose.unsqueeze(0),
                      global_orient=global_orient.unsqueeze(0),
                      transl=transl.unsqueeze(0))

        loss = torch.nn.functional.mse_loss(output.vertices[0], target_vertices)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}")

    params = {
        'betas': betas.detach().cpu().numpy(),
        'body_pose': body_pose.detach().cpu().numpy(),
        'global_orient': global_orient.detach().cpu().numpy(),
        'transl': transl.detach().cpu().numpy(),
        'gender': gender,
        'scale_factor': scale_factor
    }

    return params, output.vertices[0].detach().cpu().numpy(), mesh.faces


def save_fbx_with_smpl_controls(vertices, faces, params, fbx_path, smpl_addon_path=None):
    """Save FBX with SMPL rig and controls"""

    # If no addon path provided, try common locations
    if smpl_addon_path is None:
        possible_paths = [
            os.path.expanduser("~/.config/blender/*/scripts/addons/smpl"),
            "C:/Users/*/AppData/Roaming/Blender Foundation/Blender/*/scripts/addons/smpl",
        ]
        smpl_addon_path = "YOUR_SMPL_ADDON_PATH"  # User needs to specify

    blender_script = f"""
import bpy
import sys
import json

# Add SMPL addon path if provided
smpl_addon = r'{smpl_addon_path}'
if smpl_addon != 'YOUR_SMPL_ADDON_PATH':
    sys.path.append(smpl_addon)

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Try to use SMPL addon
try:
    bpy.ops.scene.smpl_add_gender(gender='{params['gender'].upper()}')
    smpl_obj = bpy.context.object

    # Set shape parameters
    betas = {params['betas'].tolist()}
    for i, beta in enumerate(betas[:10]):
        smpl_obj[f'Shape{{i:03d}}'] = float(beta)

    # Update mesh to reflect shape changes
    bpy.ops.object.smpl_set_betas()

    print("SMPL rig created with controls!")

except:
    print("Creating manual rig (install SMPL addon for full controls)")

    # Create basic mesh with armature
    vertices = {vertices.tolist()}
    faces = {faces.tolist()}

    mesh = bpy.data.meshes.new('SMPL_Body')
    obj = bpy.data.objects.new('SMPL_Body', mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Store parameters as custom properties
    obj['smpl_betas'] = json.dumps({params['betas'].tolist()})
    obj['smpl_body_pose'] = json.dumps({params['body_pose'].tolist()})
    obj['smpl_gender'] = '{params['gender']}'

    # Create simple armature
    bpy.ops.object.armature_add()
    armature = bpy.context.object
    armature.name = "SMPL_Rig"
    obj.parent = armature

# Export FBX
bpy.ops.export_scene.fbx(
    filepath=r'{fbx_path}',
    use_selection=False,
    use_armature_deform_only=True,
    add_leaf_bones=False,
    embed_textures=False
)

print("FBX saved!")
"""

    script_path = fbx_path.replace('.fbx', '_export.py')
    with open(script_path, 'w') as f:
        f.write(blender_script)

    print(f"\n{'=' * 60}")
    print("IMPORTANT: For full SMPL controls in Blender:")
    print("1. Download SMPL Blender add-on from https://smpl.is.tue.mpg.de/")
    print("2. Install it in Blender: Edit > Preferences > Add-ons > Install")
    print("3. Update 'smpl_addon_path' in the script")
    print(f"{'=' * 60}\n")
    print(f"Blender script: {script_path}")
    print(f"Run with: blender --background --python {script_path}")

    return script_path


# Main workflow
if __name__ == "__main__":
    obj_path = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"
    smpl_model_path = r"D:\projects\ClProjects\SMPL_Model"  # Download from SMPL website

    # Fit SMPL
    params, fitted_vertices, faces = fit_smpl_to_obj(
        obj_path, smpl_model_path,
        gender='female', scale_factor=0.01
    )

    # Save parameters
    # np.savez(obj_path.replace('.obj', '_params.npz'), **params)

    # Save with controls
    # fbx_path = obj_path.replace('.obj', '_with_rig.fbx')
    # script_path = save_fbx_with_smpl_controls(
    #     fitted_vertices, faces, params, fbx_path,
    #     smpl_addon_path=r"C:\Users\Lab\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons\smpl_blender_addon"  # Update this!
    # )

