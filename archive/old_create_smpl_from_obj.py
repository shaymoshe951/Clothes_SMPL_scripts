# import numpy as np
# import trimesh
# from smplx import SMPL
# import torch
#
# # Load your OBJ mesh
# mesh = trimesh.load(r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj")
# print(f"Vertices: {len(mesh.vertices)}")  # SMPL has 6890 vertices
# print(f"Faces: {len(mesh.faces)}")        # SMPL has 13776 faces
# target_vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
#
# # Create SMPL model
# smpl = SMPL(model_path=r"D:\projects\ClProjects\SMPL_Model" , gender='female')
#
# # Initialize parameters to optimize
# betas = torch.zeros(10, requires_grad=True)  # shape parameters
# body_pose = torch.zeros(69, requires_grad=True)  # pose parameters
# global_orient = torch.zeros(3, requires_grad=True)  # global rotation
# transl = torch.zeros(3, requires_grad=True)  # translation
#
# # Optimization
# optimizer = torch.optim.Adam([betas, body_pose, global_orient, transl], lr=0.01)
#
# for i in range(10000):
#     optimizer.zero_grad()
#
#     # Generate SMPL mesh with current parameters
#     output = smpl(betas=betas.unsqueeze(0),
#                   body_pose=body_pose.unsqueeze(0),
#                   global_orient=global_orient.unsqueeze(0),
#                   transl=transl.unsqueeze(0))
#
#     # Loss: difference between generated and target vertices
#     loss = torch.nn.functional.mse_loss(output.vertices[0], target_vertices)
#
#     loss.backward()
#     optimizer.step()
#
#     if i % 100 == 0:
#         print(f"Iteration {i}, Loss: {loss.item()}")
#
# # Save parameters
# np.savez('smpl_params.npz',
#          betas=betas.detach().numpy(),
#          body_pose=body_pose.detach().numpy(),
#          global_orient=global_orient.detach().numpy(),
#          transl=transl.detach().numpy())

import numpy as np
import torch
from smplx import SMPL
import trimesh
import pickle


def fit_smpl_to_obj(obj_path, smpl_model_path, gender='female', device='cuda', scale_factor = 0.01):
    """Fit SMPL parameters to an OBJ mesh"""

    # Load target mesh
    mesh = trimesh.load(obj_path)
    # Apply scaling factor
    print(f"Original mesh bounds: {mesh.bounds}")
    mesh.vertices *= scale_factor
    print(f"Scaled mesh bounds (scale={scale_factor}): {mesh.bounds}")

    target_vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)

    # Create SMPL model
    smpl = SMPL(model_path=smpl_model_path, gender=gender).to(device)

    # Initialize parameters
    betas = torch.zeros(10, requires_grad=True, device=device)
    body_pose = torch.zeros(69, requires_grad=True, device=device)
    global_orient = torch.zeros(3, requires_grad=True, device=device)
    transl = torch.zeros(3, requires_grad=True, device=device)

    # Optimize
    optimizer = torch.optim.Adam([betas, body_pose, global_orient, transl], lr=0.3)

    print("Fitting SMPL parameters...")
    for i in range(1500):
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

    # Return fitted parameters
    params = {
        'betas': betas.detach().cpu().numpy(),
        'body_pose': body_pose.detach().cpu().numpy(),
        'global_orient': global_orient.detach().cpu().numpy(),
        'transl': transl.detach().cpu().numpy(),
        'gender': gender
    }

    return params, output.vertices[0].detach().cpu().numpy()


def save_fbx_with_smpl_params(vertices, faces, params, fbx_path):
    """Save mesh as FBX with SMPL parameters embedded"""

    # Method 1: Use Blender Python API (most reliable)
    blender_script = f"""
import bpy
import json
from numpy import array ,float32

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create mesh
vertices = {vertices.tolist()}
faces = {faces.tolist()}

mesh = bpy.data.meshes.new('SMPL_Body')
obj = bpy.data.objects.new('SMPL_Body', mesh)
bpy.context.collection.objects.link(obj)

# Add geometry
mesh.from_pydata(vertices, [], faces)
mesh.update()

# Embed SMPL parameters as custom properties
params = {repr(params)}
obj['smpl_betas'] = json.dumps(params['betas'].tolist())
obj['smpl_body_pose'] = json.dumps(params['body_pose'].tolist())
obj['smpl_global_orient'] = json.dumps(params['global_orient'].tolist())
obj['smpl_transl'] = json.dumps(params['transl'].tolist())
obj['smpl_gender'] = params['gender']

# Export to FBX
bpy.ops.export_scene.fbx(filepath=r'{fbx_path}', 
                         use_selection=True,
                         embed_textures=False)

print("FBX saved with SMPL parameters!")
"""

    # Save script
    script_path = fbx_path.replace('.fbx', '_export2.py')
    with open(script_path, 'w') as f:
        f.write(blender_script)

    print(f"\nBlender script saved to: {script_path}")
    print(f"Run with: blender --background --python {script_path}")

    return script_path


# Main workflow
if __name__ == "__main__":
    # Step 1: Fit SMPL to OBJ
    obj_path = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"
    smpl_model_path = r"D:\projects\ClProjects\SMPL_Model"  # Download from SMPL website

    params, fitted_vertices = fit_smpl_to_obj(obj_path, smpl_model_path, gender='female')

    # Load original faces
    mesh = trimesh.load(obj_path)

    # Step 2: Save as FBX with parameters
    fbx_path = obj_path.replace('.obj', '_with_params.fbx')
    script_path = save_fbx_with_smpl_params(fitted_vertices, mesh.faces, params, fbx_path)

    # Step 3: Also save parameters separately
    np.savez(obj_path.replace('.obj', '_params.npz'), **params)
    print(f"\nParameters also saved to: {obj_path.replace('.obj', '_params.npz')}")