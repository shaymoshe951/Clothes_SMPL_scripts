# ──────────────────────────────────────────────────────────────────────────────
# Path setup & imports
# ──────────────────────────────────────────────────────────────────────────────
import sys
import os
EXTRA_PY_PATHS = [r"C:\Users\Lab\AppData\Roaming\Python\Python311\Scripts" + "\\..\\site-packages"]
for p in EXTRA_PY_PATHS:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

import numpy as np
import torch
from smplx import SMPL
import trimesh
import math
from mathutils import Vector
import numpy as np
import bpy

# ---------- Config ----------
OBJ_BODY_PATH = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"
SMPL_MODEL_PATH = r"D:\projects\ClProjects\SMPL_Model"  # Download from SMPL website
SMPL_GENDER       = "FEMALE"  # "MALE" | "FEMALE" | "NEUTRAL"
OUTPUT_FBX        = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m2.fbx"
FBX_TEMPLATE_PATH_N_PREFIX = r"C:\Users\Lab\Downloads\template_smpl_"

SCALE_FACTOR = 0.01  # scales your input OBJ vertices before fitting
DEVICE = "cpu"  # 'cpu' is fine for this

AUTOMATIC_BONE_ORIENTATION = False

def axis_angle_to_rot_mat(aa):
    """
    Convert axis-angle vector to 3x3 rotation matrix using Rodrigues' formula.

    Args:
        aa (torch.Tensor): Axis-angle vector(s) of shape (..., 3)

    Returns:
        torch.Tensor: Rotation matrix/matrices of shape (..., 3, 3)
    """
    batch_size = 1 if aa.dim() == 1 else aa.shape[0]
    if aa.dim() == 1:
        aa = aa.unsqueeze(0)

    angle = torch.norm(aa, dim=-1, keepdim=True)
    # Handle zero angle case
    axis = aa / torch.clamp(angle, min=1e-8)

    # Skew-symmetric matrix K
    K = torch.zeros((batch_size, 3, 3), device=aa.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    # Identity
    I = torch.eye(3, device=aa.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # Rodrigues' formula
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    R = I + sin_a.unsqueeze(-1).unsqueeze(-1) * K + (1 - cos_a.unsqueeze(-1).unsqueeze(-1)) * torch.matmul(K, K)

    if batch_size == 1:
        R = R.squeeze(0)

    return R


def set_blender_shape_keys(obj, shape_key_values):
    """
    Set the computed values on the object's shape keys.

    Args:
        obj (bpy.types.Object): The SMPL mesh object with 218 shape keys
        shape_key_values (torch.Tensor): Values from compute_smpl_shape_key_values
    """
    if obj.type != 'MESH' or not obj.data.shape_keys:
        print("Error: Object must be a mesh with shape keys.")
        return

    shape_keys = obj.data.shape_keys.key_blocks
    if len(shape_keys) != 218:
        print(f"Warning: Expected 218 shape keys, found {len(shape_keys)}.")
        return

    # Set values (convert to float)
    for i, key in enumerate(shape_keys):
        key.value = shape_key_values[i].item()

    # Update view layer
    bpy.context.view_layer.update()
    print(f"Updated {len(shape_keys)} shape keys on {obj.name}")

def compute_smpl_shape_key_values(betas, body_pose, global_orient, transl, device='cpu'):
    """
    Compute the 218 shape key values for an SMPL mesh based on the input parameters.

    - Basis (index 0): Always 1.0
    - Shape blend shapes (indices 1-10): Directly from betas (10D)
    - Pose corrective blend shapes (indices 11-217): From vec(R(θ_local) - I) for 23 body joints (207D)

    Args:
        betas (torch.Tensor): Shape parameters, shape (10,)
        body_pose (torch.Tensor): Body pose (local rotations), shape (69,) = 23 joints * 3
        global_orient (torch.Tensor): Global orientation, shape (3,) - ignored for shape keys
        transl (torch.Tensor): Global translation, shape (3,) - ignored for shape keys
        device (str): Device for computation

    Returns:
        torch.Tensor: Shape key values, shape (218,)
    """
    betas = betas.to(device)
    body_pose = body_pose.to(device)
    global_orient = global_orient.to(device)
    transl = transl.to(device)

    num_pose_keys = 207  # 9 * 23 joints
    num_shape_keys = 10
    total_keys = 1 + num_shape_keys + num_pose_keys  # 218

    values = torch.zeros(total_keys, device=device)
    values[0] = 1.0  # Basis always active

    # Shape blend shapes: direct copy of betas
    values[1:1 + num_shape_keys] = betas

    # Pose corrective coefficients: vec(R(θ_local) - I) for local body joints
    num_joints = 23
    theta_local = body_pose.view(num_joints, 3)  # (23, 3)

    # Identity vector for one rotation matrix: [1,0,0, 0,1,0, 0,0,1]
    identity_vec = torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 1.], device=device)

    # Compute flattened rotation vectors
    rot_vecs = []
    for i in range(num_joints):
        axis_angle = theta_local[i]
        rot_mat = axis_angle_to_rot_mat(axis_angle)
        rot_vec = rot_mat.view(-1)  # Flatten to 9 elements
        rot_vecs.append(rot_vec)

    rot_vecs = torch.stack(rot_vecs).view(-1)  # (207,)

    # Subtract identity (repeated for all joints)
    identity_all = identity_vec.repeat(num_joints)  # (207,)
    pose_coeffs = rot_vecs - identity_all

    # Assign to pose corrective shape keys
    values[1 + num_shape_keys:] = pose_coeffs

def fit_smpl_to_obj(obj_path, smpl_model_path, gender, device, scale_factor=1.0):
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

    # params = {
    #     'betas': betas.detach().cpu().numpy(),
    #     'body_pose': body_pose.detach().cpu().numpy(),
    #     'global_orient': global_orient.detach().cpu().numpy(),
    #     'transl': transl.detach().cpu().numpy(),
    #     'gender': gender,
    #     'scale_factor': scale_factor
    # }

    params = {
        'betas': betas.detach(),
        'body_pose': body_pose.detach(),
        'global_orient': global_orient.detach(),
        'transl': transl.detach(),
    }

    # # Save parameters
    # np.savez(obj_path.replace('.obj', '_params2.npz'), **params)

    # return params, output.vertices[0].detach().cpu().numpy(), mesh.faces
    return smpl, params

def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')

def clear_scene():
    ensure_object_mode()
    print("🧹 Clearing current Blender scene...")

    # Delete objects (safe: updates deps/refs)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Purge orphans (safe: Blender-managed)
    for _ in range(2):
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception:
            pass

    print("✅ Scene cleared.\n")

def ensure_collection(col_name):
    if col_name in bpy.data.collections:
        col = bpy.data.collections[col_name]
    else:
        col = bpy.data.collections.new(col_name)
        bpy.context.scene.collection.children.link(col)
    return col

def import_fbx(fbx_file_name):
    if not os.path.exists(fbx_file_name):
        raise FileNotFoundError(fbx_file_name)
    print(f"📂 Importing FBX from {fbx_file_name} ...")
    ensure_object_mode()
    # --- Import FBX ---
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=fbx_file_name, automatic_bone_orientation=AUTOMATIC_BONE_ORIENTATION)
    after = set(bpy.data.objects)
    new_fbx = list(after - before)
    if not new_fbx:
        raise RuntimeError("FBX import produced no objects.")

    obj = new_fbx[0]
    return obj

def select_smpl_mesh(fbx_temp_obj):
    # Get the child mesh by name
    mesh_name = "SMPL-mesh-female"
    mesh = next((child for child in fbx_temp_obj.children if child.name == mesh_name), None)

    if mesh:
        print(f"Found mesh child: {mesh.name} (type: {mesh.type})")
        # Now use 'mesh' for further operations, e.g., make_rigid(mesh)
    else:
        print(f"Error: No child named '{mesh_name}' found under {obj.name}.")
        # Fallback: Search entire scene (in case not direct child)
        mesh = bpy.data.objects.get(mesh_name)
        if mesh:
            print(f"Found mesh in scene: {mesh.name}")
        else:
            print(f"Error: '{mesh_name}' not found anywhere.")

# Main workflow
if __name__ == "__main__":
    # Fit SMPL
    model, params = fit_smpl_to_obj(
        OBJ_BODY_PATH, SMPL_MODEL_PATH,
        gender=SMPL_GENDER,device=DEVICE , scale_factor=SCALE_FACTOR
    )

    # Clear Blender scene
    clear_scene()

    # Load Template FBX
    fbx_temp_obj = import_fbx(FBX_TEMPLATE_PATH_N_PREFIX+SMPL_GENDER.lower()+".fbx")

    shape_key_values = compute_smpl_shape_key_values(params['betas'], params['body_pose'],
                                                     params['global_orient'], params['transl'], device=DEVICE)

    select_smpl_mesh(fbx_temp_obj)




