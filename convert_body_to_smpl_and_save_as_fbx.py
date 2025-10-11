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
from mathutils import Quaternion, Matrix, Vector
import numpy as np
import bpy

# ---------- Config ----------
OBJ_BODY_PATH = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"
SMPL_MODEL_PATH = r"D:\projects\ClProjects\SMPL_Model"  # Download from SMPL website
SMPL_GENDER = "FEMALE"  # "MALE" | "FEMALE" | "NEUTRAL"
OUTPUT_FBX = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m2.fbx"
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

    num_pose_keys = 207  # 9 * 23 joints
    num_shape_keys = 10
    total_keys = 1 + num_shape_keys + num_pose_keys  # 218

    values = torch.zeros(total_keys, device=device)
    values[0] = 1.0  # Basis always active

    # Shape keys: betas
    values[1:1 + num_shape_keys] = betas

    # Pose correctives: vec(R(theta_local) - I) for 23 local joints (body_pose)
    num_joints_local = 23
    theta_local = body_pose.view(num_joints_local, 3)

    identity_vec = torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 1.], device=device)

    rot_vecs = []
    for i in range(num_joints_local):
        rot_mat = axis_angle_to_rot_mat(theta_local[i])
        rot_vec = rot_mat.view(-1)
        rot_vecs.append(rot_vec)

    rot_vecs = torch.stack(rot_vecs).view(-1)  # 207
    identity_all = identity_vec.repeat(num_joints_local)
    pose_coeffs = rot_vecs - identity_all

    values[1 + num_shape_keys:] = pose_coeffs

    return values


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
        print(f"Warning: Expected 218 keys, found {len(shape_keys)}.")
        return

    for i, key in enumerate(shape_keys):
        key.value = float(shape_key_values[i])

    bpy.context.view_layer.update()
    print(f"Updated {len(shape_keys)} shape keys on {obj.name}")


def set_smpl_pose_on_armature(armature, global_orient, body_pose, device='cpu'):
    """
    Set bone rotations on the armature using full pose (global + body).
    Assumes standard SMPL joint order and bone names.
    """
    if armature.type != 'ARMATURE':
        print("Error: Object must be an armature.")
        return

    # Standard SMPL joint names (adjust if your import uses prefixes like 'mixamorig:')
    joint_names = [
        'pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee',
        'spine2', 'l_ankle', 'r_ankle', 'spine3', 'l_foot', 'r_foot',
        'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
        'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand'
    ]

    # Full theta: global (root) + body_pose (23 locals)
    theta_full = torch.cat([global_orient.to(device), body_pose.to(device)]).view(24, 3)

    # Switch to Pose Mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    # Clear existing rotations (optional: armature.pose_library.reset())
    for bone in armature.pose.bones:
        bone.rotation_quaternion = Quaternion((1, 0, 0, 0))

    # Set per-bone rotations
    success_count = 0
    joint_names_dict = {jn.lower(): i for i, jn in enumerate(joint_names)}
    #    print(joint_names_dict)
    # for i, joint_name in enumerate(joint_names):
    #     if joint_name in [b.name for b in armature.pose.bones]:
    for bone_name in [b.name for b in armature.pose.bones]:
        if bone_name.lower() in joint_names_dict:
            bone = armature.pose.bones[bone_name]
            ind = joint_names_dict[bone_name.lower()]

            aa_torch = theta_full[ind]  # Keep as torch for function
            aa_np = aa_torch.cpu().numpy()  # For potential numpy ops

            if np.linalg.norm(aa_np) > 1e-6:  # Non-zero
                R_torch = axis_angle_to_rot_mat(aa_torch)  # (3,3) or (1,3,3)

                # Ensure R is (3,3) numpy
                if R_torch.dim() == 3 and R_torch.shape[0] == 1:
                    R_torch = R_torch.squeeze(0)
                R = R_torch.cpu().numpy()  # Now (3,3)

                print(f"For bone {bone_name}: R.shape = {R.shape}")  # Debug: should be (3,3)

                # Convert matrix to quaternion properly
                rot_mat = Matrix(R.tolist())  # List for Matrix constructor
                quat = rot_mat.to_quaternion()
                bone.rotation_quaternion = quat
                success_count += 1
            else:
                # Zero rotation: identity quat
                bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
                success_count += 1
        else:
            print(f"Warning: Bone '{bone_name}' not found.")

    bpy.ops.object.mode_set(mode='OBJECT')  # Back to Object Mode
    bpy.context.view_layer.update()
    print(f"Set rotations for {success_count}/24 bones.")


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
    obj = bpy.data.objects["SMPL-female"]  # Or however you have it

    # Get the child mesh by name
    mesh_name = "SMPL-mesh-female"
    mesh = next((child for child in obj.children if child.name == mesh_name), None)
    armature = obj

    if mesh:
        print(f"Found mesh child: {mesh.name} (type: {mesh.type})")

    return mesh, armature

def export_fbx(fbx_output_filename):
    bpy.ops.export_scene.fbx(
        filepath=fbx_output_filename,
        use_selection=False,
        apply_unit_scale=True,
        bake_space_transform=False,  # Blender→Blender round-trip
        add_leaf_bones=False,
        armature_nodetype='NULL',
        use_armature_deform_only=False,
        mesh_smooth_type='FACE',
        use_mesh_modifiers=False,
        apply_scale_options='FBX_SCALE_ALL',
    )
    print(f"[OK] Exported FBX to: {fbx_output_filename}")

# Main workflow
if __name__ == "__main__":
    # Fit SMPL
    model, params = fit_smpl_to_obj(
        OBJ_BODY_PATH, SMPL_MODEL_PATH,
        gender=SMPL_GENDER, device=DEVICE, scale_factor=SCALE_FACTOR
    )

    # Clear Blender scene
    clear_scene()

    # Load Template FBX
    fbx_temp_obj = import_fbx(FBX_TEMPLATE_PATH_N_PREFIX + SMPL_GENDER.lower() + ".fbx")

    shape_key_values = compute_smpl_shape_key_values(params['betas'], params['body_pose'],
                                                     params['global_orient'], params['transl'], device=DEVICE)

    mesh_obj, armature_obj = select_smpl_mesh(fbx_temp_obj)
    set_blender_shape_keys(mesh_obj, shape_key_values)

    # Set bone poses
    set_smpl_pose_on_armature(armature_obj, params['global_orient'], params['body_pose'], DEVICE)

    # Apply translation (to armature or mesh; here to armature)
    armature_obj.location = Vector(params['transl'].cpu().numpy())

    bpy.context.view_layer.update()

    export_fbx(OUTPUT_FBX)

    print("Done!")

