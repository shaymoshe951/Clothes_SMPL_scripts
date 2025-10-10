import sys, os, json, math
from mathutils import Matrix, Vector, Quaternion
import numpy as np
import bpy

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG (edit these paths)
# ──────────────────────────────────────────────────────────────────────────────
SMPL_GENDER       = "FEMALE"               # "MALE" | "FEMALE" | "NEUTRAL"
SMPL_PARAMS_FILE = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified_params.npz"
SMPL_MODEL_FOLDER = r"D:\projects\ClProjects\SMPL_Model"  # Folder containing preprocessed SMPL_FEMALE.npz, etc.
OUTPUT_FBX = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m.fbx"


# If smplx/torch aren't importable by Blender, add site-packages here:
EXTRA_PY_PATHS = [
    r"D:\venv\py310\Lib\site-packages",    # example
]

# Create 10 beta keys (standard). Increase if your model supports more.
NUM_BETAS = 10

# Unit scale: SMPL is in meters. Blender default unit is meters, so keep 1.0.
UNIT_SCALE = 1.0

# ──────────────────────────────────────────────────────────────────────────────
# Imports (after path fixes)
# ──────────────────────────────────────────────────────────────────────────────
for p in EXTRA_PY_PATHS:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

import torch
from smplx import SMPL

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def np_to_bmesh_object(name, verts_np, faces_np):
    """Create a Mesh Object from numpy arrays."""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts_np.tolist(), [], faces_np.astype(int).tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj

def ensure_collection(col_name):
    if col_name in bpy.data.collections:
        col = bpy.data.collections[col_name]
    else:
        col = bpy.data.collections.new(col_name)
        bpy.context.scene.collection.children.link(col)
    return col

def clear_scene_objects(prefix="SMPL_"):
    # Remove previous objects created by this script
    for obj in list(bpy.data.objects):
        if obj.name.startswith(prefix):
            bpy.data.objects.remove(obj, do_unlink=True)

def create_armature(name, joint_names, joint_rest_locs, kintree):
    """Create an Armature with given joint hierarchy and rest joint positions."""
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
    arm_obj = bpy.context.object
    arm_obj.name = name
    arm = arm_obj.data
    arm.name = name + "_Data"

    # Make sure we're in Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Create edit bones at joint positions
    ebones = {}
    for i, jn in enumerate(joint_names):
        eb = arm.edit_bones.new(jn)
        # Tiny bone pointing upwards; we'll set head at joint and a small tail offset
        head = Vector(joint_rest_locs[i])
        tail = head + Vector((0, 0.01, 0))  # small tail so bone isn't zero-length
        eb.head = head
        eb.tail = tail
        ebones[i] = eb

    # Parent according to kinematic tree (parent index, child index)
    for child_idx in range(1, len(joint_names)):
        parent_idx = kintree[0, child_idx]
        ebones[child_idx].parent = ebones[parent_idx]

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj

def add_vertex_groups_and_weights(obj, joint_names, weights):  # weights: [V, J]
    """Create vertex groups and assign LBS weights per joint."""
    mesh = obj.data
    V, J = weights.shape
    # Create groups
    vg_map = {}
    for j, jn in enumerate(joint_names):
        vg = obj.vertex_groups.new(name=jn)
        vg_map[j] = vg

    # Assign weights (sparse-write for efficiency)
    # For each vertex, add its weight for each joint (> 0)
    for v_idx in range(V):
        w_row = weights[v_idx]
        nz = np.nonzero(w_row > 0.0)[0]
        for j in nz:
            vg_map[j].add([v_idx], float(w_row[j]), 'REPLACE')

def add_armature_modifier(mesh_obj, arm_obj):
    mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = arm_obj
    # Keep "Preserve Volume" disabled by default; enable if you prefer DQ skinning look
    return mod

def create_betas_shape_keys(mesh_obj, shapedirs, num_betas=10, beta_values=None):
    """
    Create shape keys Beta_00..Beta_09 from shapedirs (V, 3, NB).
    Basis is the mean shape (betas=0). Set key values to beta_values if provided.
    """
    me = mesh_obj.data

    # Ensure a Basis key exists that matches the mean (betas=0). The current mesh is already mean if we used betas=0.
    if not me.shape_keys:
        mesh_obj.shape_key_add(name="Basis", from_mix=False)

    V = len(me.vertices)
    # Build one key per beta dimension
    NB = min(num_betas, shapedirs.shape[2])
    for bi in range(NB):
        key_name = f"Beta_{bi:02d}"
        sk = mesh_obj.shape_key_add(name=key_name, from_mix=False)
        # delta for this beta is shapedirs[:,:,bi]
        delta = shapedirs[:, :, bi]  # (V, 3)
        for vid in range(V):
            sk.data[vid].co = me.vertices[vid].co + Vector(delta[vid])

    # Set initial slider values to match the provided beta coefficients
    if beta_values is not None:
        for bi in range(NB):
            val = float(beta_values[bi])
            mesh_obj.data.shape_keys.key_blocks[f"Beta_{bi:02d}"].value = val

import math
from mathutils import Vector

def axis_angle_to_blender_tuple(aax):
    """SMPL axis-angle (3,) -> Blender rotation_axis_angle tuple [angle, x, y, z]."""
    # aax is Rodrigues vector r = theta * n (axis * angle)
    theta = float(np.linalg.norm(aax))
    if theta < 1e-8:
        return [0.0, 1.0, 0.0, 0.0]
    axis = (aax / theta).astype(np.float64)
    return [theta, float(axis[0]), float(axis[1]), float(axis[2])]

def apply_smpl_pose_to_armature(arm_obj, joint_names, global_orient, body_pose):
    """
    Set Pose Mode rotations from SMPL axis-angle parameters.
    - global_orient: (3,)
    - body_pose: (23*3,)
    The joint order here must match joint_names:
        Pelvis, L_Hip, R_Hip, Spine1, L_Knee, R_Knee, Spine2, L_Ankle, R_Ankle,
        Spine3, L_Foot, R_Foot, Neck, L_Collar, R_Collar, Head,
        L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist, L_Hand, R_Hand
    """
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # Map bone names to pose bones
    pb = arm_obj.pose.bones
    # 0: Pelvis uses global_orient
    if "Pelvis" in pb:
        pb["Pelvis"].rotation_mode = 'AXIS_ANGLE'
        pb["Pelvis"].rotation_axis_angle = axis_angle_to_blender_tuple(global_orient)

    # Remaining 23 joints from body_pose
    body = body_pose.reshape(23, 3)
    for i in range(23):
        jname = joint_names[i+1]  # skip Pelvis
        if jname in pb:
            pb[jname].rotation_mode = 'AXIS_ANGLE'
            pb[jname].rotation_axis_angle = axis_angle_to_blender_tuple(body[i])

    bpy.ops.object.mode_set(mode='OBJECT')

def export_fbx(path):
    # Key FBX options: embed textures False (no textures here), apply scalings FBX units
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=False,
        apply_unit_scale=True,
        bake_space_transform=False,
        add_leaf_bones=False,
        armature_nodetype='NULL',
        use_armature_deform_only=True,
        mesh_smooth_type='FACE',
        use_mesh_modifiers=True,
        use_mesh_modifiers_render=True,
        apply_scale_options='FBX_SCALE_ALL'
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    clear_scene_objects(prefix="SMPL_")

    # Load SMPL model
    device = torch.device('cpu')
    model = SMPL(model_path=SMPL_MODEL_FOLDER, gender=SMPL_GENDER.lower()).to(device)  # smplx.SMPL expects lowercase gender
    faces = model.faces.astype(np.int32)

    # Load SMPL params
    # Expect: betas [B,10], body_pose [B,69], global_orient [B,3], transl [B,3]
    # Accept either with batch or flat; we’ll take first if batched
    if SMPL_PARAMS_FILE.endswith(".npz"):
        data = np.load(SMPL_PARAMS_FILE, allow_pickle=True)
    elif SMPL_PARAMS_FILE.endswith(".pkl"):
        import pickle
        with open(SMPL_PARAMS_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError("SMPL_PARAMS_FILE must be .npz or .pkl")

    def _get(name, default_shape=None):
        if name in data:
            arr = np.array(data[name])
        else:
            # allow different keys
            m = {
                'betas': ['betas', 'shape', 'betas_numpy'],
                'body_pose': ['body_pose', 'pose_body', 'bodyPose'],
                'global_orient': ['global_orient', 'root_orient', 'pose_root', 'globalOrient'],
                'transl': ['transl', 'translation', 'trans'],
            }[name]
            found = None
            for k in m:
                if k in data:
                    found = np.array(data[k])
                    break
            if found is None:
                if default_shape is None:
                    raise KeyError(f"Missing SMPL param '{name}' and no alternative key found.")
                found = np.zeros(default_shape, dtype=np.float32)
            arr = found
        # Squeeze to [D]
        arr = np.asarray(arr).reshape(-1)
        return arr

    betas = _get('betas', default_shape=(NUM_BETAS,))
    if betas.shape[0] < NUM_BETAS:
        pad = np.zeros(NUM_BETAS - betas.shape[0], dtype=np.float32)
        betas = np.concatenate([betas, pad], axis=0)
    betas_t = torch.from_numpy(betas[:NUM_BETAS]).float().unsqueeze(0)

    body_pose = _get('body_pose', default_shape=(69,))  # 23*3 axis-angle
    global_orient = _get('global_orient', default_shape=(3,))
    transl = _get('transl', default_shape=(3,))

    body_pose_t = torch.from_numpy(body_pose).float().unsqueeze(0)
    global_orient_t = torch.from_numpy(global_orient).float().unsqueeze(0)
    transl_t = torch.from_numpy(transl * UNIT_SCALE).float().unsqueeze(0)

    # Build posed output
    with torch.no_grad():
        out = model(
            betas=betas_t,
            body_pose=body_pose_t,
            global_orient=global_orient_t,
            transl=transl_t,
            pose2rot=True
        )
        # Vertices already include LBS; joints is regressed
        verts_posed = out.vertices[0].cpu().numpy() * UNIT_SCALE
        joints_global = out.joints[0].cpu().numpy() * UNIT_SCALE

    # We also need shapedirs (V,3,NB) and weights (V,J) for skinning + beta keys around mean (betas=0)
    # shapedirs in smplx.SMPL: model.shapedirs: [1, V, 3, NB] or [V, 3, NB] depending on version; normalize shape
    # We compute mean-shape vertices (betas=0) for the Basis
    with torch.no_grad():
        out_neutral = model(
            betas=torch.zeros_like(betas_t),
            body_pose=torch.zeros_like(body_pose_t),
            global_orient=torch.zeros_like(global_orient_t),
            transl=torch.zeros_like(transl_t),
            pose2rot=True
        )
        verts_mean = out_neutral.vertices[0].cpu().numpy() * UNIT_SCALE
        joints_rest = out_neutral.joints[0].cpu().numpy() * UNIT_SCALE

    # Extract shapedirs and lbs weights
    # shapedirs is torch tensor [V, 3, NB] (some versions wrap a leading 1-dim)
    shapedirs = model.shapedirs
    if shapedirs.dim() == 4 and shapedirs.shape[0] == 1:
        shapedirs = shapedirs[0]
    shapedirs_np = shapedirs[:, :, :NUM_BETAS].cpu().numpy() * UNIT_SCALE  # (V,3,NB)

    lbs_weights = model.lbs_weights.cpu().numpy()  # (V, J)
    J = lbs_weights.shape[1]

    # Kinematic tree (parents): usually a tensor [2, 24], row 0 = parent index
    kintree = model.parents.clone()
    # Convert to 2 x J format: parent in row 0; indices 0..J-1 in row 1
    # Many SMPL wrappers store parents as length-J vector; we’ll make (2, J) for convenience
    parents = kintree.cpu().numpy()
    parents[0] = -1  # root parent fix if needed
    parents_vec = parents
    # If model.parents is length J vector, build a 2xJ table
    if parents_vec.ndim == 1:
        ptab = np.vstack([parents_vec, np.arange(len(parents_vec))])[None, ...]
        # reduce extra dim
        ptab = np.squeeze(ptab, axis=0)
        # Make it shape (2, J)
        kintree_table = np.vstack([ptab[0:1, :], ptab[1:2, :]])
    else:
        # Fallback: construct trivial chain
        kintree_table = np.vstack([np.concatenate([[-1], np.arange(J-1)]), np.arange(J)])

    # Joint names (SMPL’s canonical 24)
    joint_names = [
        "Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle",
        "Spine3","L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder",
        "L_Elbow","R_Elbow","L_Wrist","R_Wrist","L_Hand","R_Hand"
    ]
    if J != len(joint_names):
        # If your SMPL has different joint count/names, adjust here:
        joint_names = [f"J{i}" for i in range(J)]

    # ── Build objects in Blender ──────────────────────────────────────────────
    col = ensure_collection("SMPL_Collection")

    # Create mesh at MEAN/BASIS first (for clean shape keys). Then we’ll snap/pose.
    mesh_obj = np_to_bmesh_object("SMPL_Mesh", verts_mean, faces)
    mesh_obj.location = (0, 0, 0)
    col.objects.link(mesh_obj)

    # Create armature in REST pose using rest joints (mean T-pose)
    arm_obj = create_armature("SMPL_Armature", joint_names, joints_rest, kintree_table)
    col.objects.link(arm_obj)

    # Parent mesh to armature (deform)
    # - Modifier first:
    add_armature_modifier(mesh_obj, arm_obj)
    # - And parent relation (good practice to keep transforms tied)
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE', keep_transform=True)
    mesh_obj.select_set(False)
    arm_obj.select_set(False)

    # Add vertex groups & weights
    add_vertex_groups_and_weights(mesh_obj, joint_names, lbs_weights)

    # Create beta shape keys around Basis and set their initial values
    create_betas_shape_keys(mesh_obj, shapedirs_np, num_betas=NUM_BETAS, beta_values=betas)

    # Apply the SMPL pose in Pose Mode (rotations)
    apply_smpl_pose_to_armature(
        arm_obj,
        joint_names,
        global_orient=global_orient,          # (3,)
        body_pose=body_pose                   # (69,) = 23*3
    )

    # Apply the global translation at the object level (meters)
    arm_obj.location = Vector((float(transl[0]), float(transl[1]), float(transl[2])))

    # Optionally place the whole rig at transl (already baked in the out.vertices -> joints_global),
    # but we’ve used joints_global which already includes transl. So object origins can remain at (0,0,0).

    # Make sure everything is updated
    bpy.context.view_layer.update()

    # ── Export FBX ───────────────────────────────────────────────────────────
    export_fbx(OUTPUT_FBX)
    print(f"[OK] Exported FBX to: {OUTPUT_FBX}")

if __name__ == "__main__":
    main()
