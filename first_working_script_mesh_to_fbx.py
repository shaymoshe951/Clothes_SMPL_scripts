import sys, os, math
from mathutils import Matrix, Vector, Quaternion, Euler
import numpy as np
import bpy

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SMPL_MODEL_FOLDER = r"D:\projects\ClProjects\SMPL_Model"
SMPL_GENDER       = "FEMALE"               # "MALE" | "FEMALE" | "NEUTRAL"
SMPL_PARAMS_FILE  = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified_params.npz"
OUTPUT_FBX        = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m.fbx"

EXTRA_PY_PATHS = [r"C:\Users\Lab\AppData\Roaming\Python\Python311\Scripts" + "\\..\\site-packages"]
NUM_BETAS = 10
UNIT_SCALE = 1.0

# Fix SMPL → Blender frame (Z-forward → Z-up)
FIX_SMPL_FRAME = True

# ──────────────────────────────────────────────────────────────────────────────
# Imports
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
    for obj in list(bpy.data.objects):
        if obj.name.startswith(prefix):
            bpy.data.objects.remove(obj, do_unlink=True)

def create_armature(name, joint_names, joint_rest_locs, kintree):
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
    arm_obj = bpy.context.object
    arm_obj.name = name
    arm = arm_obj.data
    arm.name = name + "_Data"

    bpy.ops.object.mode_set(mode='EDIT')
    ebones = {}
    for i, jn in enumerate(joint_names):
        eb = arm.edit_bones.new(jn)
        head = Vector(joint_rest_locs[i])
        tail = head + Vector((0, 0.02, 0))
        eb.head = head
        eb.tail = tail
        ebones[i] = eb

    for child_idx in range(1, len(joint_names)):
        parent_idx = kintree[0, child_idx]
        ebones[child_idx].parent = ebones[parent_idx]

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj

def add_vertex_groups_and_weights(obj, joint_names, weights):
    V, J = weights.shape
    vg_map = {j: obj.vertex_groups.new(name=jn) for j, jn in enumerate(joint_names)}
    for v_idx in range(V):
        w_row = weights[v_idx]
        nz = np.nonzero(w_row > 0.0)[0]
        for j in nz:
            vg_map[j].add([v_idx], float(w_row[j]), 'REPLACE')

def add_armature_modifier(mesh_obj, arm_obj):
    mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = arm_obj
    mod.use_deform_preserve_volume = True
    return mod

def create_betas_shape_keys(mesh_obj, shapedirs, num_betas=10, beta_values=None):
    me = mesh_obj.data
    if not me.shape_keys:
        mesh_obj.shape_key_add(name="Basis", from_mix=False)

    V = len(me.vertices)
    NB = min(num_betas, shapedirs.shape[2])
    for bi in range(NB):
        key_name = f"Beta_{bi:02d}"
        sk = mesh_obj.shape_key_add(name=key_name, from_mix=False)
        delta = shapedirs[:, :, bi]
        for vid in range(V):
            sk.data[vid].co = me.vertices[vid].co + Vector(delta[vid])

    if beta_values is not None:
        for bi in range(NB):
            val = float(beta_values[bi])
            mesh_obj.data.shape_keys.key_blocks[f"Beta_{bi:02d}"].value = val

# --- Pose Application Helpers ---
def axis_angle_to_blender_tuple(aax):
    theta = float(np.linalg.norm(aax))
    if theta < 1e-8:
        return [0.0, 1.0, 0.0, 0.0]
    axis = (aax / theta).astype(np.float64)
    return [theta, float(axis[0]), float(axis[1]), float(axis[2])]

def apply_smpl_pose_to_armature(arm_obj, joint_names, global_orient, body_pose):
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    pb = arm_obj.pose.bones

    # Frame correction for SMPL (Z-forward → Z-up)
    if FIX_SMPL_FRAME:
        fix_q = Euler((math.radians(90.0), 0.0, 0.0), 'XYZ').to_quaternion()
        root_q = axis_angle_to_quaternion(global_orient)
        root_q = fix_q @ root_q
        global_orient_tuple = quaternion_to_axis_angle(root_q)
    else:
        global_orient_tuple = axis_angle_to_blender_tuple(global_orient)

    if "Pelvis" in pb:
        pb["Pelvis"].rotation_mode = 'AXIS_ANGLE'
        pb["Pelvis"].rotation_axis_angle = global_orient_tuple

    body = body_pose.reshape(23, 3)
    for i in range(23):
        jname = joint_names[i+1]
        if jname in pb:
            pb[jname].rotation_mode = 'AXIS_ANGLE'
            pb[jname].rotation_axis_angle = axis_angle_to_blender_tuple(body[i])

    bpy.ops.object.mode_set(mode='OBJECT')

def axis_angle_to_quaternion(aax):
    theta = float(np.linalg.norm(aax))
    if theta < 1e-8:
        return Quaternion((1, 0, 0, 0))
    axis = (aax / theta).astype(np.float64)
    return Quaternion((math.cos(theta/2.0), *(math.sin(theta/2.0)*axis)))

def quaternion_to_axis_angle(q):
    q = q.normalized()
    angle = 2*math.acos(max(min(q.w,1.0),-1.0))
    s = math.sqrt(max(1 - q.w*q.w, 0.0))
    if s < 1e-8:
        return [0.0, 1.0, 0.0, 0.0]
    return [angle, q.x/s, q.y/s, q.z/s]

# --- FBX Export ---
def export_fbx(path):
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
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Y',   # Blender forward
        axis_up='Z'          # Blender up
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    clear_scene_objects(prefix="SMPL_")
    device = torch.device('cpu')
    model = SMPL(model_path=SMPL_MODEL_FOLDER, gender=SMPL_GENDER.lower()).to(device)
    faces = model.faces.astype(np.int32)

    data = np.load(SMPL_PARAMS_FILE, allow_pickle=True)
    def _get(name, default_shape=None):
        m = {
            'betas': ['betas', 'shape', 'betas_numpy'],
            'body_pose': ['body_pose', 'pose_body', 'bodyPose'],
            'global_orient': ['global_orient', 'root_orient', 'pose_root', 'globalOrient'],
            'transl': ['transl', 'translation', 'trans'],
        }[name]
        for k in m:
            if k in data:
                arr = np.array(data[k])
                return np.asarray(arr).reshape(-1)
        return np.zeros(default_shape, dtype=np.float32)

    betas = _get('betas', (NUM_BETAS,))
    if betas.shape[0] < NUM_BETAS:
        betas = np.concatenate([betas, np.zeros(NUM_BETAS - betas.shape[0])])
    betas_t = torch.from_numpy(betas[:NUM_BETAS]).float().unsqueeze(0)
    body_pose = _get('body_pose', (69,))
    global_orient = _get('global_orient', (3,))
    transl = _get('transl', (3,))
    body_pose_t = torch.from_numpy(body_pose).float().unsqueeze(0)
    global_orient_t = torch.from_numpy(global_orient).float().unsqueeze(0)
    transl_t = torch.from_numpy(transl * UNIT_SCALE).float().unsqueeze(0)

    with torch.no_grad():
        out = model(betas=betas_t, body_pose=body_pose_t, global_orient=global_orient_t,
                    transl=transl_t, pose2rot=True)
        verts_posed = out.vertices[0].cpu().numpy() * UNIT_SCALE
        joints_global = out.joints[0].cpu().numpy() * UNIT_SCALE

    with torch.no_grad():
        out_neutral = model(
            betas=torch.zeros_like(betas_t),
            body_pose=torch.zeros_like(body_pose_t),
            global_orient=torch.zeros_like(global_orient_t),
            transl=torch.zeros_like(transl_t),
            pose2rot=True)
        verts_mean = out_neutral.vertices[0].cpu().numpy() * UNIT_SCALE
        joints_rest = out_neutral.joints[0].cpu().numpy() * UNIT_SCALE

    shapedirs = model.shapedirs
    if shapedirs.dim() == 4 and shapedirs.shape[0] == 1:
        shapedirs = shapedirs[0]
    shapedirs_np = shapedirs[:, :, :NUM_BETAS].cpu().numpy() * UNIT_SCALE
    lbs_weights = model.lbs_weights.cpu().numpy()
    J = lbs_weights.shape[1]

    kintree_table = np.vstack([np.concatenate([[-1], np.arange(J-1)]), np.arange(J)])
    joint_names = [
        "Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle",
        "Spine3","L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder",
        "L_Elbow","R_Elbow","L_Wrist","R_Wrist","L_Hand","R_Hand"
    ]
    if J != len(joint_names):
        joint_names = [f"J{i}" for i in range(J)]

    col = ensure_collection("SMPL_Collection")
    mesh_obj = np_to_bmesh_object("SMPL_Mesh", verts_mean, faces)
    col.objects.link(mesh_obj)

    arm_obj = create_armature("SMPL_Armature", joint_names, joints_rest, kintree_table)
    col.objects.link(arm_obj)

    add_armature_modifier(mesh_obj, arm_obj)
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE', keep_transform=True)
    mesh_obj.select_set(False)
    arm_obj.select_set(False)

    add_vertex_groups_and_weights(mesh_obj, joint_names, lbs_weights)
    create_betas_shape_keys(mesh_obj, shapedirs_np, num_betas=NUM_BETAS, beta_values=betas)

    apply_smpl_pose_to_armature(arm_obj, joint_names, global_orient, body_pose)
    arm_obj.location = Vector((float(transl[0]), float(transl[1]), float(transl[2])))

    bpy.context.view_layer.update()
    export_fbx(OUTPUT_FBX)
    print(f"[OK] Exported FBX to: {OUTPUT_FBX}")

if __name__ == "__main__":
    main()
