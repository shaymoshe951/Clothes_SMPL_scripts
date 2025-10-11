######### NOT_USED (generate replacement for .npz to avoid import of chumpy)
######## Done by Claude
import os
import json
import numpy as np
import torch
from smplx import SMPL
import trimesh


# ---------- Config ----------
OBJ_PATH = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"
SMPL_MODEL_DIR = r"D:\projects\ClProjects\SMPL_Model"        # folder you downloaded from SMPL website
GENDER = "female"                                            # 'neutral' | 'male' | 'female'
DEVICE = "cpu"                                               # 'cpu' is fine for this
SCALE_FACTOR = 0.01                                          # scales your input OBJ vertices before fitting

# Where to save outputs; defaults next to the OBJ
OUT_DIR = os.path.splitext(OBJ_PATH)[0] + "_smpl_out"
os.makedirs(OUT_DIR, exist_ok=True)
# ----------------------------


def robust_get_parents(model):
    """Return parents (24,) as numpy with -1 for root."""
    # smplx.Smpl has either .parents or .kintree_table (2x24 with [0] = parents)
    try:
        parents = model.parents.detach().cpu().numpy().astype(np.int32)
    except Exception:
        try:
            parents = model.kintree_table[0].detach().cpu().numpy().astype(np.int32)
        except Exception:
            raise RuntimeError("Could not find SMPL parents / kintree_table on the model.")
    # ensure root has -1 (SMPL root is pelvis index 0)
    if parents[0] != -1:
        parents = parents.copy()
        parents[0] = -1
    return parents


def robust_get_lbs_weights(model):
    """Return (N, 24) numpy LBS weights."""
    try:
        w = model.lbs_weights.detach().cpu().numpy()
    except Exception:
        try:
            w = model.weights.detach().cpu().numpy()
        except Exception:
            raise RuntimeError("Could not find SMPL LBS weights (lbs_weights or weights).")
    return w


def export_smpl_assets(model, out_dir):
    """
    Save torch-free assets for Blender:
    vertices_base.npy, faces.npy, shapedirs.npy, lbs_weights.npy,
    kintree_parents.npy, joint_names.json, v_template.npy
    """
    device = next(model.parameters()).device

    # Base T-pose vertices at betas=0, pose=0, transl=0
    betas0 = torch.zeros(10, dtype=torch.float32, device=device)   # 10 betas
    pose0  = torch.zeros(72, dtype=torch.float32, device=device)   # 24*3 axis-angles
    transl0 = torch.zeros(3, dtype=torch.float32, device=device)

    with torch.no_grad():
        out_base = model(
            betas=betas0[None],
            body_pose=pose0[3:][None],
            global_orient=pose0[:3][None],
            transl=transl0[None],
            return_full_pose=True
        )

    vertices_base = out_base.vertices[0].detach().cpu().numpy()      # (6890, 3)
    faces = model.faces                                              # (F, 3) numpy
    shapedirs = model.shapedirs.detach().cpu().numpy()               # (6890, 3, num_betas)
    lbs_weights = robust_get_lbs_weights(model)                      # (6890, 24)
    parents = robust_get_parents(model)                              # (24,)
    # v_template if present (some builds expose it)
    v_template = getattr(model, "v_template", None)
    if v_template is not None:
        v_template = v_template.detach().cpu().numpy()               # (6890, 3)

    # Canonical SMPL 24-joint names (matching the parents order)
    joint_names = [
        "Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle","Spine3",
        "L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder","L_Elbow",
        "R_Elbow","L_Wrist","R_Wrist","L_Hand","R_Hand"
    ]

    np.save(os.path.join(out_dir, "vertices_base.npy"), vertices_base)
    np.save(os.path.join(out_dir, "faces.npy"), faces.astype(np.int32))
    np.save(os.path.join(out_dir, "shapedirs.npy"), shapedirs.astype(np.float32))
    np.save(os.path.join(out_dir, "lbs_weights.npy"), lbs_weights.astype(np.float32))
    np.save(os.path.join(out_dir, "kintree_parents.npy"), parents.astype(np.int32))
    if v_template is not None:
        np.save(os.path.join(out_dir, "v_template.npy"), v_template.astype(np.float32))
    with open(os.path.join(out_dir, "joint_names.json"), "w", encoding="utf-8") as f:
        json.dump(joint_names, f, ensure_ascii=False, indent=2)

    print(f"[SMPL assets] Saved to: {out_dir}")
    print(" - vertices_base.npy")
    print(" - faces.npy")
    print(" - shapedirs.npy")
    print(" - lbs_weights.npy")
    print(" - kintree_parents.npy")
    print(" - joint_names.json")
    if v_template is not None:
        print(" - v_template.npy")


def fit_smpl_to_obj(obj_path, smpl_model_dir, gender='female', device='cpu', scale_factor=0.01):
    """Fit SMPL parameters to an OBJ mesh with optional scaling (same idea as your original)."""
    mesh = trimesh.load(obj_path, process=False)
    print(f"Original mesh bounds:\n{mesh.bounds}")
    mesh.vertices *= scale_factor
    print(f"Scaled mesh bounds (scale={scale_factor}):\n{mesh.bounds}")

    target_vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)

    smpl = SMPL(model_path=smpl_model_dir, gender=gender).to(device)

    # Fit variables
    betas = torch.zeros(10, requires_grad=True, device=device)
    body_pose = torch.zeros(69, requires_grad=True, device=device)   # 23*3 (excl. global_orient)
    global_orient = torch.zeros(3, requires_grad=True, device=device)
    transl = torch.zeros(3, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([betas, body_pose, global_orient, transl], lr=0.01)

    print("Fitting SMPL parameters...")
    for i in range(1000):
        optimizer.zero_grad()
        output = smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            transl=transl.unsqueeze(0)
        )
        # MSE to target obj vertices
        loss = torch.nn.functional.mse_loss(output.vertices[0], target_vertices)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i:04d} | Loss: {loss.item():.6f}")

    fitted_vertices = output.vertices[0].detach().cpu().numpy()
    faces = smpl.faces

    params = {
        'betas': betas.detach().cpu().numpy(),
        'body_pose': body_pose.detach().cpu().numpy(),
        'global_orient': global_orient.detach().cpu().numpy(),
        'transl': transl.detach().cpu().numpy(),
        'gender': gender,
        'scale_factor': scale_factor
    }
    return params, fitted_vertices, faces, smpl


def main():
    # 1) Fit like before
    params, fitted_vertices, faces, smpl_model = fit_smpl_to_obj(
        OBJ_PATH, SMPL_MODEL_DIR, gender=GENDER, device=DEVICE, scale_factor=SCALE_FACTOR
    )

    # 2) Save your NPZ params next to outputs
    npz_path = os.path.join(OUT_DIR, "fitted_params.npz")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(npz_path, **params)
    print(f"[Params] Saved: {npz_path}")

    # 3) Export reusable SMPL assets (for Blender NumPy-only pipeline)
    export_smpl_assets(smpl_model, OUT_DIR)

    # 4) (Optional) Save the fitted mesh OBJ for inspection
    try:
        fitted_mesh_path = os.path.join(OUT_DIR, "fitted_mesh.obj")
        trimesh.Trimesh(fitted_vertices, faces).export(fitted_mesh_path)
        print(f"[Mesh] Saved fitted mesh: {fitted_mesh_path}")
    except Exception as e:
        print(f"Could not save fitted OBJ (ok to ignore): {e}")


if __name__ == "__main__":
    main()
