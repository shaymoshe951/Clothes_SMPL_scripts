__filepath__ = r"C:\Users\Lab\Downloads\clothes_images\mesh_w_clothes_from_opensite\model_xs1_mesh.obj"

"""
Measure garment circumferences (waist/hip/chest/thigh) from a clothed surface mesh.

Usage (single clothed mesh):
    CLOTHED_OBJ = r"C:\path\to\clothed_mesh.obj"
    BODY_OBJ    = None  # optional; if given, ease (garment - body) is reported
    python clothes_measurements.py

Notes:
- We estimate landmark planes (waist/hip/chest/thigh) from SMPL joints regressed
  by SMPL-Anthropometry (MeasureBody). Then we slice the CLOTHED mesh at those Z
  heights and measure the dominant closed loop length (circumference).
- If BODY_OBJ is provided (preferably a fitted SMPL/SMPL-X), we slice it at the
  same Z heights and compute EASE = garment - body per band.
"""

import os
import numpy as np
import trimesh
import torch

from measure import MeasureBody

# ----------------- USER SETTINGS -----------------
CLOTHED_OBJ = __filepath__
BODY_OBJ    = r"C:\Users\Lab\Downloads\clothes_images\model_yali2_mesh_0_0.obj"  # e.g., a fitted SMPL mesh path; or None
MODEL_TYPE  = "smpl"    # "smpl" or "smplx" (used only for joint regression)
GENDER      = "FEMALE"  # NEUTRAL / MALE / FEMALE (does not change planes much)
SAVE_DEBUG  = None      # e.g., r"D:\garment_slices.glb" to export slice visuals
# -------------------------------------------------


# ---------- helpers ----------
def to_numpy(v):
    try:
        return v.detach().cpu().numpy()
    except Exception:
        return np.asarray(v)

def load_mesh(path):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        # If it's a scene, merge all geometry into a single Trimesh
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    m.remove_unreferenced_vertices()
    return m

def _maybe_get_joints(measurer):
    """
    Try to pull joints from MeasureBody (robust against minor API differences).
    Returns (J, names) where J shape is (N,3) and names is a list or None.
    """
    names = None
    # common attributes
    for attr in ["joints", "J_tr", "J", "_joints"]:
        if hasattr(measurer, attr):
            J = getattr(measurer, attr)
            J = to_numpy(J)
            if J is not None:
                if J.ndim == 3 and J.shape[-1] == 3:
                    J = J[0]
                if J.ndim == 2 and J.shape[1] == 3:
                    # try names
                    for n_attr in ["joint_names", "J_names", "joints_names", "_joint_names"]:
                        if hasattr(measurer, n_attr):
                            try:
                                names = getattr(measurer, n_attr)
                            except Exception:
                                names = None
                    return J, names
    return None, None

def z_up_transform(mesh):
    """
    Ensure Z is the up axis: if mesh's up isn't Z, rotate to make the longest
    spread axis vertical. (Heuristic; skip if your data is already Z-up.)
    """
    extents = mesh.extents  # (dx, dy, dz)
    # If Z is already the largest extent, assume Z-up
    if extents[2] >= extents[0] and extents[2] >= extents[1]:
        return mesh, np.eye(4)

    # Otherwise, rotate so the max extent becomes Z
    axes = np.argsort(extents)
    major = axes[-1]  # 0:x, 1:y, 2:z
    if major == 0:
        # X is tallest → rotate +90° around Y to put X → Z
        R = trimesh.transformations.rotation_matrix(np.deg2rad(90), [0, 1, 0])
    else:
        # Y is tallest → rotate -90° around X to put Y → Z
        R = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [1, 0, 0])
    mesh2 = mesh.copy()
    mesh2.apply_transform(R)
    return mesh2, R

def plane_section_length(mesh, z, tol=2.0, search_steps=9, up=np.array([0, 0, 1.0])):
    """
    Intersect mesh with a plane at height z (Z-up), optionally searching within
    ±tol cm for the most stable loop. Returns (best_length, best_loop_points, best_z)
    or (None, None, None) if no valid loop.

    The 'stability' is simply the longest closed loop length at nearby heights.
    """
    zs = np.linspace(z - tol, z + tol, search_steps)
    best = (None, None, None)  # length, loop, z
    for zz in zs:
        sec = mesh.section(plane_origin=[0, 0, zz], plane_normal=up)
        if sec is None:
            continue
        # discretize and try to form polygons
        try:
            path = sec.to_planar()  # returns a Path2D
        except Exception:
            # fallback
            path = sec
        # collect closed loops
        loops = []
        if hasattr(path, "polygons_full"):
            polys = list(path.polygons_full)
            loops = [np.asarray(poly.exterior.coords) for poly in polys if poly.is_valid]
        elif hasattr(path, "discrete"):
            # a set of polylines; we try to stitch but this is less robust
            dis = path.discrete
            if isinstance(dis, list):
                for arr in dis:
                    if len(arr) > 2 and np.allclose(arr[0], arr[-1], atol=1e-5):
                        loops.append(np.asarray(arr))
        # compute lengths
        if not loops:
            continue
        lengths = [np.linalg.norm(np.diff(L, axis=0), axis=1).sum() for L in loops]
        idx = int(np.argmax(lengths))
        Lmax, loop = lengths[idx], loops[idx]
        if best[0] is None or Lmax > best[0]:
            best = (Lmax, loop, float(zz))
    return best

def estimate_landmark_heights_from_joints(J, names=None):
    """
    Derive Z-heights for chest, waist, hip, and mid-thigh from SMPL joints.
    If names are unavailable, fall back to indices commonly used in SMPL (24 joints):
        0 pelvis, 1 L_hip, 2 R_hip, 3 spine1, 6 spine2, 9 spine3/neck,
        12 L_shoulder, 16 R_shoulder, 4 L_knee, 5 R_knee.
    This works as a decent heuristic on most fitted bodies.
    """
    Z = J[:, 2]
    def z_of(label_list, default_idx):
        if names is None:
            return float(J[default_idx, 2])
        # search by substrings
        idxs = []
        for i, n in enumerate(names):
            if n is None:
                continue
            low = str(n).lower()
            if any(s in low for s in label_list):
                idxs.append(i)
        if idxs:
            return float(Z[idxs].mean())
        return float(J[default_idx, 2])

    z_pelvis   = z_of(["pelvis", "root", "hip_c", "spine0"], 0)
    z_spine1   = z_of(["spine1", "spine_1", "lower_back"], 3)
    z_spine2   = z_of(["spine2", "spine_2", "mid_back"], 6)
    z_spine3   = z_of(["spine3", "chest", "upper_chest", "neck_base"], 9)
    z_lhip     = z_of(["l_hip", "left_hip"], 1)
    z_rhip     = z_of(["r_hip", "right_hip"], 2)
    z_lknee    = z_of(["l_knee", "left_knee"], 4)
    z_rknee    = z_of(["r_knee", "right_knee"], 5)

    z_hip_band   = np.mean([z_lhip, z_rhip])                 # around greater trochanter
    z_waist_band = 0.5 * (z_pelvis + z_spine2)               # between pelvis & mid-spine
    z_chest_band = 0.5 * (z_spine2 + z_spine3)               # between mid & upper chest
    z_thigh_mid  = 0.5 * (np.mean([z_lhip, z_rhip]) + np.mean([z_lknee, z_rknee]))

    return {
        "chest": float(z_chest_band),
        "waist": float(z_waist_band),
        "hip":   float(z_hip_band),
        "thigh": float(z_thigh_mid)
    }

def measure_bands(clothed_mesh, body_mesh, band_zs_cm, search_tol_cm=2.0):
    """
    Slice clothed_mesh (and optionally body_mesh) at provided Z-heights (in same units
    as the mesh; assumed to be centimeters if your meshes are already scaled).
    Returns a dict with garment and (optional) body circumference + ease per band.
    """
    results = {}
    up = np.array([0, 0, 1.0], dtype=float)

    for band, z in band_zs_cm.items():
        g_len, _, g_z = plane_section_length(clothed_mesh, z, tol=search_tol_cm, up=up)
        if g_len is None:
            results[band] = {"garment": None, "body": None, "ease": None, "z_used": None}
            continue
        if body_mesh is not None:
            b_len, _, _ = plane_section_length(body_mesh, z, tol=search_tol_cm, up=up)
        else:
            b_len = None
        ease = (g_len - b_len) if (g_len is not None and b_len is not None) else None
        results[band] = {"garment": float(g_len), "body": (None if b_len is None else float(b_len)),
                         "ease": (None if ease is None else float(ease)),
                         "z_used": float(g_z)}
    return results


# ---------- main ----------
def main():
    assert os.path.exists(CLOTHED_OBJ), f"Missing CLOTHED_OBJ: {CLOTHED_OBJ}"
    clothed = load_mesh(CLOTHED_OBJ)
    clothed, R_c = z_up_transform(clothed)

    body = None
    if BODY_OBJ:
        assert os.path.exists(BODY_OBJ), f"Missing BODY_OBJ: {BODY_OBJ}"
        body = load_mesh(BODY_OBJ)
        body.apply_transform(R_c)  # keep same frame as clothed (rough)

    # Build a measurer to regress joints and set landmark heights.
    # We feed the *body* if provided (best), else the clothed mesh (still OK for heights).
    meas_mesh = body if body is not None else clothed

    V = np.asarray(meas_mesh.vertices).astype(np.float32)
    measurer = MeasureBody(MODEL_TYPE)  # "smpl" or "smplx"
    # choose device
    try:
        device = measurer.joint_regressor.device
    except Exception:
        try:
            device = next(measurer.model.parameters()).device
        except Exception:
            device = "cpu"

    V_t = torch.from_numpy(V).to(device=device, dtype=torch.float32)
    measurer.from_verts(verts=V_t)

    # Grab joints
    J, names = _maybe_get_joints(measurer)
    if J is None:
        raise RuntimeError("Could not obtain joints from MeasureBody; check topology/inputs.")

    bands_z = estimate_landmark_heights_from_joints(J, names=names)

    # Units:
    # If your meshes are in meters, convert z to meters and lengths to meters, then *100 for cm.
    # Many SMPL pipelines keep units in meters; if so, multiply all reported lengths by 100.
    # Here we detect scale by bounding-box height; if < 5.0 → assume meters → convert to cm.
    bbox_h = clothed.extents[2]
    in_meters = bbox_h < 5.0
    scale = 100.0 if in_meters else 1.0  # factor to convert lengths → cm

    # Convert Zs to mesh units
    for k in bands_z:
        bands_z[k] = bands_z[k] * (1.0 if not in_meters else 1.0)  # J came from same mesh; no change

    # Measure
    res = measure_bands(clothed, body, bands_z, search_tol_cm=(2.0 / scale if in_meters else 2.0))

    # Scale loop lengths to cm if needed
    for band, d in res.items():
        if d["garment"] is not None:
            d["garment"] *= scale
        if d["body"] is not None:
            d["body"] *= scale
        if d["ease"] is not None:
            d["ease"] *= scale
        if d["z_used"] is not None:
            d["z_used"] = d["z_used"] * (100.0 if in_meters else 1.0)

    # Pretty print
    print("\n== Garment measurements ==")
    print("(All lengths in cm; z_used = slice height in cm above origin)")
    order = ["chest", "waist", "hip", "thigh"]
    for band in order:
        d = res.get(band, {})
        g = d.get("garment"); b = d.get("body"); e = d.get("ease"); z = d.get("z_used")
        def fmt(x): return f"{x:6.2f}" if x is not None else "   n/a"
        print(f"{band:>5}: garment={fmt(g)}  body={fmt(b)}  ease={fmt(e)}  (z_used={fmt(z)})")

    # Optional: export the slice loops as debug geometry
    if SAVE_DEBUG:
        # For brevity we only drop the original mesh; you can add polylines as Path3D
        clothed.export(SAVE_DEBUG)
        print(f"Saved debug scene → {SAVE_DEBUG}")


if __name__ == "__main__":
    main()
