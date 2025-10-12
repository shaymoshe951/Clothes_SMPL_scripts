## Not working

__filepath__ = r"C:\Users\Lab\Downloads\clothes_images\mesh_w_clothes_from_opensite\model_xs1_mesh.obj"
__smpl_fit_filepath__ = r"C:\Users\Lab\Downloads\clothes_images\model_yali2_mesh_0_0.obj"
# clothes_measurements_body_guided_xy_final.py
import os
import numpy as np
import trimesh
import torch

# Robust polygon ops (recommended)
try:
    from shapely.geometry import Polygon, Point, LineString, MultiPoint
    from shapely.ops import unary_union
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS

# ================= USER SETTINGS =================
CLOTHED_OBJ = __filepath__
BODY_OBJ    = __smpl_fit_filepath__    # fitted SMPL(/X) .obj if available
MODEL_TYPE  = "smpl"                    # "smpl" or "smplx"

# Scaling / calibration
KNOWN_HEIGHT_CM      = 172.0           # set real height in cm, or None
CALIB_SOURCE         = "body"          # "body" (preferred) or "bbox"
KNOWN_SCALE          = None            # overrides height calibration if set (unitless)
FORCE_IN_METERS      = True            # meshes look like meters

# Fallback band placement (only if joints missing)
RATIOS_HINT          = {"chest":0.62, "waist":0.53, "hip":0.50, "thigh":0.37}

# Slicing tolerances (forgiving + robust)
TOL_UNITS_M          = 0.06            # ±6 cm plane wiggle (in meters)
SLAB_MIN_HALF_THICK  = 0.03            # ≥3 cm slab half-thickness
DEBUG                = True
SAVE_DEBUG_SCENE     = None
# ==================================================

SMPL_VERTS  = 6890
SMPLX_VERTS = 10475

# ---------- utils ----------
def to_numpy(v):
    try: return v.detach().cpu().numpy()
    except Exception: return np.asarray(v)

def load_mesh(path):
    if path is None: return None
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        parts = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts: raise ValueError("Scene contains no mesh geometry.")
        m = trimesh.util.concatenate(parts)
    m.remove_unreferenced_vertices()
    return m

def repair_mesh_inplace(mesh: trimesh.Trimesh):
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    try: trimesh.repair.fix_normals(mesh)
    except Exception: pass
    try: trimesh.repair.fill_holes(mesh)
    except Exception: pass
    try: mesh.merge_vertices()
    except Exception: pass

def z_up_transform(mesh: trimesh.Trimesh):
    ext = mesh.extents
    if DEBUG: print(f"[dbg] extents (x,y,z) = {ext}")
    if ext[2] >= ext[0] and ext[2] >= ext[1]:
        return mesh, np.eye(4)
    axes = np.argsort(ext); major = axes[-1]
    if major == 0:
        R = trimesh.transformations.rotation_matrix(np.deg2rad(90), [0,1,0])
    else:
        R = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [1,0,0])
    m2 = mesh.copy(); m2.apply_transform(R)
    if DEBUG: print(f"[dbg] applied rotation to make Z-up; new extents = {m2.extents}")
    return m2, R

def apply_uniform_scale(mesh: trimesh.Trimesh, s: float):
    if mesh is None or s is None or abs(s - 1.0) < 1e-12: return
    S = np.eye(4); S[:3,:3] *= float(s)
    mesh.apply_transform(S)

def is_smpl_topology(nv, model_type):
    target = SMPL_VERTS if model_type.lower()=="smpl" else SMPLX_VERTS
    return int(nv) == int(target)

# ---------- joints & band windows ----------
def _maybe_get_joints(measurer: MeasureBody):
    names = None
    for attr in ["joints", "J_tr", "J", "_joints"]:
        if hasattr(measurer, attr):
            J = to_numpy(getattr(measurer, attr))
            if J is not None:
                if J.ndim == 3 and J.shape[-1] == 3: J = J[0]
                if J.ndim == 2 and J.shape[1] == 3:
                    for n_attr in ["joint_names","J_names","joints_names","_joint_names"]:
                        if hasattr(measurer, n_attr):
                            try: names = getattr(measurer, n_attr)
                            except Exception: names = None
                    return J, names
    return None, None

def band_windows_from_joints(J):
    Z = J[:,2]
    pelvis = Z[0]
    spine2 = Z[6]
    upper  = Z[9]
    lhip, rhip = Z[1], Z[2]
    lknee, rknee = Z[4], Z[5]
    hip   = float(np.mean([lhip, rhip]))
    knee  = float(np.mean([lknee, rknee]))
    torso_span = float(upper - pelvis)

    zmin, zmax = float(np.min(Z)), float(np.max(Z))
    H = float(zmax - zmin)

    chest_lo, chest_hi = float(spine2), float(upper)
    waist_lo, waist_hi = float(pelvis - 0.03*H), float(spine2 + 0.03*H)
    hip_lo,   hip_hi   = float(hip - 0.05*H),    float(hip + 0.03*H)
    thigh_lo = float(knee + 0.04*torso_span)
    thigh_hi = float(hip  - 0.04*torso_span)
    if thigh_lo > thigh_hi: thigh_lo, thigh_hi = thigh_hi, thigh_lo

    return {
        "chest": (chest_lo, chest_hi),
        "waist": (waist_lo, waist_hi),
        "hip":   (hip_lo,   hip_hi),
        "thigh": (thigh_lo, thigh_hi),
    }

def fallback_windows_from_bbox(mesh, hints):
    z0, z1 = mesh.bounds[0][2], mesh.bounds[1][2]
    H = float(z1 - z0)
    pad = 0.06*H
    windows = {}
    for k,r in hints.items():
        zc = z0 + r*H
        windows[k] = (zc - pad, zc + pad)
    return windows

# ---------- cross-section in shared XY ----------
def mesh_plane_paths(mesh: trimesh.Trimesh, z):
    p3d = None
    try:
        p3d = trimesh.intersections.mesh_plane(mesh, plane_normal=[0,0,1.0], plane_origin=[0,0,float(z)])
    except Exception:
        pass
    lines = []
    if p3d is not None and hasattr(p3d, "discrete"):
        for seg in p3d.discrete:
            if len(seg) >= 2: lines.append(np.asarray(seg, dtype=float))
    if lines: return lines

    # thin slab fallback
    try:
        planes = np.linspace(float(z - 1e-5), float(z + 1e-5), 3)
        paths = mesh.section_multiplane(plane_origin=[0,0,0], plane_normal=[0,0,1.0], heights=planes)
        geoms = [p for p in paths if p is not None]
        if geoms:
            merged = geoms[0]
            for p in geoms[1:]:
                merged = merged + p
            if hasattr(merged, "discrete"):
                for seg in merged.discrete:
                    if len(seg) >= 2: lines.append(np.asarray(seg, dtype=float))
    except Exception:
        pass
    return lines

def polyline3d_to_closed_2d_xy(lines3d):
    loops = []
    for L in lines3d:
        xy = L[:, :2]
        if len(xy) >= 3 and np.linalg.norm(xy[0] - xy[-1]) < 1e-5:
            loops.append(xy)
    if loops: return loops

    # gap-healing
    if HAVE_SHAPELY and lines3d:
        try:
            segs = [LineString(seg[:, :2]) for seg in lines3d if len(seg) >= 2]
            ml = unary_union(segs)
            buff = ml.buffer(2e-3)  # 2 mm
            geoms = []
            if buff.geom_type == "Polygon": geoms = [buff]
            elif buff.geom_type == "MultiPolygon": geoms = list(buff.geoms)
            for poly in geoms:
                loops.append(np.asarray(poly.exterior.coords))
        except Exception:
            pass
    return loops

def robust_loops_xy_at_z(mesh, z, tol_units, in_meters):
    zs = np.linspace(z - tol_units, z + tol_units, 9)
    for zz in zs:
        lines = mesh_plane_paths(mesh, zz)
        loops = polyline3d_to_closed_2d_xy(lines)
        if loops: return loops, float(zz)
    # thicker slab
    half = max(tol_units, (SLAB_MIN_HALF_THICK if in_meters else 1.0))
    for factor in [1.0, 2.0, 3.0]:
        z_low, z_high = z - half*factor, z + half*factor
        planes = np.linspace(float(z_low), float(z_high), 11)
        try:
            paths = mesh.section_multiplane(plane_origin=[0,0,0], plane_normal=[0,0,1.0], heights=planes)
            geoms = [p for p in paths if p is not None]
            if geoms:
                merged = geoms[0]
                for p in geoms[1:]:
                    merged = merged + p
                if hasattr(merged, "discrete"):
                    lines = [np.asarray(seg, dtype=float) for seg in merged.discrete if len(seg) >= 2]
                    loops = polyline3d_to_closed_2d_xy(lines)
                    if loops: return loops, float(z)
        except Exception:
            pass
    return [], None

# ---------- LAST-RESORT: convex hull fallback ----------
def _convex_hull_xy(points):
    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) < 3:
        return np.empty((0,2))
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return np.empty((0,2))
    return np.vstack([hull, hull[0]])

def buffered_pointloop_xy(mesh, z, slab_half, in_meters, body_centroid=None, max_radius_cm=None):
    V = np.asarray(mesh.vertices)
    z0, z1 = z - slab_half, z + slab_half
    mask = (V[:,2] >= z0) & (V[:,2] <= z1)
    P = V[mask, :2]
    if P.shape[0] < 50:
        return []
    # optional radius cap around body centroid (to avoid far flares)
    if body_centroid is not None and max_radius_cm is not None:
        unit_per_cm = (0.01 if in_meters else 1.0)
        r_units = max_radius_cm * unit_per_cm
        d = np.linalg.norm(P - body_centroid[None,:], axis=1)
        sel = d <= r_units
        if np.count_nonzero(sel) > 30:
            P = P[sel]
    if HAVE_SHAPELY:
        try:
            mp = MultiPoint([tuple(p) for p in P])
            r = 0.010 if in_meters else 1.0  # ~1 cm
            poly = mp.buffer(r).buffer(0)
            if poly.is_empty:
                return []
            if poly.geom_type == "Polygon":
                return [np.asarray(poly.exterior.coords)]
            elif poly.geom_type == "MultiPolygon":
                areas = [g.area for g in poly.geoms]
                gi = int(np.argmax(areas))
                return [np.asarray(poly.geoms[gi].exterior.coords)]
        except Exception:
            pass
    H = _convex_hull_xy(P)
    if H.shape[0] == 0: return []
    return [H]

# ---------- calibration ----------
def get_bbox_height_units(mesh: trimesh.Trimesh) -> float:
    b = mesh.bounds
    return float(b[1][2] - b[0][2])

def get_body_height_cm_with_measurer(body_mesh: trimesh.Trimesh, model_type: str) -> float:
    measurer = MeasureBody(model_type)
    V = np.asarray(body_mesh.vertices, dtype=np.float32)
    try: device = measurer.joint_regressor.device
    except Exception:
        try: device = next(measurer.model.parameters()).device
        except Exception: device = "cpu"
    measurer.from_verts(torch.from_numpy(V).to(device=device))
    measurer.measure(measurer.all_possible_measurements)
    measurer.label_measurements(STANDARD_LABELS)
    labeled = getattr(measurer, "labeled_measurements", {}) or {}
    h_cm = labeled.get("P", None)
    return float(h_cm) if h_cm is not None else None

def compute_scale_factor(clothed_mesh, body_mesh, in_meters, known_height_cm, calib_source, known_scale, model_type):
    if known_scale is not None:
        if DEBUG: print(f"[dbg] Using KNOWN_SCALE={known_scale}"); return float(known_scale)
    if known_height_cm is None:
        if DEBUG: print("[dbg] No known height provided; s=1."); return 1.0
    cm_per_unit = (100.0 if in_meters else 1.0)
    desired_units = float(known_height_cm) / cm_per_unit
    if calib_source == "body" and body_mesh is not None and is_smpl_topology(len(body_mesh.vertices), model_type):
        h_cm = get_body_height_cm_with_measurer(body_mesh, model_type)
        if h_cm is not None:
            current_units = float(h_cm) / cm_per_unit
            if DEBUG: print(f"[dbg] Current BODY height ≈ {h_cm:.2f} cm; desired {known_height_cm:.2f} cm")
        else:
            current_units = get_bbox_height_units(body_mesh)
            if DEBUG: print(f"[dbg] BODY bbox height ≈ {current_units*cm_per_unit:.2f} cm")
    else:
        current_units = get_bbox_height_units(clothed_mesh)
        if DEBUG: print(f"[dbg] CLOTHED bbox height ≈ {current_units*cm_per_unit:.2f} cm (may include shoes/hair)")
    if current_units <= 0: return 1.0
    s = desired_units / current_units
    if DEBUG: print(f"[dbg] scale factor s = desired({desired_units:.6f}) / current({current_units:.6f}) = {s:.6f}")
    return float(s)

# ---------- gating params ----------
# Perimeter is noisy; only enforce a tiny lower bound (≥ body)
PERIM_MIN_RATIO = 1.0000

# Tight, band-specific clearance & centroid caps (cm)
CLEARANCE_MIN_CM = { "chest": 0.2, "waist": 0.2, "hip": 0.5, "thigh": 0.5 }
CLEARANCE_MAX_CM = { "chest": 4.0, "waist": 4.0, "hip": 6.0, "thigh": 8.0 }
CENTROID_MAX_CM  = { "chest": 6.0, "waist": 6.0, "hip": 6.0, "thigh": 8.0 }

# ---------- metrics ----------
def loop_length_xy(L2d):
    return float(np.linalg.norm(np.diff(L2d, axis=0), axis=1).sum())

def centroid_xy(L2d):
    if len(L2d) >= 2 and np.allclose(L2d[0], L2d[-1], atol=1e-9):
        return np.mean(L2d[:-1], axis=0)
    return np.mean(L2d, axis=0)

def contains_point_xy(loop2d, point2d):
    if HAVE_SHAPELY:
        try: return Polygon(loop2d).contains(Point(point2d))
        except Exception: return False
    # winding fallback
    c = 0; x, y = point2d; pts = loop2d
    for i in range(len(pts)-1):
        x1,y1 = pts[i]; x2,y2 = pts[i+1]
        if ((y1 <= y < y2) or (y2 <= y < y1)):
            xin = x1 + (y - y1)*(x2 - x1)/((y2 - y1) + 1e-12)
            if xin > x: c ^= 1
    return bool(c)

def clearance_metrics(loop_g, loop_b, in_meters):
    unit_to_cm = 100.0 if in_meters else 1.0
    cb = centroid_xy(loop_b); cg = centroid_xy(loop_g)
    centroid_shift_cm = float(np.linalg.norm(cg - cb)) * unit_to_cm
    if HAVE_SHAPELY:
        try:
            Pg = Polygon(loop_g); Pb = Polygon(loop_b)
            if not Pg.is_valid or not Pb.is_valid: raise ValueError
            A = max(Pg.area - Pb.area, 0.0)
            per_b = max(Pb.length, 1e-9)
            avg_clear_units = A / per_b
            return float(avg_clear_units * unit_to_cm), float(centroid_shift_cm)
        except Exception:
            pass
    # fallback
    if HAVE_SHAPELY:
        Pb = Polygon(loop_b)
        dists = [Pb.exterior.distance(Point(p)) for p in loop_g[::max(1, len(loop_g)//200)]]
        avg_clear_units = float(np.mean(dists))
    else:
        dists = [np.linalg.norm(p - cb) for p in loop_g[::max(1, len(loop_g)//200)]]
        dists_b = [np.linalg.norm(p - cb) for p in loop_b[::max(1, len(loop_b)//200)]]
        avg_clear_units = max(float(np.mean(dists) - np.mean(dists_b)), 0.0)
    return float(avg_clear_units * unit_to_cm), float(centroid_shift_cm)

# ---------- selection helpers ----------
def robust_body_loop_at_z(body_mesh, z, tol_units, in_meters):
    loops_b, _ = robust_loops_xy_at_z(body_mesh, z, tol_units, in_meters)
    if loops_b:
        blens = [loop_length_xy(L) for L in loops_b]
        bi = int(np.argmax(blens))
        return loops_b[bi]
    return []

def evaluate_candidates(cands, z_used, band, body_c, body_len, body_L, in_meters):
    """
    Two-pass gating:
      Pass A: require per >= body (lo); centroid & clearance within caps.
      Pass B: if A empty, drop perimeter bound and pick by minimal clearance (still centroid & clearance caps).
    """
    unit_to_cm = 100.0 if in_meters else 1.0
    lo = body_len * PERIM_MIN_RATIO

    cleared_A, cleared_B = [], []
    n_total = len(cands)
    n_contain = n_per = n_center = n_clear = 0

    for L in cands:
        if not contains_point_xy(L, body_c):
            continue
        n_contain += 1

        per = loop_length_xy(L)
        avg_clear_cm, centroid_shift_cm = clearance_metrics(L, body_L, in_meters)
        centroid_ok = (centroid_shift_cm <= CENTROID_MAX_CM[band])
        clear_ok    = (CLEARANCE_MIN_CM[band] <= avg_clear_cm <= CLEARANCE_MAX_CM[band])

        # Pass B (no perimeter bound)
        if centroid_ok and clear_ok:
            cleared_B.append((avg_clear_cm, abs(per - lo), per, L, z_used))

        # Pass A (enforce per >= lo)
        if per + 1e-9 >= lo and centroid_ok and clear_ok:
            n_per += 1; n_center += 1; n_clear += 1
            cleared_A.append((avg_clear_cm, abs(per - lo), per, L, z_used))

    if DEBUG:
        hi = lo * 10.0  # not used; printed for context
        print(f"[dbg] {band}: z={z_used:.4f} candidates:"
              f" total={n_total}, contain={n_contain},"
              f" per_ok={n_per}, centroid_ok={n_center}, clear_ok={n_clear},"
              f" lo={lo*unit_to_cm:.2f}cm hi={hi*unit_to_cm:.2f}cm")

    pick_from = cleared_A if cleared_A else cleared_B
    if not pick_from:
        return None
    pick_from.sort(key=lambda t: (t[0], t[1]))  # min clearance, then |per-lo|
    return pick_from[0]  # (clear, |per-lo|, per, L, z)

def find_garment_loop_in_window(band, z_lo, z_hi, clothed_mesh, body_mesh, in_meters):
    tol = TOL_UNITS_M if in_meters else 2.0
    zs = np.linspace(z_lo, z_hi, 13)

    # body reference at mid-height
    body_L = robust_body_loop_at_z(body_mesh, float(0.5*(z_lo + z_hi)), tol, in_meters)
    if not len(body_L):
        return None
    body_len = loop_length_xy(body_L)
    body_c   = centroid_xy(body_L)
    lo = body_len * PERIM_MIN_RATIO

    def try_candidates(cands, z_used):
        return evaluate_candidates(cands, z_used, band, body_c, body_len, body_L, in_meters)

    # pass 1: geometric loops
    best = None
    for z in zs:
        loops_g, zg = robust_loops_xy_at_z(clothed_mesh, z, tol, in_meters)
        if not loops_g: continue
        pick = try_candidates(loops_g, zg)
        if pick is not None and (best is None or pick < best):
            best = pick

    # pass 2: point-cloud buffered ring (only if needed)
    if best is None:
        slab_half = max(tol, (SLAB_MIN_HALF_THICK if in_meters else 1.0))
        mean_body_radius_units = (body_len / (2.0 * np.pi))
        max_radius_cm = (mean_body_radius_units * (100.0 if in_meters else 1.0)) * 1.10  # +10%
        for z in zs:
            loops_buf = buffered_pointloop_xy(
                clothed_mesh, z, slab_half, in_meters,
                body_centroid=body_c, max_radius_cm=max_radius_cm
            )
            if not loops_buf: continue
            pick = try_candidates(loops_buf, z)
            if pick is not None and (best is None or pick < best):
                best = pick

    if best is None:
        return None
    clear_cm, _, per, _, z_used = best  # 5-tuple
    return float(per), float(body_len), float(z_used), float(clear_cm)

def measure_bands(clothed_mesh, body_mesh, band_windows, in_meters):
    out = {}
    for band, (z_lo, z_hi) in band_windows.items():
        picked = find_garment_loop_in_window(band, z_lo, z_hi, clothed_mesh, body_mesh, in_meters)
        if picked is None:
            if DEBUG: print(f"[dbg] {band}: no valid loop in window [{z_lo:.4f},{z_hi:.4f}]")
            out[band] = {"garment":None,"body":None,"ease":None,"z_used":None}
            continue
        g_len, b_len, z_used, clr = picked
        ease = g_len - b_len
        out[band] = {"garment":g_len, "body":b_len, "ease":ease, "z_used":z_used}
        if DEBUG:
            cm = 100.0 if in_meters else 1.0
            print(f"[dbg] {band}: z_used={z_used:.4f}, body_per={b_len*cm:.2f}cm, "
                  f"gar_per={g_len*cm:.2f}cm, ease={ease*cm:.2f}cm, clear≈{clr:.2f}cm")
    return out

# ---------- main ----------
def main():
    clothed = load_mesh(CLOTHED_OBJ); assert clothed is not None, "Missing CLOTHED_OBJ"
    repair_mesh_inplace(clothed); clothed, R = z_up_transform(clothed)

    body = load_mesh(BODY_OBJ) if BODY_OBJ else None
    assert body is not None, "This method requires BODY_OBJ (fitted SMPL/SMPL-X)."
    repair_mesh_inplace(body); body.apply_transform(R)

    # Units & calibration
    in_meters = bool(FORCE_IN_METERS)
    if DEBUG: print(f"[dbg] bbox_h_before={clothed.extents[2]:.4f}, in_meters={in_meters}")
    s = compute_scale_factor(clothed, body, in_meters, KNOWN_HEIGHT_CM, CALIB_SOURCE, KNOWN_SCALE, MODEL_TYPE)
    apply_uniform_scale(clothed, s); apply_uniform_scale(body, s)
    if DEBUG:
        cm_per_unit = (100.0 if in_meters else 1.0)
        print(f"[dbg] bbox_h_after={clothed.extents[2]:.4f} units (~{clothed.extents[2]*cm_per_unit:.2f} cm)")
        print(f"[dbg] HAVE_SHAPELY = {HAVE_SHAPELY}")

    # Joints → windows (fallback to bbox if extraction fails)
    V = np.asarray(body.vertices, dtype=np.float32)
    measurer = MeasureBody(MODEL_TYPE)
    try: device = measurer.joint_regressor.device
    except Exception:
        try: device = next(measurer.model.parameters()).device
        except Exception: device = "cpu"
    measurer.from_verts(torch.from_numpy(V).to(device=device))
    J, _ = _maybe_get_joints(measurer)

    if J is not None:
        band_windows = band_windows_from_joints(J)
    else:
        if DEBUG: print("[dbg] joints unavailable; using bbox ratio windows.")
        band_windows = fallback_windows_from_bbox(clothed, RATIOS_HINT)

    # Measure
    res = measure_bands(clothed, body, band_windows, in_meters)

    # Report (cm)
    to_cm = 100.0 if in_meters else 1.0
    print("\n== Garment measurements (cm) ==")
    print("(z_used = slice height above origin, in cm)")
    order = ["chest","waist","hip","thigh"]
    f = lambda x: f"{x*to_cm:7.2f}" if x is not None else "   n/a "
    for band in order:
        d = res.get(band, {})
        print(f"{band:>5}: garment={f(d.get('garment'))}  body={f(d.get('body'))}  "
              f"ease={f(d.get('ease'))}  (z_used={f(d.get('z_used'))})")

    if SAVE_DEBUG_SCENE:
        clothed.export(SAVE_DEBUG_SCENE)
        print(f"[info] Saved debug scene → {SAVE_DEBUG_SCENE}")

if __name__ == "__main__":
    main()
