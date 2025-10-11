############### IMPORTANT FILE ###############

# __my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\liran_mesh_0_0.obj"
# __my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\model_xs1_mesh_0_1 (1).obj"
# __my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\model_l1_mesh_0_0.obj"
# __my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\model_l1_mesh_fused.obj"
# __my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\liran_mesh_fused.obj"
__my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\liran_focal120_mesh_0_0.obj"
# __my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\mesh_w_clothes_from_opensite\model_xs1_mesh.obj"
# __my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\mesh_w_clothes_from_opensite\liran_mesh.obj"
# __my_file__ =  r"C:\Users\Lab\Downloads\clothes_images\model_yali2_mesh_0_3.obj"


# smpl_measure_and_viz.py
# Minimal example:
# - Loads an SMPL(/X)-topology OBJ with trimesh
# - Measures key anthropometrics via SMPL-Anthropometry
# - Visualizes the mesh (and joints if available) + overlays the numbers

import numpy as np
import trimesh
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS

# ---- user settings (edit these) ----
OBJ_PATH   = __my_file__   # SMPL: 6890 verts; SMPL-X: 10475 verts
MODEL_TYPE = "smpl"                         # "smpl" or "smplx"
GENDER     = "FEMALE"                      # NEUTRAL / MALE / FEMALE
NORMALIZE_HEIGHT = 160                     # e.g., 175 (cm) or None to keep original scale
SAVE_IMAGE = None                           # e.g., r"D:\out.png" or None to just show
# ------------------------------------

LABEL_TO_NAME = {
    "P": "height",
    "E": "waist circumference",
    "F": "hip circumference",
    "D": "chest circumference",
    "O": "shoulder breadth",
    "K": "inside leg height",
}
KEYS = ["P", "E", "F", "D", "O", "K"]

def maybe_get_joints(measurer):
    """Try to pull joints from the measurer (defensive to API changes)."""
    for attr in ["joints", "J_tr", "J", "_joints"]:
        if hasattr(measurer, attr):
            J = getattr(measurer, attr)
            try:
                J = J.detach().cpu().numpy()
            except Exception:
                J = np.asarray(J)
            if J is not None and J.ndim == 2 and J.shape[1] == 3:
                return J
            if J is not None and J.ndim == 3 and J.shape[-1] == 3:
                return np.asarray(J)[0]
    return None

def set_equal_3d(ax, V):
    """Equal aspect ratio for 3D axes based on vertices V (N,3)."""
    mins = V.min(axis=0); maxs = V.max(axis=0)
    centers = (mins + maxs) / 2.0
    range_ = (maxs - mins).max() / 2.0
    ax.set_xlim(centers[0]-range_, centers[0]+range_)
    ax.set_ylim(centers[1]-range_, centers[1]+range_)
    ax.set_zlim(centers[2]-range_, centers[2]+range_)

def visualize(V, F, labeled, measurer, suffix="", save_path=None):
    """Simple 3D render + measurements panel."""
    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, projection='3d')

    if F is not None and len(F) > 0:
        tris = V[F]
        coll = Poly3DCollection(tris, linewidths=0.1, alpha=0.85)
        ax.add_collection3d(coll)
        # Light triangulated surface for nicer shading
        ax.plot_trisurf(V[:,0], V[:,1], V[:,2], triangles=F, linewidth=0.0, alpha=0.10)
    else:
        ax.scatter(V[:,0], V[:,1], V[:,2], s=0.5, depthshade=True, alpha=0.8)

    # Joints if available
    J = maybe_get_joints(measurer)
    if J is not None:
        ax.scatter(J[:,0], J[:,1], J[:,2], s=20, marker='o', alpha=1.0)

    set_equal_3d(ax, V)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=15, azim=-70)

    lines = [f"{LABEL_TO_NAME[k]} ({k}): {labeled[k]:.2f} cm"
             for k in KEYS if k in labeled and labeled[k] is not None]
    txt = "\n".join(lines) + (("\n" + suffix) if suffix else "")
    fig.text(0.02, 0.02, txt, fontsize=10, family="monospace")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization → {save_path}")
    else:
        plt.show()

# ---- main (no args, just run) ----
mesh = trimesh.load(OBJ_PATH, process=False)
V_np = np.asarray(mesh.vertices)
F = np.asarray(mesh.faces) if getattr(mesh, "faces", None) is not None else None

print("Vertices:", V_np.shape, "Faces:", (None if F is None else F.shape))

measurer = MeasureBody(MODEL_TYPE)

# Choose device used internally (falls back to CPU)
device = "cpu"
try:
    device = measurer.joint_regressor.device
except Exception:
    try:
        device = next(measurer.model.parameters()).device
    except Exception:
        pass

V_t = torch.from_numpy(V_np).to(device=device, dtype=torch.float32)
measurer.from_verts(verts=V_t)

# Measure & label
measurer.measure(measurer.all_possible_measurements)
measurer.label_measurements(STANDARD_LABELS)

if NORMALIZE_HEIGHT is not None:
    # 1) compute normalized values
    measurer.height_normalize_measurements(float(NORMALIZE_HEIGHT))
    hn = measurer.height_normalized_measurements  # dict: {measurement_name: value}

    # 2) build a labeled dict for normalized values
    name2label = {v: k for k, v in STANDARD_LABELS.items()}  # e.g., {"height": "P", "waist circumference": "E", ...}
    labeled = {name2label.get(name, name): val for name, val in hn.items()}
    suffix = f"(height-normalized to {float(NORMALIZE_HEIGHT):.1f} cm)"
else:
    labeled = measurer.labeled_measurements
    suffix = ""

print("\n== Key measurements (cm) %s ==" % (suffix or ""))
for k in KEYS:
    val = labeled.get(k)
    print(f"{k:>2} {LABEL_TO_NAME[k]:<26}: {val:.2f}" if val is not None else
          f"{k:>2} {LABEL_TO_NAME[k]:<26}: (n/a)")

# Visualize
visualize(V_np, F, labeled, measurer, suffix=suffix, save_path=SAVE_IMAGE)
