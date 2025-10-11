import bpy
import os
import mathutils
from math import radians

FBX_BODY_PATH = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_no_smpl.fbx"
AUTOMATIC_BONE_ORIENTATION = False


def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


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


def rot_obj(obj, axis_name='X', degree=90):
    # Store original location (to preserve it)
    loc = obj.location.copy()

    # Create rotation matrix for 90 degrees around global X
    rot_matrix = mathutils.Matrix.Rotation(radians(90), 4, 'X')  # 4x4 for full transform

    # Apply to object's matrix (combines with existing rotation)
    obj.matrix_world = obj.matrix_world @ rot_matrix

    # Reset location to original (cancels any induced translation)
    obj.location = loc


def import_and_rotate_body():
    if not os.path.exists(FBX_BODY_PATH):
        raise FileNotFoundError(FBX_BODY_PATH)
    print(f"📂 Importing FBX body from {FBX_BODY_PATH} ...")
    ensure_object_mode()
    # --- Import FBX ---
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=FBX_BODY_PATH, automatic_bone_orientation=AUTOMATIC_BONE_ORIENTATION)
    after = set(bpy.data.objects)
    new_fbx = list(after - before)
    if not new_fbx:
        raise RuntimeError("FBX import produced no objects.")

    obj = new_fbx[0]
    rot_obj(obj, 'X', 90)
    print(f"✅ FBX body imported and rotated.\n")
    return obj


# ----------------------------- MAIN -----------------------------
def main():
    clear_scene()
    body = import_and_rotate_body()


    print("🎉 Complete! play the sim to test.")


if __name__ == "__main__":
    main()
