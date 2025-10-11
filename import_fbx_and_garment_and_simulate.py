import bpy
import os
import mathutils
from math import radians

FBX_BODY_PATH = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_no_smpl.fbx"
AUTOMATIC_BONE_ORIENTATION = False
OBJ_GARMENT_PATH  = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_garment_modified_rot_scaled.obj"


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
    return obj

def import_and_rotate_garment():
    if not os.path.exists(OBJ_GARMENT_PATH):
        raise FileNotFoundError(OBJ_GARMENT_PATH)
    print(f"📂 Importing OBJ garment from {OBJ_GARMENT_PATH} ...")
    ensure_object_mode()

    # --- Import OBJ (garment) with axis settings ---
    if not hasattr(bpy.ops.wm, "obj_import"):
        raise RuntimeError("This Blender build lacks 'bpy.ops.wm.obj_import' (OBJ importer).")
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(
        filepath=OBJ_GARMENT_PATH,
        validate_meshes=True
        # split_mode='OFF',  # uncomment if you want one object (when supported by exporter)
    )
    after = set(bpy.data.objects)
    new_obj = [o for o in (after - before) if o.type == 'MESH']
    if not new_obj:
        raise RuntimeError("OBJ import: no MESH objects found for garment.")

    # If multiple meshes came in, join into one garment
    if len(new_obj) > 1:
        deselect_all()
        for o in new_obj:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new_obj[0]
        bpy.ops.object.join()
        garment = new_obj[0]
    else:
        garment = new_obj[0]

    rot_obj(garment, 'X', 90)
    return garment


def make_rigid(obj):
    if not obj:
        print("Error: No object provided to make_rigid!")
        return

    # Ensure obj is in the current view layer
    view_layer = bpy.context.view_layer
    if obj.name not in [o.name for o in view_layer.objects]:
        print(f"Error: {obj.name} not in current view layer!")
        return

    # Deselect all first
    deselect_all()

    # Set selection and active (ensures context for mode_set)
    obj.select_set(True)
    view_layer.objects.active = obj

    # Now safely switch to Object Mode
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError as e:
        print(f"Mode switch failed: {e}. Attempting fallback...")
        if bpy.context.mode != 'OBJECT':
            print("Warning: Could not switch to Object Mode. Ensure object is valid.")
            return

    # Safely switch Properties editor to Physics context
    physics_context_set = False
    for area in bpy.context.screen.areas:
        if area.type == 'PROPERTIES':
            area.spaces.active.context = 'PHYSICS'
            physics_context_set = True
            break

    if not physics_context_set:
        print("Warning: No Properties editor found. Physics tab won't switch, but modifier still adds.")
        # Optional: Create a Properties area if needed (splits the active screen)
        # bpy.ops.screen.area_split(direction='RIGHT', factor=0.3)
        # But avoid auto-splitting unless desired

    # Add the Collision modifier
    bpy.ops.object.modifier_add(type='COLLISION')

    # Optional: Configure the modifier
    mod = obj.modifiers[-1]
    #    mod.name = "CollisionRigid"
    #    mod.settings.thickness_outer = 0.04  # Adjust as needed
    #
    print(f"Collision modifier added to {obj.name}. Properties tab switched to Physics.")

# ----------------------------- MAIN -----------------------------
def main():
    clear_scene()
    body = import_and_rotate_body()
    make_rigid(body)
    garment = import_and_rotate_garment()

    print("🎉 Complete! play the sim to test.")


if __name__ == "__main__":
    main()
