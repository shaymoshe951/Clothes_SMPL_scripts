# import bpy

# view_layer = bpy.context.view_layer
# obj = view_layer.objects.active

# print(obj)
##obj = bpy.data.objects["SMPL-female"]

# obj.keys()          # Custom properties (user-defined only)
# obj.bl_rna.properties.keys()  # All RNA properties (defined by Blender schema)

# for prop in obj.bl_rna.properties:
#    if not prop.is_hidden:
#        try:
#            print(f"{prop.identifier}: {getattr(obj, prop.identifier)}")
#        except Exception as e:
#            print(f"{prop.identifier}: <error {e}>")

# bpy.data.shape_keys["Key.002"].key_blocks["Shape000"].value = 3

import bpy
import os
import mathutils
from math import radians

FBX_BODY_PATH = r"C:\Users\Lab\Downloads\template_smpl_female.fbx"
AUTOMATIC_BONE_ORIENTATION = False


def import_fbx():
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
    return obj


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


def select_obj(obj):
    if not obj:
        print("Error: No object provided!")
        return

    # Ensure obj is in the current view layer
    view_layer = bpy.context.view_layer
    if obj.name not in [o.name for o in view_layer.objects]:
        print(f"Error: {obj.name} not in current view layer!")
        return

    # Deselect all first
    deselect_all()


# ----------------------------- MAIN -----------------------------
def main():
    clear_scene()
    body = import_fbx()
    #    mesh_obj = bpy.data.objects["SMPL-shapes-female"]
    deselect_all()
    #    select_obj(mesh_obj)

    # Assume obj is already 'SMPL-female' (your variable)
    obj = bpy.data.objects["SMPL-female"]  # Or however you have it

    # Get the child mesh by name
    mesh_name = "SMPL-mesh-female"
    mesh = next((child for child in obj.children if child.name == mesh_name), None)

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
    #    print("Here")
    #    bpy.data.objects
    #    print(body.keys())
    #
    shape_keys = mesh.data.shape_keys.key_blocks
    # Find and set the specific shape key
    target_key = 'Shape00'  # Matches the UI name
    shape_keys[1].value = 1
    print(shape_keys[1].name)
    if target_key in shape_keys:
        print("Found")
        shape_keys[target_key].value = 1.0  # Set to 1.0 (or e.g., 0.5 for half influence)

        # Optional: Update the view (forces refresh)
    bpy.context.view_layer.update()


main()

