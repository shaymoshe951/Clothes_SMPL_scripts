import bpy
import os

# ──────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────
OUTPUT_FBX        = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m.fbx"


FBX_BODY_PATH = OUTPUT_FBX          # Path to SMPL FBX
OBJ_GARMENT_PATH = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_garment_modified.obj"         # Path to garment OBJ
ADD_CLOTH_SIMULATION = True                         # Set False for only rigging
GARMENT_SCALE = 0.01                                # Scale factor for garment

# ──────────────────────────────────────────────────────────────

def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def clear_scene():
    ensure_object_mode()

    """Remove all existing objects, collections, and data blocks."""
    print("🧹 Clearing current Blender scene...")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Remove orphan data
    for block_type in [
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.collections,
    ]:
        for block in block_type:
            try:
                block.user_clear()
                block_type.remove(block)
            except:
                pass

    print("✅ Scene cleared.\n")


def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')


def import_models():
    print("📥 Importing models...")

    # Import SMPL FBX
    if os.path.exists(FBX_BODY_PATH):
        bpy.ops.import_scene.fbx(
            filepath=FBX_BODY_PATH,
            automatic_bone_orientation=True
        )
    else:
        raise FileNotFoundError(FBX_BODY_PATH)

    # Import Garment OBJ with Blender 4.x native importer
    if not hasattr(bpy.ops.wm, "obj_import"):
        raise RuntimeError("This Blender build lacks the native OBJ importer operator 'bpy.ops.wm.obj_import'.")

    if os.path.exists(OBJ_GARMENT_PATH):
        bpy.ops.wm.obj_import(
            filepath=OBJ_GARMENT_PATH,
            # optional axis options if needed:
            # forward_axis='-Z', up_axis='Y',
            # split_mode='ON', validate_meshes=True
        )
    else:
        raise FileNotFoundError(OBJ_GARMENT_PATH)


def get_objects():
    """Auto-detect armature, body, and garment objects."""
    armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    body = next((o for o in bpy.data.objects if o.type == 'MESH' and 'body' in o.name.lower()), None)
    garment = next((o for o in bpy.data.objects if o.type == 'MESH' and 'garment' in o.name.lower()), None)

    if not body:
        body = next((o for o in bpy.data.objects if o.type == 'MESH'), None)
    if not garment:
        meshes = [o for o in bpy.data.objects if o.type == 'MESH']
        garment = meshes[-1] if len(meshes) > 1 else meshes[0]

    if not armature or not body or not garment:
        raise RuntimeError("Couldn't detect armature/body/garment objects")

    print(f"🦴 Armature: {armature.name}")
    print(f"👤 Body: {body.name}")
    print(f"👕 Garment: {garment.name}\n")
    return armature, body, garment


def apply_transforms(obj):
    ensure_object_mode()

    deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def scale_garment(garment, scale_factor):
    print(f"📏 Scaling garment by {scale_factor} ...")
    ensure_object_mode()

    deselect_all()
    garment.select_set(True)
    bpy.context.view_layer.objects.active = garment
    garment.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print("✅ Garment scaled and transforms applied.\n")


def transfer_weights(body, garment):
    print("🎨 Transferring weights from body → garment ...")
    ensure_object_mode()
    deselect_all()

    # Active = SOURCE (body), Selected non-active = DESTINATION (garment)
    garment.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body

    bpy.ops.object.data_transfer(
        data_type='VGROUP_WEIGHTS',
        vert_mapping='NEAREST',
        layers_select_src='ALL',   # ✅ ACTIVE | ALL
        layers_select_dst='NAME',  # ✅ NAME | INDEX
        use_auto_transform=True,
        use_create=True,
        mix_mode='REPLACE',
        mix_factor=1.0,
        use_reverse_transfer=False
    )
    print("✅ Weights transferred.\n")

def parent_to_armature(armature, garment):
    print("🔗 Parenting garment to armature ...")
    ensure_object_mode()

    deselect_all()
    garment.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE', keep_transform=True)
    print("✅ Parent relationship created.\n")


def setup_cloth_simulation(garment, body):
    print("🧶 Adding cloth simulation and collision ...")
    ensure_object_mode()

    # ── Cloth on garment
    cloth_mod = garment.modifiers.new("ClothSim", 'CLOTH')
    cloth = cloth_mod.settings
    cloth.quality = 10
    cloth.use_pressure = False

    # Cloth collision settings (modifier-level)
    coll = cloth_mod.collision_settings
    coll.collision_quality = 4
    coll.use_collision = True
    coll.distance_min = 0.003          # object ↔ cloth gap
    coll.use_self_collision = True
    coll.self_distance_min = 0.002     # cloth ↔ cloth gap
    coll.friction = 5.0
    coll.self_friction = 5.0
    coll.impulse_clamp = 0.0
    coll.self_impulse_clamp = 0.0

    # ── Enable the BODY as a collider (physics Collision block)
    # Make body active for the operator to affect it
    deselect_all()
    body.select_set(True)
    bpy.context.view_layer.objects.active = body

    # If the Collision physics block doesn't exist yet, create it
    if body.collision is None:
        bpy.ops.object.modifier_add(type='COLLISION')  # creates body.collision

    # Now safe to access body.collision
    body.collision.use = True
    body.collision.thickness_outer = 0.003
    body.collision.thickness_inner = 0.003
    # Optional:
    # body.collision.cloth_friction = 5.0
    # body.collision.use_normal = True
    # body.collision.use_culling = False

    print("✅ Cloth & collisions configured for Blender 4.5 API.")


# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
def main():
    clear_scene()
    import_models()

    armature, body, garment = get_objects()

    # Apply transforms
    apply_transforms(body)

    # Scale garment (e.g. 0.01)
    scale_garment(garment, GARMENT_SCALE)

    # Reapply transforms for clean scale
    apply_transforms(garment)

    # Transfer weights
    transfer_weights(body, garment)

    # Parent to armature
    parent_to_armature(armature, garment)

    # Optional cloth simulation
    if ADD_CLOTH_SIMULATION:
        setup_cloth_simulation(garment, body)

    print("🎉 All done! The garment is now dressed on the SMPL body.\n"
          "→ Pose the armature or play simulation to test.\n")


if __name__ == "__main__":
    main()
