import bpy
import os

# ──────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────
OUTPUT_FBX        = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m.fbx"


FBX_BODY_PATH = OUTPUT_FBX          # Path to SMPL FBX
OBJ_GARMENT_PATH = r"D:\models\garment.obj"         # Path to garment OBJ
BODY_OBJECT_NAME = "SMPL_Body"                      # Will be renamed if needed
GARMENT_OBJECT_NAME = "Garment"
ARMATURE_OBJECT_NAME = "Armature"
ADD_CLOTH_SIMULATION = True                         # Set False for only rigging
GARMENT_SCALE = 0.01                                # Scale factor for garment

# ──────────────────────────────────────────────────────────────


def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')


def import_models():
    print("Importing models...")
    if os.path.exists(FBX_BODY_PATH):
        bpy.ops.import_scene.fbx(filepath=FBX_BODY_PATH, automatic_bone_orientation=True)
    else:
        raise FileNotFoundError(FBX_BODY_PATH)

    if os.path.exists(OBJ_GARMENT_PATH):
        bpy.ops.import_scene.obj(filepath=OBJ_GARMENT_PATH)
    else:
        raise FileNotFoundError(OBJ_GARMENT_PATH)


def get_objects():
    # Attempt to auto-detect objects
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

    print(f"Detected armature: {armature.name}")
    print(f"Detected body: {body.name}")
    print(f"Detected garment: {garment.name}")
    return armature, body, garment


def apply_transforms(obj):
    deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def scale_garment(garment, scale_factor):
    print(f"Scaling garment by {scale_factor} ...")
    deselect_all()
    garment.select_set(True)
    bpy.context.view_layer.objects.active = garment
    garment.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print("Garment scaled and transforms applied.")


def transfer_weights(body, garment):
    print("Transferring weights from body → garment ...")
    deselect_all()
    garment.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.data_transfer(
        use_reverse_transfer=True,
        data_type='VGROUP_WEIGHTS',
        vert_mapping='NEAREST',
        layers_select_src='NAME',
        layers_select_dst='ALL',
        use_auto_transform=True,
        use_create=True
    )
    print("Weights transferred.")


def parent_to_armature(armature, garment):
    print("Parenting garment to armature ...")
    deselect_all()
    garment.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE', keep_transform=True)


def setup_cloth_simulation(garment, body):
    print("Adding cloth simulation and collision ...")

    # Garment cloth
    cloth_mod = garment.modifiers.new("ClothSim", 'CLOTH')
    cloth = cloth_mod.settings
    cloth.quality = 10
    cloth.use_pressure = False
    cloth.self_collision_distance = 0.002
    cloth.use_self_collision = True
    cloth.collision_settings.distance_min = 0.003

    # Body collision
    body_collision = body.modifiers.new("Collision", 'COLLISION')
    body.collision.use = True
    body.collision.thickness_outer = 0.003
    body.collision.thickness_inner = 0.003

    print("Cloth and collision added. You can now play the animation to simulate.")


# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
def main():
    import_models()
    armature, body, garment = get_objects()

    # Apply transforms
    apply_transforms(body)

    # Scale garment (0.01 = 1% of original)
    scale_garment(garment, GARMENT_SCALE)

    # Apply transforms again for clean local coords
    apply_transforms(garment)

    # Transfer weights
    transfer_weights(body, garment)

    # Parent to armature
    parent_to_armature(armature, garment)

    # Optional cloth simulation
    if ADD_CLOTH_SIMULATION:
        setup_cloth_simulation(garment, body)

    print("✅ Dressing pipeline complete!")
    print("→ Move bones in Pose Mode or play simulation to test.")


if __name__ == "__main__":
    main()
