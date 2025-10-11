import bpy
import os

# ──────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────
# Input paths
FBX_BODY_PATH        = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m_fix.fbx"
OBJ_GARMENT_PATH  = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_garment_modified.obj"

# Canonical object names to assign after import
CANON_ARMATURE_NAME = "Armature"
CANON_BODY_NAME     = "SMPL_Body"
CANON_GARMENT_NAME  = "Garment"

# Optional canonical collections (created if missing; set to None to skip)
CANON_COLL_RIG     = "Rig"
CANON_COLL_BODY    = "Body"
CANON_COLL_GARMENT = "Garment"

# OBJ axis config (use instead of post-import rotation)
GARMENT_IMPORT_FORWARD = "NEGATIVE_Y"
GARMENT_IMPORT_UP      = "Z"

# Fixed garment scaling
GARMENT_SCALE = 0.01

# Simulation toggles & tunables
ADD_CLOTH_SIMULATION = True
USE_SHRINKWRAP_OFFSET = True
SHRINKWRAP_OFFSET     = 0.004   # 4 mm
CLOTH_QUALITY_STEPS   = 12
CLOTH_COLL_QUALITY    = 8
CLOTH_OBJ_GAP         = 0.006   # object↔cloth (m)
CLOTH_SELF_GAP        = 0.003   # cloth↔cloth (m)
COLLIDER_THICK_OUTER  = 0.010   # collider shell outward (m)
COLLIDER_THICK_INNER  = 0.003

# Optional: build a separate "collision shell" (hidden, slightly inflated copy)
USE_COLLISION_SHELL   = False
SHELL_NAME            = "CollisionShell"
SHELL_SOLIDIFY_THICK  = 0.008   # outward thickness (m)
SHELL_VIEWPORT_HIDE   = True
SHELL_RENDER_HIDE     = True
# ──────────────────────────────────────────────────────────────


# ---------------------------- Utils ----------------------------
def _mod_index(obj, name):
    return obj.modifiers.find(name)

def _move_modifier_to_index(obj, name, target_index, max_steps=128):
    """Safely move a modifier to a target index; prevents infinite loops."""
    ensure_object_mode()
    if name not in [m.name for m in obj.modifiers]:
        return False
    steps = 0
    prev = -1
    while steps < max_steps:
        idx = _mod_index(obj, name)
        if idx == -1 or idx == target_index:
            return True
        # If we can't move further (idx didn't change), stop to avoid infinite loop
        if idx == prev:
            # Give one last try in the other direction, then bail
            if idx > target_index:
                bpy.ops.object.modifier_move_up(modifier=name)
            elif idx < target_index:
                bpy.ops.object.modifier_move_down(modifier=name)
            if _mod_index(obj, name) == idx:
                return False
        if idx > target_index:
            bpy.ops.object.modifier_move_up(modifier=name)
        elif idx < target_index:
            bpy.ops.object.modifier_move_down(modifier=name)
        prev = idx
        steps += 1
    return False

def ensure_modifier_order(garment):
    """Enforce Armature → (Shrinkwrap optional) → Cloth at indices 0..2."""
    ensure_object_mode()

    names = {m.type: m.name for m in garment.modifiers}
    arm = next((m for m in garment.modifiers if m.type == 'ARMATURE'), None)
    cloth = next((m for m in garment.modifiers if m.type == 'CLOTH'), None)
    sw = next((m for m in garment.modifiers if m.type == 'SHRINKWRAP' and m.name.startswith("SW_")), None)

    # Desired indices:
    # 0: Armature (if present)
    # 1: SW_InitOffset (if present)
    # 2: Cloth
    target = 0
    if arm:
        _move_modifier_to_index(garment, arm.name, target); target += 1
    if sw:
        _move_modifier_to_index(garment, sw.name, target); target += 1
    if cloth:
        _move_modifier_to_index(garment, cloth.name, target)

def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')

def ensure_collection(name):
    if not name:
        return None
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def move_to_collection(obj, coll_name):
    if not coll_name:
        return
    coll = ensure_collection(coll_name)
    # unlink from all current collections, then link to target
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)

def apply_transforms(obj, loc=True, rot=True, scale=True):
    ensure_object_mode()
    deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=loc, rotation=rot, scale=scale)


# ---------------------------- Import ----------------------------
def clear_scene():
    ensure_object_mode()
    print("🧹 Clearing current Blender scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Remove orphans
    for block_type in [
        bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.textures,
        bpy.data.images, bpy.data.curves, bpy.data.lights, bpy.data.cameras, bpy.data.collections,
    ]:
        for block in list(block_type):
            try:
                block.user_clear()
                block_type.remove(block)
            except:
                pass
    print("✅ Scene cleared.\n")

def import_models():
    """Import FBX (body+rig) and OBJ (garment), assign canonical names/collections, and return refs."""
    print("📥 Importing models...")
    if not os.path.exists(FBX_BODY_PATH):
        raise FileNotFoundError(FBX_BODY_PATH)
    if not os.path.exists(OBJ_GARMENT_PATH):
        raise FileNotFoundError(OBJ_GARMENT_PATH)

    ensure_object_mode()

    # --- Import FBX ---
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=FBX_BODY_PATH, automatic_bone_orientation=True)
    after = set(bpy.data.objects)
    new_fbx = list(after - before)
    if not new_fbx:
        raise RuntimeError("FBX import produced no objects.")

    # Pick armature & body from imported batch
    armature = next((o for o in new_fbx if o.type == 'ARMATURE'), None)
    body     = next((o for o in new_fbx if o.type == 'MESH'), None)
    if not body:
        raise RuntimeError("FBX import: no MESH found for body.")

    # Canonical rename and collection placement
    if armature:
        armature.name = CANON_ARMATURE_NAME
        move_to_collection(armature, CANON_COLL_RIG)
    body.name = CANON_BODY_NAME
    move_to_collection(body, CANON_COLL_BODY)

    # --- Import OBJ (garment) with axis settings ---
    if not hasattr(bpy.ops.wm, "obj_import"):
        raise RuntimeError("This Blender build lacks 'bpy.ops.wm.obj_import' (OBJ importer).")
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(
        filepath=OBJ_GARMENT_PATH,
        forward_axis=GARMENT_IMPORT_FORWARD,
        up_axis=GARMENT_IMPORT_UP,
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

    garment.name = CANON_GARMENT_NAME
    move_to_collection(garment, CANON_COLL_GARMENT)

    print(f"🦴 Armature: {armature.name if armature else 'None'}")
    print(f"👤 Body: {body.name}")
    print(f"👕 Garment: {garment.name}\n")
    return armature, body, garment


# ------------------------- Rigging/Weights -------------------------
def transfer_weights_with_modifier(body, garment):
    """Robust vertex-group weight transfer via DataTransfer modifier."""
    print("🎨 Transferring weights via DataTransfer modifier ...")
    ensure_object_mode()
    deselect_all()
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)

    mod = garment.modifiers.new("DT_Weights", 'DATA_TRANSFER')
    mod.object = body
    mod.use_object_transform = True
    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}
    mod.vert_mapping = 'NEAREST'            # try 'NEAREST_FACE_INTERPOLATED' for smoother mapping
    mod.layers_vgroup_select_src = 'ALL'    # ACTIVE | ALL
    mod.layers_vgroup_select_dst = 'NAME'   # NAME | INDEX
    mod.mix_mode = 'REPLACE'
    mod.mix_factor = 1.0

    bpy.ops.object.modifier_apply(modifier=mod.name)
    print("✅ Weights transferred (modifier applied).\n")

def parent_to_armature(armature, garment):
    print("🔗 Parenting garment to armature ...")
    ensure_object_mode()
    deselect_all()
    garment.select_set(True)
    if armature:
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.parent_set(type='ARMATURE', keep_transform=True)
        # Ensure Armature modifier references this armature
        mod = next((m for m in garment.modifiers if m.type == 'ARMATURE'), None)
        if mod is None:
            mod = garment.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = armature
    print("✅ Parent/Armature set.\n")


# ----------------------- Cloth & Collisions -----------------------
def ensure_modifier_order(garment):
    """Ensure Armature → (Shrinkwrap) → Cloth on garment."""
    ensure_object_mode()
    arm_mod = next((m for m in garment.modifiers if m.type == 'ARMATURE'), None)
    cloth_mod = next((m for m in garment.modifiers if m.type == 'CLOTH'), None)
    if arm_mod and cloth_mod:
        # Move Armature to top
        while garment.modifiers.find(arm_mod.name) > 0:
            bpy.ops.object.modifier_move_up(modifier=arm_mod.name)
        # Place Cloth right after Armature
        while garment.modifiers.find(cloth_mod.name) < garment.modifiers.find(arm_mod.name) + 1:
            bpy.ops.object.modifier_move_down(modifier=cloth_mod.name)

def add_shrinkwrap_offset(garment, target, offset=0.004):
    """Add Shrinkwrap above Cloth (and below nothing but Armature), safely."""
    ensure_object_mode()
    if target is None or target.type != 'MESH' or garment is target:
        print("⚠️ Skipping Shrinkwrap: invalid or self target.")
        return

    sw = garment.modifiers.new("SW_InitOffset", 'SHRINKWRAP')
    sw.target = target
    sw.wrap_method = 'NEAREST_SURFACEPOINT'
    sw.offset = offset

    # Reorder safely (Armature -> SW -> Cloth)
    ensure_modifier_order(garment)

def strengthen_cloth_and_collisions(garment, collider_obj):
    """Raise cloth quality & collider thickness/gaps to reduce tunneling."""
    ensure_object_mode()
    cloth_mod = next((m for m in garment.modifiers if m.type == 'CLOTH'), None)
    if cloth_mod:
        cloth = cloth_mod.settings
        cloth.quality = CLOTH_QUALITY_STEPS
        coll = cloth_mod.collision_settings
        coll.use_collision = True
        coll.collision_quality = CLOTH_COLL_QUALITY
        coll.distance_min = CLOTH_OBJ_GAP
        coll.use_self_collision = True
        coll.self_distance_min = CLOTH_SELF_GAP
        coll.friction = 5.0
        coll.self_friction = 5.0

    # Enable Collision on collider_obj
    deselect_all()
    collider_obj.select_set(True)
    bpy.context.view_layer.objects.active = collider_obj
    if collider_obj.collision is None:
        bpy.ops.object.modifier_add(type='COLLISION')
    collider_obj.collision.use = True
    collider_obj.collision.thickness_outer = COLLIDER_THICK_OUTER
    collider_obj.collision.thickness_inner = COLLIDER_THICK_INNER
    if hasattr(collider_obj.collision, "use_deforming"):
        collider_obj.collision.use_deforming = True

def build_collision_shell(body, armature=None):
    """Make a hidden, slightly-inflated copy of the body for more forgiving collisions."""
    print("🛡️ Building collision shell...")
    ensure_object_mode()
    deselect_all()
    shell = body.copy()
    shell.data = body.data.copy()
    shell.name = SHELL_NAME
    bpy.context.scene.collection.objects.link(shell)

    # Deform with the same armature
    if armature:
        shell_mod = shell.modifiers.new(name="Armature", type='ARMATURE')
        shell_mod.object = armature

    # Inflate via Solidify (outward only)
    solid = shell.modifiers.new(name="ShellSolidify", type='SOLIDIFY')
    solid.thickness = SHELL_SOLIDIFY_THICK
    solid.offset = 1.0

    # Hide if desired
    shell.hide_viewport = SHELL_VIEWPORT_HIDE
    shell.hide_render = SHELL_RENDER_HIDE
    return shell

def setup_cloth_simulation(garment, body, armature=None):
    print("🧶 Adding cloth simulation and collision ...")
    ensure_object_mode()

    # Cloth on garment
    cloth_mod = garment.modifiers.new("ClothSim", 'CLOTH')
    cloth = cloth_mod.settings
    cloth.quality = CLOTH_QUALITY_STEPS
    cloth.use_pressure = False

    # Cloth collisions
    coll = cloth_mod.collision_settings
    coll.use_collision = True
    coll.collision_quality = CLOTH_COLL_QUALITY
    coll.distance_min = CLOTH_OBJ_GAP
    coll.use_self_collision = True
    coll.self_distance_min = CLOTH_SELF_GAP
    coll.friction = 5.0
    coll.self_friction = 5.0

    # Collider: shell (if enabled) else body
    collider = build_collision_shell(body, armature=armature) if USE_COLLISION_SHELL else body

    # Strengthen & enable collider physics
    strengthen_cloth_and_collisions(garment, collider)

    # Modifier order & initial offset
    ensure_modifier_order(garment)
    if USE_SHRINKWRAP_OFFSET:
        add_shrinkwrap_offset(garment, collider, offset=SHRINKWRAP_OFFSET)
        ensure_modifier_order(garment)  # ensure final positions after adding SW

    print("✅ Cloth & collisions configured.\n")


# ----------------------------- MAIN -----------------------------
def main():
    clear_scene()
    armature, body, garment = import_models()

    # Apply clean scales (keep loc/rot)
    apply_transforms(body,  loc=False, rot=False, scale=True)

    # Fixed garment scaling
    print(f"📏 Scaling garment by {GARMENT_SCALE} ...")
    apply_transforms(garment, loc=False, rot=False, scale=False)
    garment.scale = (GARMENT_SCALE, GARMENT_SCALE, GARMENT_SCALE)
    apply_transforms(garment, loc=False, rot=False, scale=True)
    print("✅ Garment scaled.\n")

    # Weights & parenting
    transfer_weights_with_modifier(body, garment)
    parent_to_armature(armature, garment)

    # Optional cloth sim
    if ADD_CLOTH_SIMULATION:
        setup_cloth_simulation(garment, body, armature=armature)

    print("🎉 Complete! Pose the armature or play the sim to test.")

if __name__ == "__main__":
    main()
