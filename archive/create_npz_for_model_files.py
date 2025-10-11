"""
SMPL Model Pre-processor
Run this script in regular Python (NOT in Blender) to convert SMPL .pkl files to .npz format
This only needs to be run once per SMPL model (male/female/neutral)
"""

import numpy as np
import pickle
import os

# ======================== CONFIGURATION ========================
smpl_model_path = r"D:\projects\ClProjects\SMPL_Model"
output_path = r"D:\projects\ClProjects\SMPL_Model"  # Where to save .npz files


# ======================== FUNCTIONS ========================

def convert_smpl_to_numpy(smpl_data):
    """Convert all SMPL data to pure numpy arrays"""
    result = {}

    for key, value in smpl_data.items():
        # Handle scipy sparse matrices
        if hasattr(value, 'toarray'):
            result[key] = value.toarray()
        # Handle chumpy arrays
        elif hasattr(value, 'r'):
            result[key] = np.array(value.r)
        # Already numpy
        elif isinstance(value, np.ndarray):
            result[key] = value
        else:
            result[key] = value

    return result


def process_smpl_model(pkl_path, output_npz_path):
    """
    Load SMPL .pkl file and save essential data to .npz

    Args:
        pkl_path: Path to SMPL .pkl file
        output_npz_path: Path to save the .npz file
    """
    print(f"Loading {pkl_path}...")

    with open(pkl_path, 'rb') as f:
        smpl_data = pickle.load(f, encoding='latin1')

    print("Converting to numpy arrays...")
    smpl_numpy = convert_smpl_to_numpy(smpl_data)

    # Extract only the essential data needed for FBX export
    essential_data = {
        'v_template': smpl_numpy['v_template'],  # (6890, 3) - Template vertices
        'shapedirs': smpl_numpy['shapedirs'],  # Shape blend shapes
        'J_regressor': smpl_numpy['J_regressor'],  # Joint regressor (24, 6890)
        'weights': smpl_numpy['weights'],  # LBS weights (6890, 24)
        'f': smpl_numpy['f'],  # Faces (13776, 3)
        'kintree_table': smpl_numpy['kintree_table']  # Kinematic tree
    }

    # Add posedirs if available
    if 'posedirs' in smpl_numpy:
        essential_data['posedirs'] = smpl_numpy['posedirs']

    print(f"Saving to {output_npz_path}...")
    np.savez_compressed(output_npz_path, **essential_data)

    print(f"✓ Successfully saved!")
    print(f"  v_template shape: {essential_data['v_template'].shape}")
    print(f"  shapedirs shape: {essential_data['shapedirs'].shape}")
    print(f"  weights shape: {essential_data['weights'].shape}")
    print(f"  faces shape: {essential_data['f'].shape}")


def main():
    """Process all SMPL models"""
    models = {
        'male': 'SMPL_MALE.pkl',
        'female': 'SMPL_FEMALE.pkl',
        'neutral': 'SMPL_NEUTRAL.pkl'
    }

    for gender, filename in models.items():
        pkl_path = os.path.join(smpl_model_path, filename)

        if not os.path.exists(pkl_path):
            print(f"⚠ Skipping {gender}: {pkl_path} not found")
            continue

        output_npz = os.path.join(output_path, f'SMPL_{gender.upper()}.npz')

        try:
            process_smpl_model(pkl_path, output_npz)
            print()
        except Exception as e:
            print(f"✗ Error processing {gender}: {e}")
            print()

    print("=" * 60)
    print("Pre-processing complete!")
    print(f"Now use the Blender script to create FBX files.")


if __name__ == "__main__":
    main()