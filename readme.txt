This repo do all king of manipulation on a body 3D mesh input object:
* calc_body_measurements - calculate important dimensions of the body, input is .obj (3D mesh) file, currently using smpl model (can be switched to smplx)
* create_smpl_params_from_obj_as_npz.py - fits the parameters representation of the SMPL model, saved in .npz file
* first_working_script_mesh_to_fbx.py - creates a fbx file from the input .npz file - run in Blender
* import_fbx_and_garment_and_dress.py - import the fbx file into blender, import a garment obj file, dress the body with the garment - run in Blender

