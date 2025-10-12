import pickle
import torch
from smplx import SMPL
import trimesh

# --- paths ---
MODEL_PATH = r"D:\projects\ClProjects\SMPL_Model"      # folder with SMPL_MALE.pkl, SMPL_FEMALE.pkl, etc.
REG_PATH = r"D:\projects\ClProjects\BlenderGarmentExample\registration.pkl"
OUT_PATH = r"D:\projects\ClProjects\BlenderGarmentExample\output.obj"

# --- load registration file ---
with open(REG_PATH, 'rb') as f:
    reg = pickle.load(f, encoding='latin1')

# select gender model
gender = reg.get('gender', 'neutral')
model = SMPL(MODEL_PATH, gender=gender, batch_size=1)

# convert to tensors
betas = torch.tensor(reg['betas'][None, :], dtype=torch.float32)
pose = torch.tensor(reg['pose'][None, :], dtype=torch.float32)
trans = torch.tensor(reg['trans'][None, :], dtype=torch.float32)

# forward the model
output = model(betas=betas, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=trans)
vertices = output.vertices[0].detach().cpu().numpy()
faces = model.faces

# --- export to .obj ---
mesh = trimesh.Trimesh(vertices, faces, process=False)
mesh.export(OUT_PATH)
print(f"Saved to {OUT_PATH}")
