import trimesh
from PIL.ImageOps import scale
import torch

obj_path = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_modified.obj"
# obj_path_fbx = r"\\wsl.localhost\Ubuntu-22.04\home\shay\projects\GarVerseLOD\outputs\temp\coarse_garment\66859611_lbs_spbs_human_m.fbx"
obj_path_fbx = r"C:\Users\Lab\Downloads\fbx_as_obj.obj"
smpl_model_path = r"D:\projects\ClProjects\SMPL_Model"  # Download from SMPL website
scale_factor = 0.01

mesh_org = trimesh.load(obj_path)
mesh_fbx = trimesh.load(obj_path_fbx)
# Apply scaling factor
mesh_org.vertices *= scale_factor


print(f"Original mesh bounds: {mesh_org.bounds}")
print(f"Fbx mesh bounds: {mesh_fbx.bounds}")

v_org = torch.tensor(mesh_org.vertices, dtype=torch.float32, requires_grad=False)
v_fbx = torch.tensor(mesh_fbx.vertices[:mesh_org.vertices.shape[0],:], dtype=torch.float32, requires_grad=False)
# T = torch.tensor(torch.ones((3,3)), dtype=torch.float32)
# bias = torch.tensor(torch.zeros((3,1)), dtype=torch.float32)
# torch.linalg.lstsq(A,B) - Find X which minimize |AX-B|
# A - (N,4), B - (N,3)
A = torch.cat((v_org, torch.ones((v_org.shape[0], 1), dtype=torch.float32)), dim=1)
B = v_fbx

# Test
T = torch.randn(3,3)
bias = torch.randn(1,3)
# B = A @ torch.cat((T, bias), dim=0)
B = v_org @ T + bias

ls_result = torch.linalg.lstsq(A, B)

B_est = A @ ls_result.solution
error_norm = torch.norm(B - B_est)
print('ls fit error is ', error_norm.numpy(), '(abs), and ',(error_norm/ ((A.norm())+B.norm()/2)*100).numpy(), '(%)')
T = ls_result.solution[:3]
bias = ls_result.solution[3:]
print(f"A: {A}")
print(f"b: {b.T}")


