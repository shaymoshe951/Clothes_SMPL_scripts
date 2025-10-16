import numpy as np
import torch
from smplx import SMPL
import trimesh
import itertools

__ALLOWED_ERROR_TRESHOLD_IN_M__ = 0.015  # 2 cm
__MAX_NUMBER_OF_FIT_TRAINING_ITERS__ = 3000

def load_mesh_obj(obj_path, flag_verify_boundaries = True, scale_factor=0.01):
    """Load and scale a mesh from an OBJ file."""
    mesh = trimesh.load(obj_path)
    if scale_factor != 1.0:
        print(f"Original mesh bounds: {mesh.bounds}")
        mesh.vertices *= scale_factor
        print(f"Scaled mesh bounds (scale={scale_factor}): {mesh.bounds}")
    if flag_verify_boundaries:
        max_bounds = np.max(mesh.bounds[1]-mesh.bounds[0])
        if max_bounds > 3.0 or max_bounds < 0.1:
            raise ValueError(f"Mesh bounds seem off: {mesh.bounds}")

    # remove mean of mesh vertices
    # mesh.vertices -= mesh.vertices.mean(axis=0, keepdims=True)
    # mesh.vertices -= mesh.vertices[4922,:]
    # rot_matrix = trimesh.transformations.rotation_matrix(
    #     np.radians(90), [0, 1, 0], point=None
    # )
    # mesh.apply_transform(rot_matrix)
    mesh.vertices -= mesh.vertices[4922,:]
    rot_matrix = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0], point=None
    )
    mesh.apply_transform(rot_matrix)
    return mesh


def cube_rotations(include_reflections: bool = False):
    """
    Generate all rotation matrices that map a unit cube to itself.

    Returns:
        List[np.ndarray] of shape (3, 3).
        - By default returns the 24 proper rotations (determinant = +1).
        - If include_reflections=True, returns all 48 symmetries (det = ±1).
    """
    mats = []
    seen = set()

    # Permute axes (columns) and assign ± signs (one nonzero per row & column).
    for perm in itertools.permutations(range(3)):  # 6 permutations
        for signs in itertools.product([-1, 1], repeat=3):  # 8 sign choices
            M = np.zeros((3, 3), dtype=int)
            for i, j in enumerate(perm):
                M[i, j] = signs[i]

            det = round(np.linalg.det(M))
            if det == 0:
                continue  # shouldn't happen, but be safe

            if include_reflections or det == 1:
                key = tuple(M.ravel())
                if key not in seen:
                    seen.add(key)
                    mats.append(M)

    return mats

def gen_all_possible_rotations(vertices):
    """Generate all 24 rotated versions of the input vertices."""
    rotation_matrices = cube_rotations(include_reflections=True)
    all_rotated_vertices = []

    for R in rotation_matrices:
        R_tensor = torch.tensor(R, dtype=vertices.dtype, device=vertices.device)
        rotated = vertices @ R_tensor.T  # Matrix multiplication
        all_rotated_vertices.append(rotated)

    return torch.stack(all_rotated_vertices, dim=0), rotation_matrices  # Shape: (24, N, 3)

def fit_smpl_to_obj(mesh, smpl_model_path, gender='female', device='cpu', flag_debug = False, flag_fit_mesh_to_smpl=False):
    """Fit SMPL parameters to an OBJ mesh with optional scaling"""

    # mesh = trimesh.load(obj_path)
    # print(f"Original mesh bounds: {mesh.bounds}")
    # mesh.vertices *= scale_factor
    # print(f"Scaled mesh bounds (scale={scale_factor}): {mesh.bounds}")

    target_vertices_tmp = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
    target_vertices_allrots, rotation_matrices = gen_all_possible_rotations(target_vertices_tmp)

    smpl = SMPL(model_path=smpl_model_path, gender=gender).to(device)

    betas = torch.zeros(10, requires_grad=True, device=device)
    body_pose = torch.zeros(69, requires_grad=True, device=device)
    global_orient = torch.zeros(3, requires_grad=True, device=device)
    transl = torch.zeros(3, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([betas, body_pose, global_orient, transl], lr=0.1)
    # Scheduler: Reduce on validation loss plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.4, patience=10, threshold=0.001, cooldown=0, min_lr=2e-3
    )
    print("Fitting SMPL parameters...")
    sqr_loss_vector = []
    max_err_vector = []
    if flag_fit_mesh_to_smpl and len(target_vertices_tmp) != 6890:
        flag_fit_mesh_to_smpl = True
    else:
        flag_fit_mesh_to_smpl = False
    for i in range(__MAX_NUMBER_OF_FIT_TRAINING_ITERS__):
        optimizer.zero_grad()

        output = smpl(betas=betas.unsqueeze(0),
                      body_pose=body_pose.unsqueeze(0),
                      global_orient=global_orient.unsqueeze(0),
                      transl=transl.unsqueeze(0))
        if flag_fit_mesh_to_smpl:
            # Compute loss with Trimesh: Closest distances from SMPL verts to target surface
            smpl_verts = output.vertices[0]  # Keep as tensor [6890, 3]
            dists, _ = torch.cdist(smpl_verts, target_vertices_tmp).min(dim=1)
            # Min dist per SMPL vert [6890]
            loss = dists.mean()  # Mean Chamfer loss (tensor, differentiable)
            max_error = dists.max().item()  # Scalar for logging
        else:
            # Choose best rotation
            if i == 0:
                for j in range(len(target_vertices_allrots)):
                    loss_tmp = torch.nn.functional.mse_loss(output.vertices[0], target_vertices_allrots[j])
                    print("j=",j, ", loss_tmp:",loss_tmp)
                    if j == 0 or loss_tmp < loss:
                        loss = loss_tmp
                        max_error = (output.vertices[0] - target_vertices_allrots[j]).abs().max().item()
                print("j=",j)
                print("Rotation Matrix:\n", rotation_matrices[j])
                target_vertices = target_vertices_tmp # target_vertices_allrots[j] #
            loss = torch.nn.functional.mse_loss(output.vertices[0], target_vertices)
            max_error = (output.vertices[0] - target_vertices).abs().max().item()
        loss.backward()
        optimizer.step()
        sqr_loss_vector.append(loss.sqrt().item())
        max_err_vector.append(max_error)

        # Step the scheduler (triggers reduction if plateau)
        scheduler.step(loss)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}")
            print("max abs error:",max_error)

        if max_error < __ALLOWED_ERROR_TRESHOLD_IN_M__:
            print(f"Converged at iteration {i}, max error: {max_error:.6f}, loss: {loss.item():.6f}")
            break

    if flag_debug:
        vertices = output.vertices[0].detach().cpu().numpy()  # (6890, 3)
        faces = smpl.faces
        fitted_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        fitted_mesh.export(r"C:\Users\Lab\Downloads\fitted_mesh.obj")

        import matplotlib.pyplot as plt
        epochs = list(range(1, len(sqr_loss_vector) + 1))  # x-axis: epochs 1 to 10

        # If losses are a torch tensor, convert: loss_vector = losses.cpu().numpy().tolist()

        # Create the plot
        plt.figure(figsize=(8, 5))  # Optional: Set figure size
        plt.plot(epochs, max_err_vector, marker='o', linewidth=2, markersize=4, color='g', label='Max Error')
        plt.plot(epochs, sqr_loss_vector, marker='_', linewidth=2, markersize=4, color='b', label='Training Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    params = {
        'betas': betas.detach().cpu(),
        'body_pose': body_pose.detach().cpu(),
        'global_orient': global_orient.detach().cpu(),
        'transl': transl.detach().cpu(),
        'gender': gender,
    }

    return smpl, params

def axis_angle_to_rot_mat(aa):
    """
    Convert axis-angle vector to 3x3 rotation matrix using Rodrigues' formula.

    Args:
        aa (torch.Tensor): Axis-angle vector(s) of shape (..., 3)

    Returns:
        torch.Tensor: Rotation matrix/matrices of shape (..., 3, 3)
    """
    batch_size = 1 if aa.dim() == 1 else aa.shape[0]
    if aa.dim() == 1:
        aa = aa.unsqueeze(0)

    angle = torch.norm(aa, dim=-1, keepdim=True)
    # Handle zero angle case
    axis = aa / torch.clamp(angle, min=1e-8)

    # Skew-symmetric matrix K
    K = torch.zeros((batch_size, 3, 3), device=aa.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    # Identity
    I = torch.eye(3, device=aa.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # Rodrigues' formula
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    R = I + sin_a.unsqueeze(-1).unsqueeze(-1) * K + (1 - cos_a.unsqueeze(-1).unsqueeze(-1)) * torch.matmul(K, K)

    if batch_size == 1:
        R = R.squeeze(0)

    return R

