import numpy as np
import torch
from smplx import SMPL
import trimesh
import os

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
    return mesh

def fit_smpl_to_obj(mesh, smpl_model_path, gender='female', device='cpu'):
    """Fit SMPL parameters to an OBJ mesh with optional scaling"""

    # mesh = trimesh.load(obj_path)
    # print(f"Original mesh bounds: {mesh.bounds}")
    # mesh.vertices *= scale_factor
    # print(f"Scaled mesh bounds (scale={scale_factor}): {mesh.bounds}")

    target_vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)

    smpl = SMPL(model_path=smpl_model_path, gender=gender).to(device)

    betas = torch.zeros(10, requires_grad=True, device=device)
    body_pose = torch.zeros(69, requires_grad=True, device=device)
    global_orient = torch.zeros(3, requires_grad=True, device=device)
    transl = torch.zeros(3, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([betas, body_pose, global_orient, transl], lr=0.1)
    # Scheduler: Reduce on validation loss plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.4, patience=10, verbose=True, threshold=0.001, cooldown=0, min_lr=2e-3
    )
    print("Fitting SMPL parameters...")
    sqr_loss_vector = []
    max_err_vector = []
    for i in range(__MAX_NUMBER_OF_FIT_TRAINING_ITERS__):
        optimizer.zero_grad()

        output = smpl(betas=betas.unsqueeze(0),
                      body_pose=body_pose.unsqueeze(0),
                      global_orient=global_orient.unsqueeze(0),
                      transl=transl.unsqueeze(0))

        loss = torch.nn.functional.mse_loss(output.vertices[0], target_vertices)
        loss.backward()
        optimizer.step()
        sqr_loss_vector.append(loss.sqrt().item())
        max_error = (output.vertices[0] - target_vertices).abs().max().item()
        max_err_vector.append(max_error)

        # Step the scheduler (triggers reduction if plateau)
        scheduler.step(loss)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}")
            print("max abs error:",max_error)

        if max_error < __ALLOWED_ERROR_TRESHOLD_IN_M__:
            print(f"Converged at iteration {i}, max error: {max_error:.6f}, loss: {loss.item():.6f}")
            break

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
        'betas': betas.detach().cpu().numpy(),
        'body_pose': body_pose.detach().cpu().numpy(),
        'global_orient': global_orient.detach().cpu().numpy(),
        'transl': transl.detach().cpu().numpy(),
        'gender': gender,
        'scale_factor': scale_factor
    }

    return params, output.vertices[0].detach().cpu().numpy(), mesh.faces
