import numpy as np
import hydra
import omegaconf
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
from einops.layers.torch import Rearrange
import sys
from omegaconf import OmegaConf
from mpmwrapper import MPMWrapperLearnableStress
import taichi as ti
import random

from houdini_visualization_gradio import visualize_simulation

torch.autograd.set_detect_anomaly(True)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_random_seed(seed)


class FprojNN(torch.nn.Module):
    def __init__(self, activation, hidden_size, device, embed_dim,
                 trajectory_latents):
        super(FprojNN, self).__init__()
        self.device = device

        if activation == "gelu":
            self.activation = torch.nn.GELU()

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)
        self.device = "cuda"

        self.fc1 = nn.Linear(27 + embed_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, 9, bias=True)

        self.trajectory_latents = trajectory_latents

    def Ftmp_U_Vt_transform(self, Ftmp, U, V):
        if (len((torch.isnan(Ftmp) == True).nonzero()) > 0 or
                len((torch.isinf(Ftmp) == True).nonzero()) > 0):
            import ipdb; ipdb.set_trace()

        U_flatten    = self.flatten(U)
        Vt_flatten   = self.flatten(V.transpose(1, 2))
        Ftmp_flatten = self.flatten(Ftmp)
        Ftmp_input   = torch.cat([Ftmp_flatten, U_flatten, Vt_flatten], dim=-1)
        return Ftmp_input

    def _build_latent_tensor(self, num_particles, particle_mat_ids):
        parts = []
        for mat_id in range(len(self.trajectory_latents)):
            mask = (particle_mat_ids == mat_id)
            if not mask.any():
                continue
            n = mask.sum().item()
            latent = self.trajectory_latents[mat_id].weight
            parts.append((mask, latent.expand(n, -1)))

        embed_dim = self.trajectory_latents[0].weight.shape[-1]
        out = torch.zeros(num_particles, embed_dim,
                          device=particle_mat_ids.device,
                          dtype=torch.float32)
        for mask, lat in parts:
            out[mask] = lat
        return out

    def forward(self, Ftmp, U, V, traj_id, particle_mat_ids=None):
        Ftmp_flatten = self.Ftmp_U_Vt_transform(Ftmp, U, V)

        if particle_mat_ids is not None:
            latent_particles = self._build_latent_tensor(
                Ftmp.shape[0], particle_mat_ids
            )
        else:
            latent_particles = (
                self.trajectory_latents[0].weight[traj_id]
                    .unsqueeze(0)
                    .repeat(Ftmp.shape[0], 1)
            )

        x   = self.activation(self.fc1(torch.cat([Ftmp_flatten, latent_particles], dim=-1)))
        x   = self.activation(self.fc2(x))
        out = self.fc3(x)

        Fproj = Ftmp + out.view(out.shape[0], 3, 3)
        return Fproj


class StressNN(torch.nn.Module):
    def __init__(self, activation, hidden_size, embed_dim, device,
                 trajectory_latents):
        super(StressNN, self).__init__()
        self.device = device

        if activation == "gelu":
            self.activation = torch.nn.GELU()

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)
        self.device = "cuda"

        self.fc1 = nn.Linear(16 + 9 + embed_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc5 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc6 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc7 = nn.Linear(hidden_size, 9, bias=True)

        self.trajectory_latents = trajectory_latents

    def FFt_logJ_sigma_J_logJ1_J1_transform(self, F):
        Ft  = F.transpose(1, 2)
        FFt = torch.matmul(F, Ft)

        J  = torch.max(torch.det(F[:, :, :]), torch.Tensor([1e-6]).cuda())
        J1 = torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda())

        _, sigma, _ = torch.svd(F)
        FFt_flatten  = self.flatten(FFt)
        J            = J.unsqueeze(-1)
        J1           = J1.unsqueeze(-1)

        strain = torch.cat(
            [FFt_flatten, torch.log(J), sigma, J, torch.log(J1), J1], dim=-1
        )
        return strain

    def _build_latent_tensor(self, num_particles, particle_mat_ids):
        parts = []
        for mat_id in range(len(self.trajectory_latents)):
            mask = (particle_mat_ids == mat_id)
            if not mask.any():
                continue
            n = mask.sum().item()
            latent = self.trajectory_latents[mat_id].weight
            parts.append((mask, latent.expand(n, -1)))

        embed_dim = self.trajectory_latents[0].weight.shape[-1]
        out = torch.zeros(num_particles, embed_dim,
                          device=particle_mat_ids.device,
                          dtype=torch.float32)
        for mask, lat in parts:
            out[mask] = lat
        return out

    def forward(self, F, C, traj_id, particle_mat_ids=None):
        strain = self.FFt_logJ_sigma_J_logJ1_J1_transform(F)

        if particle_mat_ids is not None:
            latent_particles = self._build_latent_tensor(
                F.shape[0], particle_mat_ids
            )
        else:
            latent_particles = (
                self.trajectory_latents[0].weight[traj_id]
                    .unsqueeze(0)
                    .repeat(F.shape[0], 1)
            )

        C_flatten = self.flatten(C)
        x   = self.activation(self.fc1(torch.cat([strain, C_flatten, latent_particles], dim=-1)))
        x   = self.activation(self.fc2(x))
        x   = self.activation(self.fc3(x))
        x   = self.activation(self.fc4(x))
        x   = self.activation(self.fc5(x))
        x   = self.activation(self.fc6(x))
        out = self.fc7(x)

        stress           = out.view(out.shape[0], 3, 3)
        stress_symmetric = 0.5 * (stress + stress.permute(0, 2, 1))
        stress_symmetric = torch.clamp(stress_symmetric, -1e4, 1e4)
        return stress_symmetric


def load_particle_mat_ids(traj_data_dir: str, num_particles: int,
                          device: str) -> torch.Tensor:
    id_path = os.path.join(traj_data_dir, 'MaterialID.pt')
    if os.path.exists(id_path):
        mat_ids_raw = torch.load(id_path).long().to(device)
        assert mat_ids_raw.shape[0] == num_particles, (
            f"MaterialID.pt has {mat_ids_raw.shape[0]} entries but "
            f"the scene has {num_particles} particles."
        )
        unique_ids = mat_ids_raw.unique(sorted=True)
        remap = {old_id.item(): new_id for new_id, old_id in enumerate(unique_ids)}
        mat_ids = torch.tensor(
            [remap[i.item()] for i in mat_ids_raw],
            dtype=torch.long, device=device
        )
        print(f"Loaded particle material IDs from {id_path}. "
              f"Raw unique IDs: {unique_ids.tolist()} -> "
              f"Remapped to: {list(remap.values())}")
    else:
        print(f"Warning: {id_path} not found - all particles assigned to "
              f"material 0 (single-material fallback).")
        mat_ids = torch.zeros(num_particles, dtype=torch.long, device=device)
    return mat_ids


def load_multi_material_latents(latent_path: str, num_materials: int,
                                embed_dim: int, device: str) -> nn.ModuleList:
    """
    Load saved per-material latents from a checkpoint produced by the
    multi-material inference script and return them as an nn.ModuleList
    of nn.Embedding modules, one per material.

    Args:
        latent_path:   path to the .pt file saved by the inference script,
                       e.g. 'latents_multi_material/latent_epoch_99.pt'
        num_materials: number of distinct materials expected
        embed_dim:     latent dimensionality
        device:        target device string

    Returns:
        nn.ModuleList of length num_materials, each element is an
        nn.Embedding of shape (1, embed_dim) with freeze=False
    """
    saved = torch.load(latent_path, map_location=device, weights_only=False)
    trajectory_latents = nn.ModuleList()
    for mat_id in range(num_materials):
        key = f"material_{mat_id}"
        assert key in saved, (
            f"Key '{key}' not found in checkpoint. "
            f"Available keys: {list(saved.keys())}"
        )
        weight = saved[key]  # (1, embed_dim) or nn.Embedding
        # Handle case where the full Embedding object was saved
        if isinstance(weight, nn.Embedding):
            weight = weight.weight.detach()
        emb = nn.Embedding.from_pretrained(weight, freeze=False).to(device)
        trajectory_latents.append(emb)
    print(f"Loaded {num_materials} material latents from {latent_path}.")
    return trajectory_latents


@hydra.main(config_path='configs', config_name='sim')
def main(cfg: omegaconf.DictConfig):

    ## Logging ##
    save_dir  = cfg['train_cfg']['save_dir']
    local_dir = cfg['train_cfg']['local_dir']
    logger    = logging.getLogger()

    device            = 'cuda'
    num_sim_steps     = cfg['visualization_cfg']['num_frames']
    cuda_chunk_size   = cfg['train_cfg']['cuda_chunk_size']
    particles_ti_root = cfg['train_cfg']['particles_ti_root']

    os.makedirs(f"{local_dir}/{save_dir}/logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{local_dir}/{save_dir}/logs")
    fh = logging.FileHandler(f"{local_dir}/{save_dir}/log.txt")
    with open(f"{local_dir}/{save_dir}/config.yaml", 'w') as cfg_file:
        OmegaConf.save(cfg, cfg_file)

    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ## Load trajectory data ##
    traj_data_dir  = os.path.join(local_dir, cfg['train_cfg']['traj_data_dir'])
    traj_data_orig = torch.load(os.path.join(traj_data_dir, 'GtX.pt'))
    traj_data_orig = torch.tensor(traj_data_orig).to(device)  # T x P x 3

    traj_idx      = 0
    num_particles = traj_data_orig.shape[1]

    ## Load per-particle material IDs ##
    particle_mat_ids = load_particle_mat_ids(
        traj_data_dir, num_particles, device
    )
    num_materials = cfg['train_cfg'].get('num_materials', 1)
    inferred_num_mats = len(particle_mat_ids.unique())
    if inferred_num_mats > num_materials:
        print(f"  Config num_materials={num_materials} < inferred "
              f"{inferred_num_mats}; using inferred value.")
        num_materials = inferred_num_mats
    print(f"Scene has {num_materials} distinct material(s).")

    ## Load saved multi-material latents ##
    latent_ckpt_path = os.path.join(
        local_dir, cfg['train_cfg']['traj_latent_path']
    )
    trajectory_latents = load_multi_material_latents(
        latent_path=latent_ckpt_path,
        num_materials=num_materials,
        embed_dim=cfg['train_cfg']['embed_dim'],
        device=device,
    )

    ## Build model ##
    fproj_model = FprojNN(
        activation=cfg['train_cfg']['nn_activation'],
        hidden_size=cfg['train_cfg']['hidden_size'],
        device='cuda',
        embed_dim=cfg['train_cfg']['embed_dim'],
        trajectory_latents=None,
    ).to(device)

    stress_model = StressNN(
        activation=cfg['train_cfg']['nn_activation'],
        hidden_size=cfg['train_cfg']['hidden_size'],
        embed_dim=cfg['train_cfg']['embed_dim'],
        device='cuda',
        trajectory_latents=None,
    ).to(device)

    if cfg['train_cfg']['load_model']:
        ckpt = torch.load(os.path.join(local_dir, cfg['train_cfg']['load_model']))
        ckpt_stress = {
            key.replace('module.', '')
               .replace('_stress', '')
               .replace('stress_model.', ''): value
            for key, value in ckpt.items() if 'stress' in key
        }
        ckpt_fproj = {
            key.replace('module.', '')
               .replace('_fproj', '')
               .replace('fproj_model.', ''): value
            for key, value in ckpt.items() if 'fproj' in key
        }
        stress_model.load_state_dict(ckpt_stress)
        fproj_model.load_state_dict(ckpt_fproj)

    fproj_model.trajectory_latents = trajectory_latents
    stress_model.trajectory_latents = trajectory_latents

    fproj_model.eval()
    stress_model.eval()

    ## Load the specific trajectory object's config ##
    with open(os.path.join(traj_data_dir, "config.yaml")) as traj_cfg_f:
        traj_cfg = OmegaConf.load(traj_cfg_f)

    ti_mem_fraction = 0.7
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=ti_mem_fraction, debug=True)

    init_particles_simulator   = traj_data_orig[0].contiguous()
    target_particles_simulator = traj_data_orig.permute(1, 0, 2).cpu().numpy()

    cfg['objects'] = traj_cfg['objects']

    mpmwrapper_learnable = MPMWrapperLearnableStress(
        objects=cfg['objects'],
        simulator_cfg=cfg['simulator_cfg'],
        visualization_cfg=cfg['visualization_cfg'],
        init_particles=init_particles_simulator,
        target_particles=target_particles_simulator,
        constitutive_function=None,
        particles_ti_root=particles_ti_root,
        cuda_chunk_size=cuda_chunk_size,
        stress_model=stress_model,
        tb_writer=None,
        stress_data_gt=None,
        embed_dim=cfg['train_cfg']['embed_dim'],
        fproj_model=fproj_model,
    )

    mpmwrapper_learnable.initialize_particles()
    mpmwrapper_learnable.simulator_variables_initialize()

    # Pass material 0's latent to the Taichi-side latent field
    # (used by any Taichi kernels that still read traj_latent_ti directly)
    mpmwrapper_learnable.update_traj_latent_learnable(
        trajectory_latents[0].weight[0]
    )

    substep_pred = mpmwrapper_learnable.simulator.n_substeps[None]

    with torch.no_grad():
        for i in range(num_sim_steps):
            print(f"Sim step: {i}")

            for step in range(substep_pred * i, substep_pred * (i + 1)):
                local_index = step % cuda_chunk_size

                mpmwrapper_learnable.simulator.advance_F_stress(
                    t=local_index,
                    traj_id=traj_idx,
                    particle_mat_ids=particle_mat_ids,
                )

                if not mpmwrapper_learnable.simulator.cfl_satisfy[None]:
                    mpmwrapper_learnable.simulator.cached_states.clear()
                    print("CFL not satisfied",
                          f"Sim step: {i}, Step: {step}")
                    sys.exit()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    pred_x_all_steps_orig = mpmwrapper_learnable.simulator.x.to_numpy()
    pred_x_all_steps = pred_x_all_steps_orig[
        :, :(substep_pred * num_sim_steps + 1), :
    ].transpose(1, 0, 2)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    pts_all = pred_x_all_steps[:, :, :2]  # (T, P, 2)
    mat_ids_np = particle_mat_ids.cpu().numpy()  # (P,)

    # Split particles by material
    mask0 = mat_ids_np == 0
    mask1 = mat_ids_np == 1

    pts_mat0 = pred_x_all_steps[:, mask0, :2]  # (T, P0, 2)
    pts_mat1 = pred_x_all_steps[:, mask1, :2]  # (T, P1, 2)

    # Compute normalisation bounds per material
    def get_bounds(pts):
        xy_min = pts.min(axis=(0, 1))
        xy_max = pts.max(axis=(0, 1))
        return xy_min, xy_max

    min0, max0 = get_bounds(pts_mat0)
    min1, max1 = get_bounds(pts_mat1)

    # Add a small margin so particles aren't right at the edge
    margin = 0.05
    def normalise(pts, xy_min, xy_max, margin):
        pts_norm = (pts - xy_min) / (xy_max - xy_min + 1e-8)
        return pts_norm * (1 - 2 * margin) + margin

    gui0 = ti.GUI("Material 0", (800, 800))
    gui1 = ti.GUI("Material 1", (800, 800))

    for t in range(pred_x_all_steps.shape[0]):
        pts0_norm = normalise(pred_x_all_steps[t][mask0, :2], min0, max0, margin)
        pts1_norm = normalise(pred_x_all_steps[t][mask1, :2], min1, max1, margin)

        gui0.circles(pts0_norm, radius=2, color=0x4169E1)
        gui1.circles(pts1_norm, radius=2, color=0xFF6347)

        gui0.show()
        gui1.show()

    if cfg['train_cfg']['hou_vis']:
        visualize_simulation(trajectory=pred_x_all_steps)

if __name__ == '__main__':
    main()