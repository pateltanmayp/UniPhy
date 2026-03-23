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
import time
import random
import open3d as o3d
from PIL import Image
from houdini_visualization_gradio import visualize_simulation
from sklearn.cluster import KMeans

torch.autograd.set_detect_anomaly(True)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the random seed
seed = 42
set_random_seed(seed)


class MultiMaterialLatent(torch.nn.Module):
    """
    Holds one learnable latent embedding per material ID.

    For a scene with M distinct materials the module owns M independent
    nn.Embedding tables, each of size (1, embed_dim), initialised from
    the centroid of the corresponding KMeans cluster found in the
    pre-trained trajectory-latent bank.

    Args:
        device:             target device string, e.g. 'cuda'
        traj_latent_path:   path to a saved nn.Embedding whose weights
                            are the pre-trained latent bank
        embed_dim:          dimensionality of every latent vector
        num_materials:      number of distinct material IDs in the scene
        kmeans_n_clusters:  total clusters to fit; one centroid per
                            material is selected (clusters 0..M-1)
    """
    def __init__(self, device, traj_latent_path, embed_dim,
                 num_materials, kmeans_n_clusters=8):
        super(MultiMaterialLatent, self).__init__()
        self.num_materials = num_materials
        self.embed_dim = embed_dim

        print("KMeans loading for multi-material latent initialisation …")
        trajectory_latent_embedding_orig = (
            torch.load(traj_latent_path, weights_only=False).weight.detach()
        )
        kmeans = KMeans(n_clusters=kmeans_n_clusters, random_state=42)
        kmeans.fit(trajectory_latent_embedding_orig.cpu().numpy())

        # One embedding table per material; each holds a single vector so
        # the per-material latent is trajectory_latent.weight[0].
        self.trajectory_latents = nn.ModuleList()
        for mat_id in range(num_materials):
            # Cycle through available centroids if num_materials >
            # kmeans_n_clusters, so initialisation never crashes.
            centroid_idx = mat_id % kmeans_n_clusters
            centroid = torch.tensor(
                kmeans.cluster_centers_[centroid_idx],
                dtype=torch.float32
            ).unsqueeze(0)                          # (1, embed_dim)
            emb = nn.Embedding.from_pretrained(
                centroid, freeze=False
            ).to(device)
            self.trajectory_latents.append(emb)

        print(f"  Initialised {num_materials} material latents "
              f"(embed_dim={embed_dim}).")

    def get_latent(self, mat_id: int) -> torch.Tensor:
        """Return the (1, embed_dim) weight for a single material."""
        return self.trajectory_latents[mat_id].weight   # (1, embed_dim)

    def forward(self):
        pass


class FprojNN(torch.nn.Module):
    def __init__(self, activation, hidden_size, device, embed_dim,
                 trajectory_latents):
        super(FprojNN, self).__init__()
        hidden_size = hidden_size
        self.device = device

        if activation == "gelu":
            self.activation = torch.nn.GELU()

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)
        self.device = "cuda"

        self.fc1 = nn.Linear(27 + embed_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, 9, bias=True)

        # trajectory_latents is now a nn.ModuleList (or None before wiring)
        self.trajectory_latents = trajectory_latents

    def Ftmp_U_Vt_transform(self, Ftmp, U, V):
        if (len((torch.isnan(Ftmp) == True).nonzero()) > 0 or
                len((torch.isinf(Ftmp) == True).nonzero()) > 0):
            import ipdb; ipdb.set_trace()

        U_flatten   = self.flatten(U)                       # P x 9
        Vt_flatten  = self.flatten(V.transpose(1, 2))       # P x 9
        Ftmp_flatten = self.flatten(Ftmp)                   # P x 9
        Ftmp_input  = torch.cat(
            [Ftmp_flatten, U_flatten, Vt_flatten], dim=-1  # P x 27
        )
        return Ftmp_input

    def _build_latent_tensor(self, num_particles, particle_mat_ids):
        """
        Build a (P, embed_dim) tensor by gathering each particle's
        material latent according to particle_mat_ids (LongTensor of
        shape (P,) with values in [0, num_materials)).
        """
        parts = []
        for mat_id in range(len(self.trajectory_latents)):
            mask = (particle_mat_ids == mat_id)          # (P,) bool
            if not mask.any():
                continue
            n = mask.sum().item()
            latent = self.trajectory_latents[mat_id].weight  # (1, D)
            parts.append((mask, latent.expand(n, -1)))

        # Allocate output and scatter
        embed_dim = self.trajectory_latents[0].weight.shape[-1]
        out = torch.zeros(num_particles, embed_dim,
                          device=particle_mat_ids.device,
                          dtype=torch.float32)
        for mask, lat in parts:
            out[mask] = lat
        return out

    def forward(self, Ftmp, U, V, traj_id, particle_mat_ids=None):
        """
        Args:
            Ftmp, U, V:        as before - shape (P, 3, 3)
            traj_id:           kept for API compatibility (ignored when
                               particle_mat_ids is provided)
            particle_mat_ids:  LongTensor (P,) - material ID per particle.
                               When None, falls back to the single-material
                               behaviour using traj_id as the latent index.
        """
        Ftmp_flatten = self.Ftmp_U_Vt_transform(Ftmp, U, V)   # P x 27

        if particle_mat_ids is not None:
            latent_particles = self._build_latent_tensor(
                Ftmp.shape[0], particle_mat_ids
            )                                                  # P x D
        else:
            # Legacy single-material path
            latent_particles = (
                self.trajectory_latents[0].weight[traj_id]
                    .unsqueeze(0)
                    .repeat(Ftmp.shape[0], 1)
            )

        x = self.activation(
            self.fc1(torch.cat([Ftmp_flatten, latent_particles], dim=-1))
        )
        x   = self.activation(self.fc2(x))
        out = self.fc3(x)

        Fproj = Ftmp + out.view(out.shape[0], 3, 3)
        return Fproj


class StressNN(torch.nn.Module):
    def __init__(self, activation, hidden_size, embed_dim, device,
                 trajectory_latents):
        super(StressNN, self).__init__()
        hidden_size = hidden_size
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
        Ft  = F.transpose(1, 2)                                # B x 3 x 3
        FFt = torch.matmul(F, Ft)                              # B x 3 x 3

        J  = torch.max(torch.det(F[:, :, :]),
                       torch.Tensor([1e-6]).cuda())
        J1 = torch.max(F[:, 0, 0],
                       torch.Tensor([1e-6]).cuda())

        _, sigma, _ = torch.svd(F)                             # sigma B x 3
        FFt_flatten = self.flatten(FFt)                        # P x 9
        J  = J.unsqueeze(-1)                                   # P x 1
        J1 = J1.unsqueeze(-1)                                  # P x 1

        strain = torch.cat(
            [FFt_flatten, torch.log(J), sigma, J,
             torch.log(J1), J1], dim=-1                        # P x 16
        )
        return strain

    def _build_latent_tensor(self, num_particles, particle_mat_ids):
        """Same scatter logic as FprojNN._build_latent_tensor."""
        parts = []
        for mat_id in range(len(self.trajectory_latents)):
            mask = (particle_mat_ids == mat_id)
            if not mask.any():
                continue
            n = mask.sum().item()
            latent = self.trajectory_latents[mat_id].weight    # (1, D)
            parts.append((mask, latent.expand(n, -1)))

        embed_dim = self.trajectory_latents[0].weight.shape[-1]
        out = torch.zeros(num_particles, embed_dim,
                          device=particle_mat_ids.device,
                          dtype=torch.float32)
        for mask, lat in parts:
            out[mask] = lat
        return out

    def forward(self, F, C, traj_id, particle_mat_ids=None):
        """
        Args:
            F, C:              as before - shape (P, 3, 3)
            traj_id:           kept for API compatibility
            particle_mat_ids:  LongTensor (P,) - material ID per particle
        """
        strain = self.FFt_logJ_sigma_J_logJ1_J1_transform(F)  # P x 16

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
        x   = self.activation(
            self.fc1(torch.cat([strain, C_flatten, latent_particles], dim=-1))
        )
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


# ---------------------------------------------------------------------------
# Utility: load per-particle material ID assignments
# ---------------------------------------------------------------------------

def load_particle_mat_ids(traj_data_dir: str, num_particles: int,
                          device: str) -> torch.Tensor:
    """
    Load a LongTensor of shape (P,) assigning each particle to a material.

    Expected file: <traj_data_dir>/particle_mat_ids.pt
    Fallback: if the file is absent, all particles are assigned to material 0
    (i.e. the original single-material behaviour is preserved).
    """
    id_path = os.path.join(traj_data_dir, 'MaterialID.pt')
    if os.path.exists(id_path):
        mat_ids_raw = torch.load(id_path).long().to(device)
        assert mat_ids_raw.shape[0] == num_particles, (
            f"MaterialID.pt has {mat_ids_raw.shape[0]} entries but "
            f"the scene has {num_particles} particles."
        )
        # Remap arbitrary material IDs to contiguous 0..n-1
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


@hydra.main(config_path='configs', config_name='infer')
def main(cfg: omegaconf.DictConfig):

    ## Logging ##
    save_dir  = cfg['train_cfg']['save_dir']
    local_dir = cfg['train_cfg']['local_dir']
    logger    = logging.getLogger()

    device           = 'cuda'
    num_sim_steps    = cfg['visualization_cfg']['num_frames']
    cuda_chunk_size  = cfg['train_cfg']['cuda_chunk_size']
    particles_ti_root = cfg['train_cfg']['particles_ti_root']

    os.makedirs(f"{local_dir}/{save_dir}/logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{local_dir}/{save_dir}/logs")
    fh = logging.FileHandler(f"{local_dir}/{save_dir}/log.txt")
    with open(f"{local_dir}/{save_dir}/config.yaml", 'w') as cfg_file:
        OmegaConf.save(cfg, cfg_file)

    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ## Number of materials (can be set in config; defaults to 1) ##
    num_materials = cfg['train_cfg'].get('num_materials', 1)
    print(f"Scene has {num_materials} distinct material(s).")

    ## Load the model ##
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

    fproj_model.eval()
    stress_model.eval()

    # Load trajectory data
    traj_data_dir  = os.path.join(local_dir, cfg['train_cfg']['traj_data_dir'])
    traj_data_orig = torch.load(os.path.join(traj_data_dir, 'GtX.pt'))
    traj_data_orig = torch.tensor(traj_data_orig).to(device)  # T x P x 3

    traj_idx      = 0
    num_particles = traj_data_orig.shape[1]

    ## Load per-particle material IDs ##
    particle_mat_ids = load_particle_mat_ids(
        traj_data_dir, num_particles, device
    )
    # Infer num_materials from data if not explicitly given in config
    inferred_num_mats = len(particle_mat_ids.unique())
    if inferred_num_mats > num_materials:
        print(f"  Config num_materials={num_materials} < inferred "
              f"{inferred_num_mats}; using inferred value.")
        num_materials = inferred_num_mats

    ## Build multi-material latent module ##
    latent_obj = MultiMaterialLatent(
        device="cuda",
        traj_latent_path=os.path.join(local_dir, cfg['train_cfg']['traj_latent_path']),
        embed_dim=cfg['train_cfg']['embed_dim'],
        num_materials=num_materials,
    )
    latent_obj.train()

    # Wire latents into the network heads
    fproj_model.trajectory_latents  = latent_obj.trajectory_latents
    stress_model.trajectory_latents = latent_obj.trajectory_latents

    ## Load the specific trajectory object's config ##
    with open(os.path.join(traj_data_dir, "config.yaml")) as traj_cfg_f:
        traj_cfg = OmegaConf.load(traj_cfg_f)

    ## Optimizer and scheduler – optimise ALL material latents jointly ##
    all_latent_params = list(latent_obj.trajectory_latents.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": all_latent_params,
          "lr": cfg['train_cfg']['lr']}]
    )
    step_lr_step_size = cfg['train_cfg']['step_lr_step_size']
    scheduler1 = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_lr_step_size, gamma=0.9
    )

    ti_mem_fraction = 0.7
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=ti_mem_fraction, debug=True)

    total_epochs = cfg['train_cfg']['epochs']
    for epoch in range(total_epochs):

        # Load positions
        init_particles_simulator   = traj_data_orig[0].contiguous()
        target_particles_simulator = (
            traj_data_orig.permute(1, 0, 2).cpu().numpy()   # P x T x 3
        )

        # Assign the specific trajectory object's config
        cfg['objects'] = traj_cfg['objects']

        # Initialize MPMWrapper and simulator
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

        # Initialize simulator and wrapper variables
        mpmwrapper_learnable.initialize_particles()
        mpmwrapper_learnable.simulator_variables_initialize()

        # Pass the latent for traj_idx (mat 0 by convention for the wrapper)
        mpmwrapper_learnable.update_traj_latent_learnable(
            latent_obj.trajectory_latents[0].weight[0]
        )

        substep_pred = mpmwrapper_learnable.simulator.n_substeps[None]

        for i in range(num_sim_steps):
            print(f"Epoch: {epoch}, Sim step: {i}")

            for step in range(substep_pred * i, substep_pred * (i + 1)):
                local_index = step % cuda_chunk_size

                mpmwrapper_learnable.simulator.advance_F_stress(
                    t=local_index,
                    traj_id=traj_idx,
                    particle_mat_ids=particle_mat_ids,   # <-- multi-material
                )

                if not mpmwrapper_learnable.simulator.cfl_satisfy[None]:
                    mpmwrapper_learnable.simulator.cached_states.clear()
                    print("CFL not satisfied",
                          f"Sim step: {i}, Step: {step}")
                    sys.exit()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        pred_x_all_steps_orig = mpmwrapper_learnable.simulator.x.to_numpy()
        pred_x_all_steps = pred_x_all_steps_orig[
            :, :(substep_pred * num_sim_steps + 1), :
        ].transpose(1, 0, 2)

        mpmwrapper_learnable.simulator.loss.grad[None] = 1.
        mpmwrapper_learnable.compute_loss()
        print("Loss: ", mpmwrapper_learnable.simulator.loss)
        mpmwrapper_learnable.compute_loss.grad()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        for k in range(num_sim_steps):
            print("Backpropagation step: ", k)
            mpmwrapper_learnable.simulator.advance_grad_F_stress(
                num_sim_steps - k - 1,
                traj_id=traj_idx,
                particle_mat_ids=particle_mat_ids,       # <-- multi-material
            )

        optimizer.step()

        mpmwrapper_learnable.simulator.loss[None] = 0.
        mpmwrapper_learnable.clear_grads()

        scheduler1.step()

        # Log per-material latent norms for diagnostic purposes
        for mat_id in range(num_materials):
            lat_norm = (
                latent_obj.trajectory_latents[mat_id].weight
                           .norm().item()
            )
            writer.add_scalar(
                f"latent_norm/material_{mat_id}", lat_norm, epoch
            )
            print(f"  Material {mat_id} latent norm: {lat_norm:.4f}")

        ti.reset()
        ti.init(arch=ti.gpu, device_memory_fraction=ti_mem_fraction,
                debug=True)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if cfg['train_cfg']['hou_vis']:
            visualize_simulation(trajectory=pred_x_all_steps)


if __name__ == '__main__':
    main()