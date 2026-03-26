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

try:
    sys.path.insert(0, "/home/tanmay/thesis/ICKANs/")
    from core import *
    import drivers.config as c
    from ickan import *
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    print("Warning: KAN not available, falling back to MLP stress model.")

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

        # TODO: remove extra +3 AND extra layers
        self.fc1 = nn.Linear(27 + 3 + embed_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc5 = nn.Linear(hidden_size, 9, bias=True)

        # trajectory_latents is now a nn.ModuleList (or None before wiring)
        self.trajectory_latents = trajectory_latents

    # TODO: remove sigma stuff
    def Ftmp_U_Vt_transform(self, Ftmp, U, V):
        if (len((torch.isnan(Ftmp) == True).nonzero()) > 0 or
                len((torch.isinf(Ftmp) == True).nonzero()) > 0):
            import ipdb; ipdb.set_trace()

        _, sigma, _ = torch.linalg.svd(Ftmp)
        U_flatten   = self.flatten(U)                       # P x 9
        Vt_flatten  = self.flatten(V.transpose(1, 2))       # P x 9
        Ftmp_flatten = self.flatten(Ftmp)                   # P x 9
        Ftmp_input  = torch.cat(
            [Ftmp_flatten, U_flatten, sigma, Vt_flatten], dim=-1  # P x 27
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

        # TODO: remove extra layers
        x = self.activation(
            self.fc1(torch.cat([Ftmp_flatten, latent_particles], dim=-1))
        )
        x   = self.activation(self.fc2(x))
        x   = self.activation(self.fc3(x))
        x   = self.activation(self.fc4(x))
        out = self.fc5(x)

        Fproj = Ftmp + out.view(out.shape[0], 3, 3)
        return Fproj


class MaterialHyperNet(nn.Module):
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
            nn.Softplus()
        )

    def forward(self, z):
        return self.net(z)


class BranchingConstitutiveStress(nn.Module):
    def __init__(self, hidden_size, embed_dim, n_hidden=None,
                 grid_range=None, use_kan=True, seed=0):
        super().__init__()
        self.use_kan = use_kan
        self.embed_dim = embed_dim

        if use_kan:
            assert KAN_AVAILABLE, "KAN requested but not importable."
            assert n_hidden is not None and grid_range is not None
            self.elastic_nn = KAN(
                width=n_hidden, grid=c.grid, k=c.spline_order,
                seed=seed, device='cuda', base_fun='zero', grid_eps=1.0,
                grid_range_0=grid_range, sp_trainable=c.sp_trainable,
                sb_trainable=c.sb_trainable,
                symbolic_enabled=c.symbolic_enabled,
                auto_save=False
            )
        else:
            self.elastic_nn = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1),
                nn.Softplus()
            )

        self.elastic_scale = MaterialHyperNet(embed_dim, out_dim=1)
        self.plastic_gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.branch_weights = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.Softplus()
        )

    def compute_invariants(self, F_flat):
        F_flat = torch.clamp(F_flat, min=-5.0, max=5.0)
        F00 = F_flat[:, 0:1]
        F01 = F_flat[:, 1:2]
        F10 = F_flat[:, 3:4]
        F11 = F_flat[:, 4:5]
        C00 = F00**2 + F10**2
        C01 = F00*F01 + F10*F11
        C11 = F01**2 + F11**2
        I1 = C00 + C11 + 1.0
        I3 = C00*C11 - C01**2
        I3_safe = torch.clamp(I3, min=1e-6)
        J  = torch.sqrt(I3_safe)
        K1 = I1 * torch.pow(I3_safe, -1.0/3.0) - 3.0
        K2 = torch.pow(
            (I1 + I3_safe - 1.0) * torch.pow(I3_safe, -2.0/3.0), 1.5
        ) - 3.0 * (3.0 ** 0.5)
        K3 = (J - 1.0)**2
        return torch.cat([K1, K2, K3], dim=1) #.double()

    def forward(self, F_flat, z):
        K = self.compute_invariants(F_flat)
        W_elastic     = self.elastic_nn(K)
        elastic_scale = self.elastic_scale(z)
        W_elastic     = elastic_scale * W_elastic
        plastic_factor = self.plastic_gate(z)
        W_plastic      = plastic_factor * W_elastic.detach()
        weights = self.branch_weights(z)
        alpha   = weights / weights.sum(dim=1, keepdim=True)
        W = alpha[:, 0:1] * W_elastic + alpha[:, 1:2] * W_plastic

        create_graph = torch.is_grad_enabled()
        P_flat = torch.autograd.grad(
            W.sum(), F_flat,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]

        stress = P_flat.view(-1, 3, 3)
        stress_symmetric = 0.5 * (stress + stress.permute(0, 2, 1))
        stress_symmetric = torch.clamp(stress_symmetric, -1e4, 1e4)
        return stress_symmetric

class StressNN(nn.Module):
    """
    Drop-in replacement for the old StressNN.
    Wraps BranchingConstitutiveStress and exposes the same interface
    (including trajectory_latents and _build_latent_tensor) so the
    rest of the infer script needs no changes.
    """
    def __init__(self, hidden_size, embed_dim, device,
                 trajectory_latents, use_kan=False,
                 n_hidden=None, grid_range=None, seed=0):
        super().__init__()
        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)
        self.trajectory_latents = trajectory_latents
        self.stress_model = BranchingConstitutiveStress(
            hidden_size=hidden_size,
            embed_dim=embed_dim,
            n_hidden=n_hidden,
            grid_range=grid_range,
            use_kan=use_kan,
            seed=seed,
        )

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

        F_flat = self.flatten(F).float()
        F_flat_for_kan = F_flat.detach().requires_grad_(True)

        stress_symmetric = self.stress_model(F_flat_for_kan, latent_particles)

        if F.requires_grad:
            dummy = (F_flat * F_flat_for_kan.detach()).sum()
            stress_symmetric = stress_symmetric + 0.0 * F_flat.sum()

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


@hydra.main(config_path='configs', config_name='infer_custom')
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
    use_kan    = cfg['train_cfg'].get('use_kan', False)
    n_hidden   = OmegaConf.to_container(cfg['train_cfg']['n_hidden'], resolve=True) if use_kan else None
    grid_range = OmegaConf.to_container(cfg['train_cfg']['grid_range'], resolve=True) if use_kan else None

    fproj_model = FprojNN(
        activation=cfg['train_cfg']['nn_activation'],
        hidden_size=cfg['train_cfg']['hidden_size'],
        device='cuda',
        embed_dim=cfg['train_cfg']['embed_dim'],
        trajectory_latents=None,
    ).to(device)

    stress_model = StressNN(
        hidden_size=cfg['train_cfg']['hidden_size'],
        embed_dim=cfg['train_cfg']['embed_dim'],
        device='cuda',
        trajectory_latents=None,
        use_kan=use_kan,
        n_hidden=n_hidden,
        grid_range=grid_range,
        seed=0,
    ).to(device)

    if cfg['train_cfg']['load_model']:
        ckpt = torch.load(
            os.path.join(local_dir, cfg['train_cfg']['load_model']),
            map_location=device
        )
        model_state = ckpt['model_state_dict']

        fproj_key_map = {
            'fc1_fproj': 'fc1',
            'fc2_fproj': 'fc2',
            'fc3_fproj': 'fc3',
            'fc4_fproj': 'fc4',
            'fc5_fproj': 'fc5',
        }
        ckpt_fproj = {}
        for key, value in model_state.items():
            if 'fproj' not in key:
                continue
            # Strip 'fproj_model.' prefix
            short_key = key.replace('fproj_model.', '')
            # Remap layer name e.g. 'fc1_fproj.weight' -> 'fc1.weight'
            for old, new in fproj_key_map.items():
                short_key = short_key.replace(old, new)
            ckpt_fproj[short_key] = value

        # Check what FprojNN actually expects and filter to only those keys
        fproj_expected = set(fproj_model.state_dict().keys())
        ckpt_fproj = {k: v for k, v in ckpt_fproj.items() if k in fproj_expected}
        fproj_model.load_state_dict(ckpt_fproj, strict=True)

        # Stress: strip 'stress_model.' prefix
        ckpt_stress = {
            key.replace('stress_model.', ''): value
            for key, value in model_state.items() if 'stress' in key
        }
        stress_model.stress_model.load_state_dict(ckpt_stress, strict=True)

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

    # Load positions
    init_particles_simulator = traj_data_orig[0].contiguous()
    target_particles_simulator = (
        traj_data_orig.permute(1, 0, 2).cpu().numpy()   # P x T x 3
    )

    total_epochs = cfg['train_cfg']['epochs']
    for epoch in range(total_epochs):

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
        optimizer.zero_grad()

        latent_save_dir = os.path.join(local_dir, save_dir, "latents_multi_material")
        os.makedirs(latent_save_dir, exist_ok=True)
        torch.save(
            {f"material_{mat_id}": latent_obj.trajectory_latents[mat_id]
            for mat_id in range(num_materials)},
            os.path.join(latent_save_dir, f"latent_epoch_{epoch}.pt")
        )

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

        del mpmwrapper_learnable
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        import gc; gc.collect()
        ti.reset()
        ti.init(arch=ti.gpu, device_memory_fraction=ti_mem_fraction,
                debug=True)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if cfg['train_cfg']['hou_vis']:
            visualize_simulation(trajectory=pred_x_all_steps)


if __name__ == '__main__':
    main()