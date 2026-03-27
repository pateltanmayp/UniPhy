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
import matplotlib.pyplot as plt
from houdini_visualization_gradio import visualize_simulation

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

seed = 42
set_random_seed(seed)


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

        _, sigma, _ = torch.linalg.svd(Ftmp)
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
            self.fc1(torch.cat([Ftmp_flatten, latent_particles], dim=-1).double())
        )
        x   = self.activation(self.fc2(x))
        out = self.fc3(x)

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
        return torch.cat([K1, K2, K3], dim=1).double()

    def forward(self, F_flat, z):
        z = z.double()
        K = self.compute_invariants(F_flat)
        W_elastic      = self.elastic_nn(K)
        elastic_scale  = self.elastic_scale(z)
        W_elastic      = elastic_scale * W_elastic
        plastic_factor = self.plastic_gate(z)
        W_plastic      = plastic_factor * W_elastic.detach()
        weights = self.branch_weights(z)
        alpha   = weights / weights.sum(dim=1, keepdim=True)
        W = alpha[:, 0:1] * W_elastic + alpha[:, 1:2] * W_plastic

        create_graph = torch.is_grad_enabled()
        P_flat = torch.autograd.grad(
            W.sum(), F_flat,
            create_graph=create_graph,
            retain_graph=True,
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

        # enable_grad is needed because BranchingConstitutiveStress computes
        # stress as dW/dF via autograd.grad, which requires a graph even
        # during inference (which runs inside torch.no_grad())
        with torch.enable_grad():
            F_flat = self.flatten(F).float().requires_grad_(True)
            stress_symmetric = self.stress_model(F_flat, latent_particles)

        return stress_symmetric.detach() if not torch.is_grad_enabled() else stress_symmetric


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


@hydra.main(config_path='configs', config_name='sim_custom')
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

        # Fproj: remap fc1_fproj -> fc1, etc.
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
            short_key = key.replace('fproj_model.', '')
            for old, new in fproj_key_map.items():
                short_key = short_key.replace(old, new)
            ckpt_fproj[short_key] = value
        fproj_expected = set(fproj_model.state_dict().keys())
        ckpt_fproj = {k: v for k, v in ckpt_fproj.items() if k in fproj_expected}
        fproj_model.load_state_dict(ckpt_fproj, strict=True)

        # Stress: strip 'stress_model.' prefix
        ckpt_stress = {
            key.replace('stress_model.', ''): value
            for key, value in model_state.items() if 'stress' in key
        }
        stress_model.stress_model.load_state_dict(ckpt_stress, strict=True)

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

    np.save(os.path.join(local_dir, save_dir, "trajectory.npy"), pred_x_all_steps)
    np.save(os.path.join(local_dir, save_dir, "material_ids.npy"), particle_mat_ids.cpu().numpy())

    if cfg['train_cfg']['plot_errors']:
        gt_x = traj_data_orig.cpu().numpy()  # (T, P, 3)
        num_gt_frames = gt_x.shape[0]

        # Sample predicted positions at frame boundaries to match GT timesteps
        pred_at_frames = pred_x_all_steps  # (T, P, 3) approximately
        # Trim to match GT length
        min_frames = min(num_gt_frames, pred_at_frames.shape[0])
        gt_x        = gt_x[:min_frames]
        pred_at_frames = pred_at_frames[:min_frames]

        # Per-particle position error at each frame: (T, P, 3) -> mean over particles
        error = np.linalg.norm(pred_at_frames - gt_x, axis=-1)  # (T, P)
        mean_error_per_frame    = error.mean(axis=1)             # (T,)
        per_mat_mean_error      = {}
        mat_ids_np = particle_mat_ids.cpu().numpy()
        for mat_id in range(num_materials):
            mask = mat_ids_np == mat_id
            per_mat_mean_error[mat_id] = error[:, mask].mean(axis=1)  # (T,)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Mean Position Error over Time", fontsize=13)

        # Left: overall error
        axes[0].plot(mean_error_per_frame, color='steelblue', linewidth=1.5)
        axes[0].set_title("All particles")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Mean L2 error")
        axes[0].grid(True, alpha=0.3)

        # Right: per-material error
        colors = ['royalblue', 'tomato', 'green', 'purple']
        for mat_id, err in per_mat_mean_error.items():
            axes[1].plot(err, label=f"Material {mat_id}",
                        color=colors[mat_id % len(colors)], linewidth=1.5)
        axes[1].set_title("Per material")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Mean L2 error")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(local_dir, save_dir, "position_error.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"Saved error plot to {plot_path}")

        # Print summary stats
        print(f"\nPosition error summary (all particles):")
        print(f"  Mean over all frames: {mean_error_per_frame.mean():.6f}")
        print(f"  Max over all frames:  {mean_error_per_frame.max():.6f}")
        print(f"  Final frame error:    {mean_error_per_frame[-1]:.6f}")
        for mat_id, err in per_mat_mean_error.items():
            print(f"  Material {mat_id} mean: {err.mean():.6f}, final: {err[-1]:.6f}")

    if cfg['train_cfg']['sim_materials_separately']:
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

    if cfg['train_cfg']['sim_materials_together']:
        gui = ti.GUI("Simulation", (800, 800))
        for t in range(pred_x_all_steps.shape[0]):
            pts = pred_x_all_steps[t][:, :2]
            gui.circles(pts, radius=1)
            gui.show()

    if cfg['train_cfg']['hou_vis']:
        visualize_simulation(trajectory=pred_x_all_steps)

if __name__ == '__main__':
    main()