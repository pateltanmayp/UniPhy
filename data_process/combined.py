import random
import os


current_path = os.getcwd()
local_path = '/'.join(current_path.strip().split('/')[:-1])

INTERACT = False

def falling_under_gravity(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        sand_x = 0.0 if INTERACT else 10.0
        os.system(
            f"python dataset_generate.py "
            f"objects=[blobby,sphere] "
            f"objects.blobby.geometry.num_particles=500 "
            f"objects.sphere.geometry.num_particles=500 "
            f"objects/blobby/material=elastic "
            f"objects/sphere/material=sand "
            f"visualization_cfg.num_frames=10 "
            f"objects.blobby.geometry.pos_y=0.2 "
            f"objects.sphere.geometry.pos_y=0.05 "
            f"objects.sphere.geometry.pos_x={sand_x} "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/combined_diverse/combined_falling_mu{random_mu[idx]}_lam{random_lam[idx]}_fa{random_friction_alpha[idx]}' "
            f"objects.blobby.material.mu={random_mu[idx]} "
            f"objects.blobby.material.lam={random_lam[idx]} "
            f"objects.sphere.material.mu={random_mu[idx]} "
            f"objects.sphere.material.lam={random_lam[idx]} "
            f"objects.sphere.material.friction_alpha={random_friction_alpha[idx]}"
        )

def horizontal_left(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        sand_x = 0.0 if INTERACT else 10.0
        os.system(
            f"python dataset_generate.py "
            f"objects=[blobby,sphere] "
            f"objects.blobby.geometry.num_particles=500 "
            f"objects.sphere.geometry.num_particles=500 "
            f"objects/blobby/material=elastic "
            f"objects/sphere/material=sand "
            f"visualization_cfg.num_frames=10 "
            f"objects.blobby.geometry.pos_y=0.13 "
            f"objects.blobby.geometry.pos_x=0.0 "
            f"objects.blobby.material.velocity='[2., 0., 0.]' "
            f"objects.sphere.geometry.pos_y=0.05 "
            f"objects.sphere.geometry.pos_x={sand_x} "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/combined_diverse/combined_horizontaleft_mu{random_mu[idx]}_lam{random_lam[idx]}_fa{random_friction_alpha[idx]}' "
            f"objects.blobby.material.mu={random_mu[idx]} "
            f"objects.blobby.material.lam={random_lam[idx]} "
            f"objects.sphere.material.mu={random_mu[idx]} "
            f"objects.sphere.material.lam={random_lam[idx]} "
            f"objects.sphere.material.friction_alpha={random_friction_alpha[idx]}"
        )

def horizontal_right(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        sand_x = 0.0 if INTERACT else 10.0
        os.system(
            f"python dataset_generate.py "
            f"objects=[blobby,sphere] "
            f"objects.blobby.geometry.num_particles=500 "
            f"objects.sphere.geometry.num_particles=500 "
            f"objects/blobby/material=elastic "
            f"objects/sphere/material=sand "
            f"visualization_cfg.num_frames=10 "
            f"objects.blobby.geometry.pos_y=0.13 "
            f"objects.blobby.geometry.pos_x=0.0 "
            f"objects.blobby.material.velocity='[-2., 0., 0.]' "
            f"objects.sphere.geometry.pos_y=0.05 "
            f"objects.sphere.geometry.pos_x={sand_x} "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/combined_diverse/combined_horizontalright_mu{random_mu[idx]}_lam{random_lam[idx]}_fa{random_friction_alpha[idx]}' "
            f"objects.blobby.material.mu={random_mu[idx]} "
            f"objects.blobby.material.lam={random_lam[idx]} "
            f"objects.sphere.material.mu={random_mu[idx]} "
            f"objects.sphere.material.lam={random_lam[idx]} "
            f"objects.sphere.material.friction_alpha={random_friction_alpha[idx]}"
        )

def diagonal_left(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        sand_x = 0.0 if INTERACT else 10.0
        os.system(
            f"python dataset_generate.py "
            f"objects=[blobby,sphere] "
            f"objects.blobby.geometry.num_particles=500 "
            f"objects.sphere.geometry.num_particles=500 "
            f"objects/blobby/material=elastic "
            f"objects/sphere/material=sand "
            f"visualization_cfg.num_frames=10 "
            f"objects.blobby.geometry.pos_y=0.2 "
            f"objects.blobby.geometry.pos_x=0.0 "
            f"objects.blobby.material.velocity='[2., 0., 0.]' "
            f"objects.sphere.geometry.pos_y=0.05 "
            f"objects.sphere.geometry.pos_x={sand_x} "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/combined_diverse/combined_diagonalleft_mu{random_mu[idx]}_lam{random_lam[idx]}_fa{random_friction_alpha[idx]}' "
            f"objects.blobby.material.mu={random_mu[idx]} "
            f"objects.blobby.material.lam={random_lam[idx]} "
            f"objects.sphere.material.mu={random_mu[idx]} "
            f"objects.sphere.material.lam={random_lam[idx]} "
            f"objects.sphere.material.friction_alpha={random_friction_alpha[idx]}"
        )


if __name__ == '__main__':
    min_range_mu, max_range_mu = 350.0, 2.6e6
    min_range_lam, max_range_lam = 500.0, 2.6e6
    min_range_fa, max_range_fa = 0.01, 0.4
    num_samples = 1

    random_mu = [random.uniform(min_range_mu, max_range_mu) for _ in range(num_samples)]
    random_lam = [random.uniform(min_range_lam, max_range_lam) for _ in range(num_samples)]
    random_fa = [random.uniform(min_range_fa, max_range_fa) for _ in range(num_samples)]

    falling_under_gravity(random_mu, random_lam, random_fa)
    horizontal_left(random_mu, random_lam, random_fa)
    horizontal_right(random_mu, random_lam, random_fa)
    diagonal_left(random_mu, random_lam, random_fa)
