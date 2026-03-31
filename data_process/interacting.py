import faulthandler
faulthandler.enable()

import random
import os


current_path = os.getcwd()
local_path = '/'.join(current_path.strip().split('/')[:-1])

INTERACT = True

def fabric_on_sphere(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        os.system(
            f"python dataset_generate.py "
            f"objects=[box,sphere] "
            f"objects.box.geometry.num_particles=500 "
            f"objects.sphere.geometry.num_particles=500 "
            f"objects/box/material=elastic "
            f"objects/sphere/material=plasticine "
            f"visualization_cfg.num_frames=10 "
            f"objects.box.geometry.pos_y=0.2 "
            f"objects.box.geometry.pos_x=0.0 "
            f"objects.box.geometry.pos_z=0.0 "
            f"objects.box.geometry.rot_y=0.0 "
            f"objects.box.geometry.rot_x=0.0 "
            f"objects.box.geometry.sdf_params='[0.3, 0.01, 0.3]' "
            f"objects.box.material.velocity='[0., -2., 0.]' "
            f"objects.sphere.geometry.pos_y=0.025 "
            f"objects.sphere.geometry.pos_x={0.0} "
            f"objects.sphere.geometry.pos_z={0.0} "
            f"objects.sphere.geometry.rot_x={1.57} "
            f"objects.sphere.material.velocity='[0., 0., 0.]' "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/interacting_diverse/fabric_on_sphere' "
            f"objects.box.material.mu={1e3} "
            f"objects.box.material.lam={1e3} "
            f"objects.sphere.material.mu={1e6} "
            f"objects.sphere.material.lam={1e6} "
            f"objects.sphere.material.yield_stress={1e6} "
        )

def rigid_splashing_in_sand(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        os.system(
            f"python dataset_generate.py "
            f"objects=[box,rounded_cylinder] "
            f"objects.box.geometry.num_particles=500 "
            f"objects.rounded_cylinder.geometry.num_particles=500 "
            f"objects/box/material=elastic "
            f"objects/rounded_cylinder/material=sand "
            f"visualization_cfg.num_frames=10 "
            f"objects.box.geometry.pos_y=0.2 "
            f"objects.box.geometry.pos_x=0.0 "
            f"objects.box.geometry.pos_z=0.0 "
            f"objects.box.geometry.rot_y=45.0 "
            f"objects.box.geometry.rot_x=45.0 "
            f"objects.box.geometry.sdf_params='[0.1, 0.1, 0.1]' "
            f"objects.box.material.velocity='[0., -2., 0.]' "
            f"objects.rounded_cylinder.geometry.pos_y=0.025 "
            f"objects.rounded_cylinder.geometry.pos_x={0.0} "
            f"objects.rounded_cylinder.geometry.pos_z={0.0} "
            f"objects.rounded_cylinder.geometry.rot_x={1.57} "
            f"objects.rounded_cylinder.geometry.sdf_params='[0.1, 0.05, 0.1]' "
            f"objects.rounded_cylinder.geometry.sdf_params='[1.0, 0.1, 1.0]' "
            f"objects.rounded_cylinder.material.velocity='[0., 0., 0.]' "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/interacting_diverse/rigid_splashing_in_sand' "
            f"objects.box.material.mu={1e6} "
            f"objects.box.material.lam={1e6} "
            f"objects.rounded_cylinder.material.mu={2e3} "
            f"objects.rounded_cylinder.material.lam={2e3} "
            f"objects.rounded_cylinder.material.friction_alpha={0.01} "
        )

def raining_sand_v2(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        os.system(
            f"python dataset_generate.py "
            f"objects=[box,ellipsoid] "
            f"objects.box.geometry.num_particles=500 "
            f"objects.ellipsoid.geometry.num_particles=500 "
            f"objects/box/material=plasticine "
            f"objects/ellipsoid/material=sand "
            f"visualization_cfg.num_frames=10 "
            f"objects.box.geometry.pos_y=0.04 "
            f"objects.box.geometry.pos_x=0.0 "
            f"objects.box.geometry.pos_z=0.0 "
            f"objects.box.geometry.sdf_params='[0.05, 0.05, 0.05]' "
            f"objects.box.material.velocity='[0., 0., 0.]' "
            f"objects.ellipsoid.geometry.pos_y=0.15 "
            f"objects.ellipsoid.geometry.pos_x={0.0} "
            f"objects.ellipsoid.geometry.rot_z={0.0} "
            f"objects.ellipsoid.geometry.sdf_params='[0.7, 0.1, 0.7]' "
            f"objects.ellipsoid.material.velocity='[0., -2., 0.]' "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/interacting_diverse/raining_sand_v2' "
            f"objects.box.material.mu={1e6} "
            f"objects.box.material.lam={1e6} "
            f"objects.box.material.yield_stress={7500} "
            f"objects.ellipsoid.material.mu={2e2} "
            f"objects.ellipsoid.material.lam={2e2} "
            f"objects.ellipsoid.material.friction_alpha={0.01} "
        )

def horizontal_collision(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        os.system(
            f"python dataset_generate.py "
            f"objects=[rounded_box,capped_cone] "
            f"objects.rounded_box.geometry.num_particles=500 "
            f"objects.capped_cone.geometry.num_particles=500 "
            f"objects/rounded_box/material=elastic "
            f"objects/capped_cone/material=plasticine "
            f"visualization_cfg.num_frames=10 "
            f"objects.rounded_box.geometry.pos_y=0.13 "
            f"objects.rounded_box.geometry.pos_x=0.0 "
            f"objects.rounded_box.geometry.pos_z=0.0 "
            f"objects.rounded_box.material.velocity='[2., 0., 0.]' "
            f"objects.capped_cone.geometry.pos_y=0.05 "
            f"objects.capped_cone.geometry.pos_x={0.3} "
            f"objects.capped_cone.material.velocity='[-2., 0., 0.]' "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/interacting_diverse/horizontal_collision' "
            f"objects.rounded_box.material.mu={random_mu[idx]} "
            f"objects.rounded_box.material.lam={random_lam[idx]} "
            f"objects.capped_cone.material.mu={500000} "
            f"objects.capped_cone.material.lam={1000000} "
            f"objects.capped_cone.material.yield_stress={7500}"
        )

def horizontal_right(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        sand_x = 0.3 if INTERACT else 10.0
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
            f"objects.sphere.material.velocity='[-2., 0., 0.]' "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/interacting_diverse/diverse_horizontalright_mu{random_mu[idx]}_lam{random_lam[idx]}_fa{random_friction_alpha[idx]}' "
            f"objects.blobby.material.mu={random_mu[idx]} "
            f"objects.blobby.material.lam={random_lam[idx]} "
            f"objects.sphere.material.mu={random_mu[idx]} "
            f"objects.sphere.material.lam={random_lam[idx]} "
            f"objects.sphere.material.friction_alpha={random_friction_alpha[idx]}"
        )

def elastic_splashing_in_sand(random_mu, random_lam, random_friction_alpha):
    for idx in range(len(random_mu)):
        sand_x = 0.3 if INTERACT else 10.0
        os.system(
            f"python dataset_generate.py "
            f"objects=[blobby,ellipsoid] "
            f"objects.blobby.geometry.num_particles=500 "
            f"objects.ellipsoid.geometry.num_particles=500 "
            f"objects/blobby/material=elastic "
            f"objects/ellipsoid/material=sand "
            f"visualization_cfg.num_frames=10 "
            f"objects.blobby.material.velocity='[0., -2.0, 0.]' "
            f"objects.blobby.geometry.pos_z=0.0 "
            f"objects.blobby.geometry.pos_y=0.2 "
            f"objects.blobby.geometry.pos_x=0.0 "
            f"objects.ellipsoid.geometry.pos_z=0.0 "
            f"objects.ellipsoid.geometry.pos_y=0.0 "
            f"objects.ellipsoid.geometry.pos_x=0.0 "
            f"train_cfg.cuda_chunk_size=4096 "
            f"train_cfg.particles_ti_root=1024 "
            f"train_cfg.local_dir={local_path} "
            f"train_cfg.save_dir='dataset/interacting_diverse/elastic_splashing_in_sand' "
            f"objects.blobby.material.mu={random_mu[idx]} "
            f"objects.blobby.material.lam={random_lam[idx]} "
            f"objects.ellipsoid.material.mu={random_mu[idx]} "
            f"objects.ellipsoid.material.lam={random_lam[idx]} "
            f"objects.ellipsoid.material.friction_alpha={random_friction_alpha[idx]}"
        )


if __name__ == '__main__':
    min_range_mu, max_range_mu = 350.0, 2.6e6
    min_range_lam, max_range_lam = 500.0, 2.6e6
    min_range_fa, max_range_fa = 0.01, 0.4
    num_samples = 1

    random_mu = [random.uniform(min_range_mu, max_range_mu) for _ in range(num_samples)]
    random_lam = [random.uniform(min_range_lam, max_range_lam) for _ in range(num_samples)]
    random_fa = [random.uniform(min_range_fa, max_range_fa) for _ in range(num_samples)]

    # fabric_on_sphere(random_mu, random_lam, random_fa)
    # rigid_splashing_in_sand(random_mu, random_lam, random_fa)
    raining_sand_v2(random_mu, random_lam, random_fa)
    # horizontal_collision(random_mu, random_lam, random_fa)
    # horizontal_right(random_mu, random_lam, random_fa)
    # elastic_splashing_in_sand(random_mu, random_lam, random_fa)
