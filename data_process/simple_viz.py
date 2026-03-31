import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Relative path to trajectory folder inside dataset')
    args = parser.parse_args()

    current_path = os.getcwd()
    local_path = '/'.join(current_path.strip().split('/')[:-1])
    path_to_traj = os.path.join(local_path, 'dataset', args.path)

    x_gt = torch.load(f"{path_to_traj}/GtX.pt")
    material_id = torch.load(f"{path_to_traj}/MaterialID.pt")

    num_frames, num_particles, _ = x_gt.shape

    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']  # Colour map for diff materials
    particle_colors = [colors[i % len(colors)] for i in material_id]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        x_gt[0,:,0].cpu().numpy(),
        x_gt[0,:,1].cpu().numpy(),
        x_gt[0,:,2].cpu().numpy(),
        c=particle_colors,
        s=5
    )

    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([0, 0.4])
    ax.set_zlim([-0.2, 0.2])

    def update(frame):
        scatter._offsets3d = (
            x_gt[frame,:,0].cpu().numpy(),
            x_gt[frame,:,1].cpu().numpy(),
            x_gt[frame,:,2].cpu().numpy()
        )
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)
    plt.show()


if __name__ == "__main__":
    main()
