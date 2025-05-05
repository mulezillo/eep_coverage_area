import os
import json
import statistics

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def scale_cov_mat(m: np.array, scale: float):
    # scale a covariance matrix
    # this can be used to convert an eep in meters (say) to cm. In that case, the scale would be 0.01.
    # will this same equation work for scaling up? not sure.
    # and include in paper, ect
    return m * scale


def load_profile(profile_path: os.PathLike):
    with open(profile_path, "r") as f:
        d = json.load(f)
        f.close()
    return d


def array_to_image(arr: np.array, save_name: os.PathLike):
    plt.clf()
    image = plt.imshow(arr, cmap="Oranges")
    cbar = plt.colorbar(image)
    cbar.set_label("Coverage Frequency per Cleaning")
    plt.title("Heat Map of Area Covered")
    plt.ylabel("cm")
    plt.xlabel("cm")
    plt.savefig(save_name)


def plot_eep_dist(xs: list, ys: list, save_name: os.PathLike = "eep_dist.png", scale: str = "m"):

    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    # plot axis lines
    ax.axhline(y=0, c="gray")
    ax.axvline(x=0, c="gray")
    ax.scatter(xs, ys, c="orange", label="Measured Error")
    ax.set_ylabel(f"Y Error ({scale}) (across track)")
    ax.set_xlabel(f"X Error ({scale}) (along track)")
    ax.set_title(f"EEP in Vehicle Frame per sqrt({scale}) Traveled")
    ax.set_aspect("equal")
    # make sure the x and y limits are the same to make the picture clearer. There's probably some matplotlib way to do
    # this but whatevs
    x_max = max([abs(x) for x in xs])
    y_max = max([abs(y) for y in ys])
    max_max = max([x_max, y_max]) + 1  # add some buffering to the plot so that the largest value isn't hidden
    plt.xlim(-max_max, max_max)
    plt.ylim(-max_max, max_max)

    center_x = statistics.mean(xs)
    center_y = statistics.mean(ys)

    # compute the covariance matrix
    cov_mat = np.cov([xs, ys])
    # compute eigenvalues and eigenvectors
    eigenval, eigenvec = np.linalg.eigh(cov_mat)

    # Sort eigenvalues and eigenvectors
    order = np.argsort(eigenval)[::-1]
    eigenvalues, eigenvectors = eigenval[order], eigenvec[:, order]

    k_values = {0.5: 1.1774, 0.95: 2.4477}
    dist_colors = ["blue", "darkcyan"]

    # plot a line for each k value
    for (conf, k), c in zip(k_values.items(), dist_colors):

        a = k * np.sqrt(eigenvalues[0])  # Semi-major axis
        b = k * np.sqrt(eigenvalues[1])  # Semi-minor axis

        # Compute orientation angle (radians to degrees)
        theta = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Draw error ellipse
        ellipse = Ellipse((center_x, center_y), width=2 * a, height=2 * b, angle=theta,
                          edgecolor=c, facecolor='none', linestyle='--', linewidth=1, label=f"EEP {int(conf * 100)}: ${round(a, 1)}$ {scale} x ${round(b, 1)}$ {scale}" )
        ax.add_patch(ellipse)
    ax.legend()
    plt.savefig(save_name)
    return cov_mat, center_x, center_y