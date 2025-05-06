import os
import json
import statistics

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def scale_cov_mat(m: np.array, scale: float) -> np.array:
    """
    Helper function for scaling a covariance matrix. Basically just multiples it by a constant, which is obviously
    pretty trivial. But when I made this I was playing around with other ways to scale, so it was nice to have
    consistent function to call somewhere. Decided to just leave it.
    Args:
        m: the matrix to scale
        scale: the scale factor

    Returns: the scaled matrix
    """
    # scale a covariance matrix
    # this can be used to convert an eep in meters (say) to cm. In that case, the scale would be 0.01.
    return m * scale


def load_profile(profile_path: os.PathLike) -> dict:
    """
    Load a dictionary from a json file
    Args:
        profile_path: path to the json file

    Returns: dictionary of data from the json file
    """
    with open(profile_path, "r") as f:
        d = json.load(f)
        f.close()
    return d


def array_to_image(arr: np.array, save_name: os.PathLike) -> None:
    """
    Writes a numpy array to an image
    Args:
        arr: the array to save as an image
        save_name: the filename to save as
    """
    plt.clf()
    image = plt.imshow(arr, cmap="Oranges")
    cbar = plt.colorbar(image)
    cbar.set_label("Coverage Frequency per Cleaning")
    plt.title("Heat Map of Area Covered")
    plt.ylabel("cm")
    plt.xlabel("cm")
    plt.savefig(save_name)


def plot_eep_dist(xs: list, ys: list, save_name: os.PathLike = "eep_dist.png", scale: str = "m") -> tuple[list, float, float]:
    """
    Helper function for plotting and estimating an EEP distribution.
    Args:
        xs: errors in the x direction
        ys: errors in the y direction
        save_name: the name to save the plot as
        scale: the scale to use for the plot labels

    Returns: covariance matrix defining EEP distribution, center x and center y of the distribution
    """
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    # plot axis lines
    ax.axhline(y=0, c="gray")
    ax.axvline(x=0, c="gray")
    ax.scatter(xs, ys, c="orange", label="Measured Error")
    ax.set_ylabel(f"Y Error ({scale}) (across track)")
    ax.set_xlabel(f"X Error ({scale}) (along track)")
    ax.set_title(f"EEP in Vehicle Frame per {scale} Traveled")
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


def plot_cdf_set(data: list[list], labels: list, x_label: str, title: str, save_path: os.PathLike) -> None:
    """
    Plot a set of CDF in a single plot.

    Args:
        data: list of lists of data to plot
        labels: list of labels to use for each curve
        x_label: x label of the plot
        title: title of the plot
        save_path: path to save the plot to
    """
    # data and labels must be the same length!
    if len(data) != len(labels):
        raise Exception("Must have the same number of labels as distributions!")

    plt.clf()
    for d, l in zip(data, labels):
        sorted_data = list(sorted(d))
        frequencies = [i / len(d) for i in range(len(d))]
        plt.plot(sorted_data, frequencies, label=l)
    plt.ylabel("Probability (est)")
    plt.xlabel(x_label)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
