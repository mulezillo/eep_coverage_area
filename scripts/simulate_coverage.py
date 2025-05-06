import csv
import os
import time
import pathlib
import statistics

import numpy as np
import matplotlib.pyplot as plt

from nav_error_profile import EEP
from coverage import Coverage
import utils


def simulate_coverage(nep: EEP, area_width_m: int, area_height_m: int, sims: int) -> tuple[np.array, list]:
    """
    Perform a simulation to estimate coverage area
    Args:
        nep: an EEP nav error profile
        area_width_m: the width of the area in meters
        area_height_m: the height of the area in meters
        sims: the number of simulations to run

    Returns: normalized coverage map, list of history of coverage percentages
    """
    # build a heat map to track coverage at cm level
    heat_map = np.zeros((area_height_m * 100, area_width_m * 100), dtype=int)

    # also need a history of coverage percentages
    coverage_pct_history = []

    # run sim
    for n in range(sims):
        c = Coverage(nep, area_width_m, area_height_m)
        c_map = c.simulate()
        heat_map += c_map
        c_pct = c.calc_coverage()
        coverage_pct_history.append(c_pct)

    normalized_map = heat_map / sims
    return normalized_map, coverage_pct_history


def plot_coverage_dist(maps: list[np.array], labels: list, x_label: str, title: str, save_path: os.PathLike) -> None:
    """
    Helper function for creating a CDF of coverage frequency for each location in an array map
    Args:
        maps: list of maps
        labels: labels to use for each map
        x_label: the x label for the plot
        title: the title of the plot
        save_path: the path to save the plot to
    """
    plt.clf()
    for normalized_map, l in zip(maps, labels):
        # flatten the array to 1D (don't care about position any more)
        cov_freq = normalized_map.flatten()
        sorted_data = list(sorted(cov_freq))
        frequencies = [i / len(cov_freq) for i in range(len(cov_freq))]
        plt.plot(sorted_data, frequencies, label=l)
    plt.ylabel("Probability (est)")
    plt.xlabel(x_label)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)


def save_statistics(cov_dists: list[list], cov_maps: list[np.array], labels: list, save_name: str) -> None:
    """
    Helper function for saving the resulting statistics from a simulation suite to a file
    Args:
        cov_dists: list of lists for coverages
        cov_maps: list of coverage maps
        labels: labels to match each set in the above lists
        save_name: the name to give the resuling file
    """
    # get the mean coverage areas for each of the eeps and write to file
    mean_cov = [statistics.mean(c) for c in cov_dists]

    # let's also try getting the expect coverage per square
    mean_sq_cov = [np.mean(c.flatten()) for c in cov_maps]

    # save statistics
    out_file = pathlib.Path("../", "figures", save_name)
    rows_list = list(zip(labels, mean_cov, mean_sq_cov))
    with open(out_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Mean Coverage %", "Mean Coverage per square"])
        writer.writerows(rows_list)
        f.close()


if __name__ == "__main__":
    """
    Simulate coverage for 1000 simulations for 6 different EEP distributions and analyze the results
    
    Simulate coverage for 1000 simulations for 4 different map sizes with a single EEP distribution and analyze the
    results
    
    Simulate the time it takes to cover the entire map for 100 simulations and analyze the results.
    """

    # setup simulation constants
    # we're going to use cm precision, but eep is in m. so this is our conversion factor
    step_size = 0.01
    width_m = 10
    height_m = 10
    num_sims = 1000

    # build all the eep profile paths
    profiles = [
        pathlib.Path("..", "data", "measured_eep_bad.json"),
        # pathlib.Path("..", "data", "eep_0.05.json"),  # don't think we want/need this anymore
        pathlib.Path("..", "data", "measured_eep_good.json"),
        pathlib.Path("..", "data", "eep_0.01.json"),
        pathlib.Path("..", "data", "eep_0.005.json"),
        pathlib.Path("..", "data", "eep_0.001.json"),
        pathlib.Path("..", "data", "eep_0.00.json"),
    ]

    # keep track of all the coverage distributions
    coverage_dists = []
    coverage_maps = []
    curve_labels = [f"EEP {i}" for i in range(len(profiles))]
    start = time.perf_counter()
    for p in profiles:
        sim_start = time.perf_counter()
        cov = np.array(utils.load_profile(p)["covariance"])
        # scale the eep to cm precision
        cm_cov_mat = utils.scale_cov_mat(cov, step_size)
        nav_prof = EEP(cm_cov_mat)
        h_map, cov_pct_hist = simulate_coverage(nav_prof, width_m, height_m, num_sims)

        coverage_dists.append(cov_pct_hist)
        coverage_maps.append(h_map)
        # save heat map
        im_name = pathlib.Path("..", "figures", "heat_maps", f"{p.stem}.png")
        utils.array_to_image(h_map, save_name=im_name)
        print(f"{p.name} simulation took {(time.perf_counter() - sim_start) / 60} min")

    # now plot a cdf for all eeps
    cdf_save_path = pathlib.Path("../", "figures", "coverage_cdf.png")
    utils.plot_cdf_set(coverage_dists, curve_labels, "Coverage %", "CDF of Coverage %", cdf_save_path)

    cov_freq_cdf_path = pathlib.Path("../", "figures", "coverage_frequency_dist.png")
    plot_coverage_dist(coverage_maps, curve_labels, "Coverage Frequency per Cleaning for each Square cm", "CDF of Coverage Frequency", cov_freq_cdf_path)
    print(f"{num_sims} with {len(profiles)} profiles took {(time.perf_counter() - start) / 60} min")


    save_statistics(coverage_dists, coverage_maps, curve_labels, "statistics.csv")

    # now let's investigate what happens when we change the area we are trying to cover
    # let's choose 1 EEP to stick with for all these tests. Let's choose the best "real" one
    area_size_eep = pathlib.Path("..", "data", "measured_eep_good.json")
    cov = np.array(utils.load_profile(area_size_eep)["covariance"])
    # scale the eep to cm precision
    cm_cov_mat = utils.scale_cov_mat(cov, step_size)
    area_prof = EEP(cm_cov_mat)

    dim_coverage_dists = []
    dim_coverage_maps = []
    # we're going to stick with squares again, but use the following dimensions
    dimensions = [(5, 5), (8, 8), (10, 10), (15, 15)]
    start = time.perf_counter()
    for width, height in dimensions:
        sim_start = time.perf_counter()
        h_map, cov_pct_hist = simulate_coverage(area_prof, width, height, num_sims)

        dim_coverage_dists.append(cov_pct_hist)
        dim_coverage_maps.append(h_map)
        # save heat map
        im_name = pathlib.Path("..", "figures", "heat_maps", f"{width}_x_{height}.png")
        utils.array_to_image(h_map, save_name=im_name)
        print(f"{width}x{height} simulation took {(time.perf_counter() - sim_start) / 60} min")

    # now plot a cdf for all eeps
    dim_labels = [f"{w} x {h}" for w,h in dimensions]
    cdf_dim_save_path = pathlib.Path("../", "figures", "coverage_cdf_by_dim.png")
    utils.plot_cdf_set(dim_coverage_dists, dim_labels, "Coverage %", "CDF of Coverage %", cdf_dim_save_path)

    cov_dim_freq_cdf_path = pathlib.Path("../", "figures", "coverage_frequency_dist_by_dim.png")
    plot_coverage_dist(dim_coverage_maps, dim_labels, "Coverage Frequency per Cleaning for each Square cm", "CDF of Coverage Frequency", cov_dim_freq_cdf_path)
    print(f"{num_sims} with {len(profiles)} profiles took {(time.perf_counter() - start) / 60} min")

    # NOW MAKE A SHIT LOAD OF PLOTS!
    # PUTTING AT THE END TO PREVENT TIGHT LAYOUT ISSUES

    # plot all 6 heatmaps together in case we want that!
    # Warning: this a fancy, highly specific plot. That also means it's very sensitive to changes in the simulation
    # parameters. So it may need tuning depending on simulation results, etc
    plt.clf()
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6, 6))
    fig.suptitle("Coverage Area Heat Map for each EEP Distribution")
    plt.tight_layout()

    # normalize by the worst eep, which should be the first result in the list
    norm = plt.Normalize(vmin=min(coverage_maps[0].flatten()), vmax=max(coverage_maps[0].flatten()))
    # from worst to best
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(coverage_maps[i], cmap="Oranges", norm=norm)
        ax.set_title(f"EEP {i}")
        if i == 0:
            cbar = fig.colorbar(
                im,
                ax=axes,
                location='right',
                shrink=0.6,
                extend="max")
            cbar.set_label("Coverage Frequency per Cleaning")
    axes[0][0].set_ylabel("cm")
    axes[1][0].set_ylabel("cm")
    axes[2][0].set_ylabel("cm")
    axes[2][0].set_xlabel("cm")
    axes[2][1].set_xlabel("cm")

    plt.savefig("all_heatmaps.png", dpi=300)

    # plot all 4 heatmaps together too! Unfortunately some copy-paste going on here. just easier.
    # Warning: this a fancy, highly specific plot. That also means it's very sensitive to changes in the simulation
    # parameters. So it may need tuning depending on simulation results, etc
    plt.clf()
    fig, axes = plt.subplots(2,2, figsize=(6,6))
    fig.suptitle("Coverage Area Heat Map for each Area Size")
    plt.tight_layout()

    # normalize by the worst eep, which should be the first result in the list
    norm = plt.Normalize(vmin=min(dim_coverage_maps[-1].flatten()), vmax=max(dim_coverage_maps[-1].flatten()))
    # from worst to best
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(dim_coverage_maps[i], cmap="Oranges", norm=norm)
        ax.set_title(f"{dimensions[i][0]} by {dimensions[i][1]}")
        if i == 0:
            cbar = fig.colorbar(
                im,
                ax=axes,
                location='right',
                shrink=0.6,
                extend="max")
            cbar.set_label("Coverage Frequency per Cleaning")
    axes[0][0].set_ylabel("cm")
    axes[1][0].set_ylabel("cm")
    axes[1][0].set_xlabel("cm")
    axes[1][1].set_xlabel("cm")

    plt.savefig("all_heatmaps_by_dim.png", dpi=300)
    save_statistics(dim_coverage_dists, dim_coverage_maps, dim_labels, "dim_statistics.csv")