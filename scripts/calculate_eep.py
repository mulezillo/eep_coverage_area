import os
import csv
import json
import random
import pathlib

import numpy as np

import utils
from nav_error_profile import NavErrorProfile


def load_data(dpath: os.PathLike):
    with open(dpath, "r") as f:
        data = csv.reader(f)
        out_data = [row for row in data]
        f.close()
    return out_data


def clean_data(data: list):
    # convert all values from strings to floats
    # remove the header - don't care about that
    out_data = []
    for row in data[1:]:
        out_data.append([float(i) for i in row])
    return out_data


def scale_errors_by_distance(x_errors: list, y_errors: list, distances: list):

    # assume that the expected error magnitude scales like a random walk,
    # expected displacement = num steps ^ (1/2)

    # if we treat each cm traveled as a step, then we can divide the measured error by the distance squared
    scaled_x = []
    scaled_y = []
    for x, y, d in zip(x_errors, y_errors, distances):
        sx = x / d ** (1/2)
        sy = y / d ** (1/2)
        scaled_x.append(sx)
        scaled_y.append(sy)
    return scaled_x, scaled_y


def save_eep(cov: np.array, cx: float, cy: float, save_path: os.PathLike):
    d = {
        "covariance": cov.tolist(),
        "center_x": cx,
        "center_y": cy
    }
    with open(save_path, "w") as f:
        json.dump(d, f, indent=4)
        f.close()


def test_eep(nep: NavErrorProfile, scale: float = 1.0, sim_length: int = 1000, sample_size: int = 100):

    x_true = 0
    y_true = 0
    x_cmd = 0
    y_cmd = 0
    x_history = [x_true]
    y_history = [y_true]
    x_cmd_hist = [x_cmd]
    y_cmd_hist = [y_cmd]

    # number of steps affected by scale
    for i in range(int(sim_length / scale)):

        ex, ey = nep.sample(scale)
        # walk in x, but not in y
        x_true += ex + 1
        y_true += ey
        x_cmd += 1

        x_history.append(x_true)
        y_history.append(y_true)
        x_cmd_hist.append(x_cmd)
        y_cmd_hist.append(y_cmd)

    # now randomly sample indexes from the history, excluding first index where there is never any error
    indexes = list(sorted(random.sample(list(range(1, sim_length)), k=sample_size)))
    x_errs_cum = []
    y_errs_cum = []
    dists_cum = []
    for i in indexes:
        x_e = x_history[i] - x_cmd_hist[i]
        y_e = y_history[i] - y_cmd_hist[i]
        x_errs_cum.append(x_e)
        y_errs_cum.append(y_e)
        dists_cum.append(i)

    # now convert all errors and distances to relative
    x_errs_rel = []
    y_errs_rel = []
    dist_rel = []
    # need to handle the first measurement individually
    x_errs_rel.append(x_errs_cum[0])
    y_errs_rel.append(y_errs_cum[0])
    dist_rel.append(dists_cum[0])
    # now calc all others relative
    for j, (x, y, d) in enumerate(zip(x_errs_cum[1:], y_errs_cum[1:], dists_cum[1:])):
        # don't need to subtract 1 from j because it's already shifted over 1
        # and scale as needed
        rel_x = (x - x_errs_cum[j]) * scale
        rel_y = (y - y_errs_cum[j]) * scale
        d_rel = (d - dists_cum[j]) * scale
        x_errs_rel.append(rel_x)
        y_errs_rel.append(rel_y)
        dist_rel.append(d_rel)
    return x_errs_rel, y_errs_rel, dist_rel


def compare_cm_to_m(nep: NavErrorProfile, name: str, num_samples: int = 1000):
    m_to_cm = 0.01
    # now create trajectories of 1 m so that we can compare to sampling 1 m directly
    cm_x_errors = []
    cm_y_errors = []
    for i in range(num_samples):

        cm_x_error = 0
        cm_y_error = 0
        # generate 100 cm points to simulate traveling 1 m
        for t in range(int(1 / m_to_cm)):
            x, y = nep.sample(m_to_cm)
            cm_x_error += x
            cm_y_error += y

        cm_x_errors.append(cm_x_error)
        cm_y_errors.append(cm_y_error)
    save_name = pathlib.Path("..", "figures", name + "_cm_to_m.png")
    utils.plot_eep_dist(cm_x_errors, cm_y_errors, save_name)

    # now repeat but at meter level
    m_x_errors = []
    m_y_errors = []
    for i in range(num_samples):
        x, y = nep.sample()
        m_x_errors.append(x)
        m_y_errors.append(y)

    m_save_name = pathlib.Path("..", "figures", name + "_m_to_m.png")
    utils.plot_eep_dist(m_x_errors, m_y_errors, m_save_name)


if __name__ == "__main__":

    # load the 3 datasets:
    # 1. bad dvl
    # 2. good dvl
    # 3. combined
    input_paths = [
        pathlib.Path("../data/relative_error_data_bad.csv"),
        pathlib.Path("../data/relative_error_data_good.csv")
    ]
    save_names = [
        "measured_eep_bad",
        "measured_eep_good"
    ]
    for p, s in zip(input_paths, save_names):

        # load and clean data
        error_data = load_data(p)
        cleaned_data = clean_data(error_data)

        # extract columns
        x_err = [i[0] for i in cleaned_data]
        y_err = [i[1] for i in cleaned_data]
        dists = [i[2] for i in cleaned_data]

        # scale the errors according to distance
        x_err_scaled, y_err_scaled = scale_errors_by_distance(x_err, y_err, dists)

        # estimate (and plot!) and eep distribution from these values
        fig_name = pathlib.Path("..", "figures", s + "_m.png")
        eep_cov, eep_cx, eep_cy = utils.plot_eep_dist(x_err_scaled, y_err_scaled, fig_name, scale="m")

        # save the eep to file
        json_name = pathlib.Path("..", "data", s + ".json")
        save_eep(eep_cov, eep_cx, eep_cy, json_name)

        # now simulate some data with our calculated distribution to confirm we can model it correctly
        test_nep = NavErrorProfile(eep_cov, eep_cx, eep_cy)
        test_x_err, test_y_err, test_dists = test_eep(test_nep)

        # scale simulated data according to distance
        test_x_scaled, test_y_scaled = scale_errors_by_distance(test_x_err, test_y_err, test_dists)

        # save tested eep to file for comparison
        test_fig_name = pathlib.Path("..", "figures", "simulated_" + s + ".png")
        test_eep_cov, test_eep_cx, test_eep_cy = utils.plot_eep_dist(test_x_scaled, test_y_scaled, test_fig_name, scale="m")

        # lastly, compare sampling at 1m to sampling 100 times at 1cm
        compare_cm_to_m(test_nep, s)
