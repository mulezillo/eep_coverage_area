import pathlib

import numpy as np

import utils
from nav_error_profile import EEP
from coverage import Coverage


if __name__ == "__main__":
    """
    Simulate the effects of increasing the overlap percent on each cleaning pass.
    """
    # setup sim variables
    overlap_pcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    step_size = 0.01
    width_m = 10
    height_m = 10
    num_sims = 1000
    profile = pathlib.Path("..", "data", "measured_eep_good.json")
    cov = np.array(utils.load_profile(profile)["covariance"])

    # scale the eep to cm precision
    cm_cov_mat = utils.scale_cov_mat(cov, step_size)
    nav_prof = EEP(cm_cov_mat)
    all_coverage_pcts = []
    for o_pct in overlap_pcts:
        # need a history of coverage percentages
        coverage_pct_history = []
        # run sim
        for n in range(num_sims):
            c = Coverage(nav_prof, width_m, height_m, overlap_pct=o_pct)
            c.simulate()
            c_pct = c.calc_coverage()
            coverage_pct_history.append(c_pct)
        all_coverage_pcts.append(coverage_pct_history)

    curve_labels = [f"{int(100 * i)}% overlap " for i in overlap_pcts]
    cdf_save_path = pathlib.Path("../", "figures", "overlap_coverage_cdf.png")
    utils.plot_cdf_set(all_coverage_pcts, curve_labels, "Coverage %", "CDF of Coverage %", cdf_save_path)

    # since we are (unfortunately) not set up well to calculate and display the number of cleaning passes required
    # per overlap percent, I had them all calculated once and then printed out. Here they are:
    # 0.1: 20
    # 0.2: 23
    # 0.3: 26
    # 0.4: 30
    # 0.5: 36
    # 0.6: 45