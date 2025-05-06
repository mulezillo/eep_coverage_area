import pathlib
import statistics

import numpy as np
import matplotlib.pyplot as plt

import utils
from nav_error_profile import EEP
from coverage import Coverage


if __name__ == "__main__":
    """
    Estimate the expected number of cleanings required to cover an entire area.
    """
    step_size = 0.01
    width_m = 10
    height_m = 10
    num_sims = 100

    # build all the eep profile paths
    profile = pathlib.Path("..", "data", "measured_eep_good.json")

    cov = np.array(utils.load_profile(profile)["covariance"])
    # scale the eep to cm precision
    cm_cov_mat = utils.scale_cov_mat(cov, step_size)
    nep = EEP(cm_cov_mat)

    # also need a history of coverage percentages
    coverage_after_cleaning = {}

    # run sim
    for n in range(num_sims):
        c = Coverage(nep, width_m, height_m)

        complete_coverage = False
        cleaning_number = 1
        while not complete_coverage:
            c_map = c.simulate()
            c_pct = c.calc_coverage()
            # add this pct for this cleaning number
            try:
                coverage_after_cleaning[cleaning_number].append(c_pct)
            except KeyError:
                # this is a new cleaning number index
                coverage_after_cleaning[cleaning_number] = [c_pct]

            print(c_pct)
            cleaning_number += 1
            complete_coverage = (c_pct > 0.99)
        print(f"finished sim {n}") # for debugging purposes

    # okay now we want to calculate the expected coverage after each cleaning number
    # but first, let's pad the larger numbers to make sure they are all the same size
    padded_coverage_dict = {}
    max_size = len(coverage_after_cleaning[1])  # this is safe, because there must always be at least 1 cleaning
    for num, pct_list in coverage_after_cleaning.items():
        num_items = len(pct_list)
        if num_items == max_size:
            # no padding needed
            padded_coverage_dict[num] = pct_list
        elif num_items < max_size:
            # need to pad here, all paddings are 100 pct coverages.
            new_list = pct_list + [1.0] * (max_size - num_items)
            padded_coverage_dict[num] = new_list
        else:
            raise Exception("More items than max found! Impossible!")

    # now compute the mean of each number in the dict
    expected_coverages = []
    cleaning_number = []
    for i, pcts in padded_coverage_dict.items():
        mean_cov = statistics.mean(pcts)
        expected_coverages.append(mean_cov)
        cleaning_number.append(i)
        print(f"Expected coverage after {i} cleanings: {mean_cov}")

    # now plot the results
    plt.clf()
    plt.scatter(cleaning_number, expected_coverages)
    plt.xlabel("Number of Consecutive Cleanings")
    plt.ylabel("Expected Cleaning Coverage %")
    plt.title("Expected Cleaning Coverage vs Number of Cleanings")
    save_path = pathlib.Path("..", "figures", "expected_time_complete_coverage.png")
    plt.savefig(save_path)
