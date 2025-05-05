import csv
import pathlib

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    cov_scaling_path = pathlib.Path("..", "figures", "dim_statistics.csv")
    with open(cov_scaling_path, "r") as f:
        csv_reader = csv.reader(f)
        cov_area = []
        dimensions = []
        for i, row in enumerate(csv_reader):
            if i != 0:
                # skip header
                dimension_strs = row[0].split(" x ")
                dimension = float(dimension_strs[0]) * float(dimension_strs[1])
                dimensions.append(dimension)
                cov_pct = float(row[1]) * 100
                cov_area.append(cov_pct)
        f.close()

    plt.clf()
    plt.scatter(dimensions, cov_area)
    plt.ylabel("Expected Coverage %")
    plt.xlabel("Cleaning Area (m^2)")
    plt.title("Expected Cleaning Coverage vs Cleaning Area")
    save_path = pathlib.Path("..", "figures", "coverage_scaling.png")
    plt.savefig(save_path)

    # now let's look for the scaling factor
    plt.clf()
    log10_dims = np.log10(dimensions)
    log10_areas = np.log10(cov_area)
    plt.scatter(log10_dims, log10_areas, label="Simulated Coverage")
    slope, intercept, r, p, std_err = stats.linregress(log10_dims, log10_areas)
    print(f"Scaling factor: {slope}")
    y_fit = [slope * a + intercept for a in log10_dims]
    plt.plot(log10_dims, y_fit, label="Linear Fit", color="purple")
    plt.legend()
    plt.ylabel("LOG10 Expected Coverage %")
    plt.xlabel("LOG10 Cleaning Area (m^2)")
    plt.title("Expected Cleaning Coverage vs Cleaning Area")
    save_path = pathlib.Path("..", "figures", "coverage_power_law_scaling.png")
    plt.savefig(save_path)
