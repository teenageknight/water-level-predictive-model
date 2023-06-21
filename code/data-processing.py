import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import pacf
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from helpers import Helpers

helpers = Helpers()


def plot_data(x, y, x_label: str = "", y_label: str = "", title: str = ""):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x, y)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

    # Rotate x-axis labels diagonally
    plt.xticks(rotation=45)

    plt.show()


def clean_rain_data(raw_data: list[str]) -> dict[(str, float)]:
    """
    Summary

    Parameters:
    raw_data (list[str]): List of strings containing raw data

    Returns:
    rain_data list[(str, float)]: List of lists containing the date and the amount of rain
    """
    rain_data = []

    for i, row in enumerate(raw_data):
        # Skip first row which contains the headers
        if i == 0:
            continue

        elms = row.split(",")

        # Some of the early dates before 2007 are missing data, so we fill those in with zero for now
        if len(elms) <= 1 or elms[1] == "":
            rain_data.append([elms[0], 0])
        else:
            rain_data.append([elms[0], float(elms[1])])

    return rain_data


def clean_water_level_data(raw_data: list[str]) -> list[(str, float)]:
    """
    Summary

    Parameters:
    raw_data (list[str]): List of strings containing raw data

    Returns:
    water_levels list[[str, float]]: List of lists containing the date and the amount of rain
    """
    water_levels = []

    for i, row in enumerate(raw_data):
        if i <= 28:
            continue
        elif len(row) > 0:
            x = row.split()

            water_levels.append([x[2] + " " + x[3] + ":00", float(x[5])])

    return water_levels


def sync_values(rain_data, water_level_data):
    # Aims to sync the rain data and the water level data so that they have the same dates.
    # To do this, we will average the water level data for each hour, and then average the rain data for each hour, rounding to the nearest hour.

    rounded_rain_data = [
        [helpers.round_to_nearest_hour(date), value] for date, value in rain_data
    ]
    rounded_water_level_data = [
        [helpers.round_to_nearest_hour(date), value] for date, value in water_level_data
    ]

    # Deduplicate the data
    deduplicated_rain_data = helpers.deduplicate_and_average(rounded_rain_data)
    deduplicated_water_level_data = helpers.deduplicate_and_average(
        rounded_water_level_data
    )

    all_rain_dates = [i[0] for i in deduplicated_rain_data]
    all_water_level_dates = [i[0] for i in deduplicated_water_level_data]

    # abstract to a function
    removed_dates = []

    for i in all_rain_dates:
        if i not in all_water_level_dates:
            removed_dates.append(i)

    for i in all_water_level_dates:
        if i not in all_rain_dates:
            removed_dates.append(i)

    rain_map = {}
    water_map = {}
    for i in deduplicated_rain_data:
        rain_map[i[0]] = i[1]
    for i in deduplicated_water_level_data:
        water_map[i[0]] = i[1]

    for i in removed_dates:
        if i in all_rain_dates:
            del rain_map[i]
        if i in all_water_level_dates:
            del water_map[i]

    # Maybe eventually abstract this
    data = {}
    for key in rain_map:
        data[key] = rain_map[key], water_map[key]

    return data


def calculate_pacf(data):
    # Calculate PACF
    pacf_values = pacf(water_level_data, nlags=10)

    # Plot PACF
    # plt.bar(range(len(pacf_values)), pacf_values)
    # plt.xlabel('Lag')
    # plt.ylabel('Partial Autocorrelation')
    # plt.title('Partial Autocorrelation Function (PACF)')
    # plt.show()

    # Assuming your data is stored in a DataFrame named 'data'
    precipitation_data = data["precipitation"]

    # Calculate PACF
    pacf_values = pacf(precipitation_data, nlags=10)

    # Plot PACF
    # plt.bar(range(len(pacf_values)), pacf_values)
    # plt.xlabel('Lag')
    # plt.ylabel('Partial Autocorrelation')
    # plt.title('Partial Autocorrelation Function (PACF)')
    # plt.show()


def moreStuff(data):
    X = data["precipitation"].values.reshape(-1, 1)
    y = data["water_level"].values

    # Define the kernel
    kernel = RBF()

    # Create and fit the Gaussian Process Regression model
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X, y)

    # Generate predictions for the test set
    test_precipitation = np.array([[2.0], [0.0], [0.0]])  # Example precipitation values
    predictions = gpr.predict(test_precipitation)

    print(predictions)


if __name__ == "__main__":
    # Open the data files and pull out the data
    raw_rain_data = helpers.open_csv_file("../data/rain_data_hourly.csv")
    raw_water_level_data = helpers.open_csv_file("../data/water_level_data.txt")

    rain_data = clean_rain_data(raw_rain_data)
    # print(rain_data)

    water_level_data = clean_water_level_data(raw_water_level_data)
    # print(water_level_data)
    print(len(water_level_data))
    print(len(rain_data))

    data = sync_values(rain_data, water_level_data)

    rows = []
    for i in data:
        rows.append([i, data[i][0], data[i][1]])

    helpers.write_csv_file("../data/combined_data.csv", rows, ["date", "rain", "water"])
