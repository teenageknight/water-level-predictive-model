import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import pacf
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def plot_data(x, y, x_label: str = '', y_label: str = '', title: str = ''):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x, y)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

    # Rotate x-axis labels diagonally
    plt.xticks(rotation=45)

    plt.show()

def open_csv_file(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        data = f.read().split('\n')
        f.close()
    return data

def clean_rain_data(raw_data: list[str]) -> dict[(str, float)]:
    """
    Summary

    Parameters:
    raw_data (list[str]): List of strings containing raw data

    Returns:
    list[(str, float)]: List of tuples containing the date and the amount of rain
    """
    map = {}

    for i, row in enumerate(raw_data):
        # Skip first row which contains the headers
        if i == 0:
            continue

        elms = row.split(',')
        # Skip the rows with missing data. If the key already exists, average the values. If it doesnt, put the key in the dictionary
        if elms[3] in map.keys() and elms[4] != '':
            map[elms[3]] = (float(map[elms[3]]) + float(elms[4])) / 2
        elif elms[4] != '':
            # Could get rid of try except, but I want to see what the error is if it does
            try:
                map[elms[3]] = float(elms[4])
            except:
                print('Error: ', elms[3], elms[4])

    return map

# def make_dataframe(rain_data, water_level_data) -> pd.DataFrame:
#     df = pd.DataFrame(data.items(), columns=['Date', 'Rainfall'])
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.set_index('Date', inplace=True)
#     return df

def clean_water_level_data(raw_data: list[str]) -> list[(str, float)]:
    """
    Summary

    Parameters:
    raw_data (list[str]): List of strings containing raw data

    Returns:
    list[(str, float)]: List of tuples containing the date and the amount of rain
    """
    map = {}
    all_dates = {}

    for i, row in enumerate(raw_water_level_data):
        if i <= 28:
            continue
        elif len(row) > 0:
            x = row.split()
            if x[2] in map.keys():
                # map[x[2]].append((x[3], float(x[5])))
                map[x[2]].append(float(x[5]))
            else:
                # map[x[2]] = [(x[3], float(x[5]))]
                map[x[2]] = [float(x[5])]
            all_dates[x[2] + '-' + x[3]] = float(x[5])

    return map, all_dates

def stuff(rain_data, water_level_data):
    # Basically there is an issue with the data points. Some dates do not have data, so we need to remove those dates from the data set
    # This way we have the same number of data points for both data sets
    new_map = {}
    for i in water_level_data:
        date_formatted = datetime.strptime(i, '%Y-%m-%d').strftime('%-m/%-d/%y')
        new_map[date_formatted] = np.mean(water_level_data[i])

    map_copy = new_map.copy()
    for date in new_map:
        if date not in rain_data.keys():
            del map_copy[date]

    rain_copy = rain_data.copy()
    for date in rain_data:
        if date not in new_map.keys():
            del rain_copy[date]

    data = pd.DataFrame({'precipitation': rain_copy.values(), 'water_level': map_copy.values()}, index=map_copy.keys())
    print(data)

    # Assuming your data is stored in a DataFrame named 'data'
    water_level_data = data['water_level']

    # Calculate PACF
    pacf_values = pacf(water_level_data, nlags=10)

    # Plot PACF
    # plt.bar(range(len(pacf_values)), pacf_values)
    # plt.xlabel('Lag')
    # plt.ylabel('Partial Autocorrelation')
    # plt.title('Partial Autocorrelation Function (PACF)')
    # plt.show()

    # Assuming your data is stored in a DataFrame named 'data'
    precipitation_data = data['precipitation']

    # Calculate PACF
    pacf_values = pacf(precipitation_data, nlags=10)

    # Plot PACF
    # plt.bar(range(len(pacf_values)), pacf_values)
    # plt.xlabel('Lag')
    # plt.ylabel('Partial Autocorrelation')
    # plt.title('Partial Autocorrelation Function (PACF)')
    # plt.show()

    return data

def moreStuff(data):
    X = data['precipitation'].values.reshape(-1, 1)
    y = data['water_level'].values

    # Define the kernel
    kernel = RBF()

    # Create and fit the Gaussian Process Regression model
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X, y)

    # Generate predictions for the test set
    test_precipitation = np.array([[2.0], [0.0], [0.0]])  # Example precipitation values
    predictions = gpr.predict(test_precipitation)

    print(predictions)



if __name__ == '__main__':
    # Open the data files and pull out the data
    raw_rain_data = open_csv_file('../data/rain_data.csv')
    raw_water_level_data = open_csv_file('../data/water_level_data.txt')

    rain_data = clean_rain_data(raw_rain_data)

    water_level_data, all_dates = clean_water_level_data(raw_water_level_data)

    data = stuff(rain_data, water_level_data)

    moreStuff(data)

    # Plot the Rainfall data
    # plot_data(rain_data.keys(), rain_data.values(), title='Rainfall', y_label='Rainfall (inches)')

    # Plot the Water Level data
    # plot_data(all_dates.keys(), all_dates.values(), title='Water Level', y_label='Water Level (ft)')

