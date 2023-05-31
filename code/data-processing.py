import matplotlib.pyplot as plt

def plot_data(data):
    plt.plot(data)
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

def clean_water_level_data(raw_data: list[str]) -> list[(str, float)]:
    """
    Summary

    Parameters:
    raw_data (list[str]): List of strings containing raw data

    Returns:
    list[(str, float)]: List of tuples containing the date and the amount of rain
    """
    map = {}

    for i, row in enumerate(raw_water_level_data):
        if i <= 28:
            continue
        elif len(row) > 0:
            x = row.split()
            map[x[2]] = (x[3], float(x[5]))

    return map


if __name__ == '__main__':
    raw_rain_data = open_csv_file('../data/rain_data.csv')

    rain_data = clean_rain_data(raw_rain_data)

    raw_water_level_data = open_csv_file('../data/water_level_data.txt')

    water_level_data = clean_water_level_data(raw_water_level_data)

    print(water_level_data)

    # plot_data(map.values())