import string
import matplotlib.pyplot as plt

def plot_data(data):
    plt.plot(data)
    plt.show()

def open_csv_file(filename: string):
    with open(filename, 'r') as f:
        data = f.read().split('\n')
    return data


if __name__ == '__main__':
    rainData = open_csv_file('../data/rain_data.csv')
    for row in rainData:
        print(row)