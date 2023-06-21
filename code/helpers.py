from datetime import datetime, timedelta
import csv


class Helpers:
    def __init__(self):
        pass

    def open_csv_file(self, filename: str) -> list[str]:
        """Opens a csv file and returns the data as a list of strings split by the newline charecter.

        Args:
            filename (str): is the name of the file, including the relative path to it

        Returns:
            list[str]: is a list of strings split by the newline charecter
        """
        with open(filename, "r") as f:
            data = f.read().split("\n")
            f.close()
        return data

    def write_csv_file(self, filename: str, data: list[str], header: list = []):
        with open(filename, "w", encoding="UTF8") as f:
            writer = csv.writer(f)

            if len(header) > 0:
                writer.writerow(header)

            writer.writerows(data)

            f.close()

    def round_to_nearest_hour(self, date_str: str) -> str:
        """Rounds a date string to the nearest hour.

        Args:
            date_str (str): is a string representing a date in the format of "%Y-%m-%d %H:%M:%S"

        Returns:
            str: is a string representing a date in the format of "%Y-%m-%d %H:%M:%S"

        """
        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        rounded_date = date.replace(minute=0, second=0, microsecond=0)
        if date.minute >= 30:
            rounded_date += timedelta(hours=1)
        return rounded_date.strftime("%Y-%m-%d %H:%M:%S")

    def deduplicate_and_average(
        self, arr: list[list[str, float]]
    ) -> list[list[str, float]]:
        """Deduplicates and averages the values in a 2D array. It uses the first element as a key and the second as a value

        Args:
            arr (list[list[str, float]]): is a 2D array containing a key and a value

        Returns:
            list[list[str, float]]: is a 2D array containing a key and a value with no duplicates and values averaged

        """
        deduplicated_arr = []
        temp_dict = {}

        # Iterate over the subarrays
        for subarr in arr:
            key = subarr[0]
            value = subarr[1]

            # Check if the key already exists in the dictionary
            if key in temp_dict:
                # Add the value to the existing key
                temp_dict[key].append(value)
            else:
                # Create a new key-value pair
                temp_dict[key] = [value]

        # Iterate over the dictionary and calculate the average for each key
        for key, values in temp_dict.items():
            average_value = sum(values) / len(values)
            deduplicated_arr.append([key, average_value])

        return deduplicated_arr
