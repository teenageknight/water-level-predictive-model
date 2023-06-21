import requests
from datetime import datetime, timedelta
import time
from helpers import Helpers

helpers = Helpers()

startDate = 20071005
endDate = 20230529
currentDate = startDate
date_format = "%Y%m%d"


url = "https://api.weather.com/v1/location/KAVL:9:US/observations/historical.json?apiKey={YOUR_API_KEY}&units=e&startDate={0}&endDate={1}"

fieldnames = ["time", "precip_hrly"]

data = []

# Hardcoded for now, but basically it is the number of das between the start and end date times 30 (since )
for i in range(186):
    print("Requesting data for date: " + str(currentDate))
    # Sleep for 1 minute every 60 requests so we don't get rate limited
    if i % 60 == 0 and i != 0:
        print("Sleeping for 60 seconds...")
        time.sleep(60)

    # Convert the string to a datetime object
    date = datetime.strptime(str(currentDate), date_format)

    # Increment the date by 1 day
    incremented_date = date + timedelta(days=30)

    # Convert the incremented date back to the desired format
    incremented_date_str = incremented_date.strftime(date_format)

    request_url = url.format(currentDate, incremented_date_str)

    r = requests.get(request_url)

    daily_observation = r.json()["observations"]

    for observation in daily_observation:
        timestamp = observation["valid_time_gmt"]
        date = datetime.fromtimestamp(timestamp)
        readable_date = date.strftime("%Y-%m-%d %H:%M:%S")

        precip = observation["precip_hrly"]

        data.append([readable_date, precip])

    # Increment the current date by 1 day
    currentDateIncrement = incremented_date + timedelta(days=1)
    currentDate = currentDateIncrement.strftime(date_format)


helpers.write_csv_file("../data/rain_data_hourly.csv", data, fieldnames)


print("Completed!")
