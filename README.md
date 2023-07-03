# water-level-predictive-model

A personal research project to create a predictive model to determine water level of the Chattooga River.

My goal is to gain a better understanding of predictive models, data analysis and Machine learning through a topic that I enjoy, kayaking. As a whitewater kayaker with Outdoor Recreation at Georgia Tech, I frequently take trips out to the Chatooga River. One of the main factors that determines if a trip goes out is the water level. If it is too high, we cannot kayak. However, the watershed for the river is quite big and levels fluctuate quite a bit. I want to use data from the USGS to create a predictive model that can determine the water level of the river based on the weather and other factors.

I will be using the data from the USGS website. The data is collected from a gauge on the river and is updated every 15 minutes. The data is available in a csv format. I will also be using the weather data from the IBM weather API. The data is available in a json format.

Here is a sample of the data breakdowns:

Histogram of Data
![Histogram of Data](https://raw.githubusercontent.com/teenageknight/water-level-predictive-model/main/pics/all_data_histogram.png)

All Data
![All Data](https://raw.githubusercontent.com/teenageknight/water-level-predictive-model/main/pics/all_data.png)

LSTM Model Output
![LSTM](https://raw.githubusercontent.com/teenageknight/water-level-predictive-model/main/pics/lstm_results_1.png)
