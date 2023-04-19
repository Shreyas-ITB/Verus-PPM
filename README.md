# Verus-PPM
This is a basic artificial intelligence model from Keras module that predicts the future price of VerusCoin. It also plots a graph of the actual price and its predictions made! Verus-PPM stands for Verus Price Prediction Model.

## Dataset

The model dataset is automatically fetched by the python file from [yahoo finance]{ https://finance.yahoo.com/quote/VRSC-USD?p=VRSC-USD} website. The price gets tracked from the specified time range in the python file and gets converted to a proper dataset with which the model trains on.

## Instructions

*Tested python version* `Python 3.9.6`

First, install the dependencies.
- On Windows
- `pip install -r requirements.txt`
- On linux and Mac
- `pip3 install -r requirements.txt`

Secondly, train the model
- `crypto_currency = "VRSC-USD"` Edit this line (number 10) to get the prediction of any currency. Make sure it is available on yahoo finance.
- `prediction_days = 60` Edit the Number `60` according to your preference. This is the number of days that it should filter from the dataset and train on, the higher the better.
- `future_prediction = 60` Edit the Number `60` according to your preference. This is the number of days that the model should predict, for example if mentioned 10 it will predict the future price of 10 days.
- `chart_predictionstart = dt.datetime(2018, 6, 1)` Here edit the date which is in the perenthesis seperated by comma which goes by `year month day`, This specifies the starting of the dataset meaning it will start collecting the data from that day.
- `single_predictionstart = dt.datetime(2018, 6, 1)` The same goes here edit the date accordingly
- `end = dt.datetime.now()` This is the end date, This specifies the ending of the dataset meaning it will stop collecting the data after that date. Automatically it is set to the current date

## My Thoughts
This is a basic sequential model from keras library that is trained on the crypto price dataset, This model is very light weight and can be trained on CPU with very less RAM. The price prediction will be good or average depending on how much data you provide. This is not the best price prediction model but its good enough.
