import os
import datetime

import IPython
import IPython.display
from pylab import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
# import holidays
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
# import tensorflow as tf
# Setup environment
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# set styles
# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1)

# set plotting parameters
rcParams['figure.figsize'] = 16,8

# Load data windows
# df_input = pd.read_csv('C:/Users/timon/Documents/GitHub/Economics-Project/Data/weather-energy-data.csv', index_col="Datetime", parse_dates=True).iloc[:,1:]
# Load data Mac
df_input = pd.read_csv('/Users/timongodt/Documents/GitHub/Economics-Project/Data/weather-energy-data-update.csv', index_col="Datetime", parse_dates=True).iloc[:,1:]

# add time signal 
timestamp_s = df_input.index.map(pd.Timestamp.timestamp)
day = 24*60*60
week = 24*60*60*7
year = (365.2425)*day

df_input['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df_input['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))


# plt.plot(np.array(df_input['Day sin'])[:25])
# plt.plot(np.array(df_input['Day cos'])[:25])
# plt.xlabel('Time [h]')
# plt.title('Time of day signal')

# import libraries for Deep Learning
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# prepare data
df = df_input[['kWh']]

# Train Test set split - we want to forecast 1 month into the future so out test set should be at least one month 
len(df)
# we will go with a 90-10 train-test split such that our test set represents 3 months worth of data
train =  df[:(round(0.9*len(df)))]
test = df[round(0.9*len(df)):]
len(df) == len(train) + len(test)

# Scale data
scaler = MinMaxScaler()

# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
# scaler.fit(train)
scaled_train = train
scaled_test = test


# Let's define to get 168 Days back wbich represents one week and then predict the next week out
n_input = 24
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=10)

# Check Generated time series object 
len(scaled_train)
len(generator) # n_input = 2
scaled_train
X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')

# DEFINE THE MODEL 
# define model
model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
# comile model
model.compile(optimizer='adam', loss='mse')

# get model summary
model.summary()

# fit model
model.fit_generator(generator,epochs=20, shuffle=False)

# model performance
model.history.history.keys()
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

# Evaluate on Test Data
# first_eval_batch = scaled_train[-24:]
# first_eval_batch
# first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
# model.predict(first_eval_batch)
# scaled_test[0]

# LOOP to get predictions for entire test set
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
test_predictions

# INverse transform to compare to actual data
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions

# IGNORE WARNINGS
test['Predictions'] = true_predictions

# plot predictions 
test.plot(figsize=(12,8))





