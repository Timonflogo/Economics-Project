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
rcParams['figure.figsize'] = 16, 6

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


plt.plot(np.array(df_input['Day sin'])[:25])
plt.plot(np.array(df_input['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')

# import libraries for time series analysis
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima  

# prepare data
df = df_input[['kWh', 'hour', 'Day sin']]

# decompose series
# df.interpolate(inplace=True) 
df.index.freq = 'H'
result = seasonal_decompose(df['kWh'])
result.plot();

# run ADF test
# from statsmodels.tsa.stattools import adfuller

# def adf_test(series,title=''):
#     """
#     Pass in a time series and an optional title, returns an ADF report
#     """
#     print(f'Augmented Dickey-Fuller Test: {title}')
#     result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
#     labels = ['ADF test statistic','p-value','# lags used','# observations']
#     out = pd.Series(result[0:4],index=labels)

#     for key,val in result[4].items():
#         out[f'critical value ({key})']=val
        
#     print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
#     if result[1] <= 0.05:
#         print("Strong evidence against the null hypothesis")
#         print("Reject the null hypothesis")
#         print("Data has no unit root and is stationary")
#     else:
#         print("Weak evidence against the null hypothesis")
#         print("Fail to reject the null hypothesis")
#         print("Data has a unit root and is non-stationary")
        
# adf_test(df['kWh'])

# run Auto Arima to determine model
auto_arima(df['kWh'],seasonal=True,m=24).summary()




