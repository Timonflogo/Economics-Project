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
import holidays
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
df_input = 

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

# import libraries for time series analysis
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tools.eval_measures import mse,rmse     # for ETS Plots
from pmdarima import auto_arima  

# prepare data
df = df_input[['kWh', 'hour', 'Day sin']]

# decompose series
# df.interpolate(inplace=True) 
# reduce time series to one year of data
df_SD = df['20201101':'20211101']

# resample by day
df_SD = df.resample('D').sum()

# Reduce dataframe to include only single variable 
df_SD = df_SD[['kWh']]
df_SD.index.freq = 'D'

df_SD.plot()

df_SD['20210801':'20211101'].plot()

# Subplots for series
fig, axs = plt.subplots(3)
fig.suptitle('kWh Series')
axs[0].plot(df_input["kWh"])
axs[1].plot(df_SD["kWh"])
axs[2].plot(df[['kWh']].resample('M').sum())



result = seasonal_decompose(df_SD['kWh'], period=7, model = "additive")
result.plot({'figure.figsize': (16,9)});
result.seasonal['20210801':'20211101'].plot() # weekly seasonality 

# use STL decomposition
from statsmodels.tsa.seasonal import STL 
loess = STL(df_SD['kWh'], period=7, robust = True)
res = loess.fit()
res.plot()


# add holiday column to dataframe
hol = holidays.DK
hol = list(holidays.DK(years = {2018,2019,2020,2021}))
hol = pd.DataFrame(list(holidays.DK(years = {2018,2019,2020,2021})))
hol["Holiday"] = 1
hol.columns = ['Datetime', 'Holiday']
hol['Datetime'] = pd.to_datetime(hol['Datetime'])

# merge into dataframe 
df_SD = pd.merge(df_SD, hol, how='left', on = 'Datetime')
df_SD['Holiday'] = df_SD['Holiday'].replace(np.nan, 0)
df_SD = df_SD.set_index('Datetime')
df_SD.index.freq = 'D'

# run ADF test
from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
adf_test(df['kWh'])


# Modelling preparation 
# Train Test set split - we want to forecast 1 month into the future so out test set should be at least one month 
len(df_SD)
df_SD[:987]
# we will go with a 90-10 train-test split such that our test set represents 3 months worth of data
train =  df_SD[:(round(0.9*len(df_SD)))]
test = df_SD[round(0.9*len(df_SD)):]
len(df_SD) == len(train) + len(test) # True


######################### Daily forecasts

######################### ARIMA(2, 0, 1)

# run Auto Arima to determine model without specified seasonality
auto_arima(df_SD['kWh'],seasonal=True).summary()
# SARIMAX(1, 0, 3)

# Fit SARIMA NO EXOGENOUS
model1 = SARIMAX(train['kWh'], order=(1,0,3),enforce_invertibility=False)
results1 = model1.fit()
results1.summary()

# obtain predicted results
start = len(train)
end = len(train)+len(test)-1
predictions1 = results1.predict(start=start, end=end, dynamic=False).rename("ARIMA(2,0,1) Predictions")
    
# Evaluate model performance
from statsmodels.tools.eval_measures import mse,rmse 
error1 = mse(test['kWh'], predictions1)
error2 = rmse(test['kWh'], predictions1)

print(f'ARIMA(2, 0, 1) MSE Error: {error1:11.10}')
print(f'ARIMA(2, 0, 1) RMSE Error: {error2:11.10}')


########################## SARIMA(2, 0, 2)x(1, 0, [1], 7) 

# run Auto Arima to determine model
auto_arima(df_SD['kWh'],seasonal=True,m=7).summary()
# SARIMAX(3, 0, 0)x(1, 0, [1], 7) 

# Fit SARIMA NO EXOGENOUS
model2 = SARIMAX(train['kWh'], order=(3,0,0), seasonal_order=(1,0,1,7),enforce_invertibility=False)
results2 = model2.fit()
results2.summary()

# obtain predicted results
start = len(train)
end = len(train)+len(test)-1
predictions2 = results2.predict(start=start, end=end, dynamic=False).rename("SARIMA(2,0,2)(1,0,1,7) Predictions")

# plot predictions
title='Electricity Demand Forecast DAILY'
ylabel='kWh'
xlabel=''

ax = test['kWh'].plot(legend=True,figsize=(12,6),title=title)
predictions1.plot(legend=True)
predictions2.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# Evaluate model performance
error3 = mse(test['kWh'], predictions2)
error4 = rmse(test['kWh'], predictions2)

print(f'SARIMA(1,0,0)(2,0,0,7) MSE Error: {error1:11.10}')
print(f'SARIMA(1,0,0)(2,0,0,7) RMSE Error: {error2:11.10}')
    
######################## SARIMAX(2, 0, 2)x(1, 0, [1], 7) 
# run Auto Arima to determine model
auto_arima(df_SD['kWh'],seasonal=True,m=7).summary()
# SARIMAX(2, 0, 2)x(1, 0, [1], 7) 

# Fit SARIMA WITH EXOGENOUS
model3 = SARIMAX(train['kWh'],exog=train['Holiday'],order=(3,0,0),seasonal_order=(1,0,1,7),enforce_invertibility=False)
results3 = model3.fit()
results3.summary()

# obtain predicted results
start = len(train)
end = len(train)+len(test)-1
predictions3 = results3.predict(start=start, end=end, dynamic=False).rename("SARIMAX(2,0,2)(1,0,1,7) Predictions")

# plot predictions
title='Electricity Demand Forecast DAILY'
ylabel='kWh'
xlabel=''

ax = test['kWh'].plot(legend=True,figsize=(12,6),title=title)
predictions1.plot(legend=True)
predictions2.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

# Evaluate model performance
error5 = mse(test['total'], predictions3)
error6 = rmse(test['total'], predictions3)

print(f'SARIMAX(2,0,2)(1,0,1,7) MSE Error: {error1:11.10}')
print(f'SARIMAX(2,0,2)(1,0,1,7) RMSE Error: {error2:11.10}')


################ Hourly forecasts 

df_H = df['20211001':'20211101']

# run auto arima on hourly data
auto_arima(df_H['kWh'],seasonal=True,m=24).summary()
# SARIMAX(2, 0, 0)x(2, 0, 0, 24) 

# Train Test set split - we want to forecast 1 month into the future so out test set should be at least one month 
len(df)
# we will go with a 90-10 train-test split such that our test set represents 3 months worth of data
train1 =  df[:(round(0.9*len(df)))]
test1 = df[round(0.9*len(df)):]
len(df) == len(train1) + len(test1) # True

# Fit SARIMA WITH EXOGENOUS
model4 = SARIMAX(train1['kWh'],exog=train1['hour'],order=(2,0,0),seasonal_order=(2,0,0,24),enforce_invertibility=False)
results4 = model4.fit()
results4.summary()

# obtain predicted results
start1 = len(train1)
end1 = len(train1)+len(test1)-1

exog_forecast = test1[['hour']] 
predictions4 = results4.predict(start=start1, end=end1, exog=exog_forecast).rename('SARIMAX(2,0,0)(2,0,0,24) Predictions')

# plot predictions
title='Electricity Demand Forecast HOURLY'
ylabel='kWh'
xlabel=''

ax = test1['kWh'].plot(legend=True,figsize=(12,6),title=title)
predictions4.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# Evaluate model performance
error7 = mse(test1['total'], predictions4)
error8 = rmse(test1['total'], predictions4)

print(f'SARIMAX(2,0,2)(2,0,0,24) MSE Error: {error1:11.10}')
print(f'SARIMAX(2,0,2)(2,0,0,24) RMSE Error: {error2:11.10}')






