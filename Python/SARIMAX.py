import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans, KMeans
import warnings
import itertools
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time
import folium
from folium import Choropleth ,  Circle, Marker

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# set styles
# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1)

# set plotting parameters
rcParams['figure.figsize'] = 22, 10

# set random seed
random_seed = 20
np.random.seed(random_seed)

# load dataset 
df = pd.read_csv('/Users/timongodt/Documents/GitHub/Economics-Project/Data/weather-energy-data.csv')

# Change Datetime to date/time datatype
df['Datetime'] = pd.to_datetime(df['Datetime'], infer_datetime_format=True)
df.info()

# set Datetime as index
df = df.set_index('Datetime')

# reduce dataframe and include only variables used for the model 
df = df[['cross_cafe', 'cloud_cover', 'humidity', 'pressure', 'temp_mean_past1h', 
         'wind_max_per10min_past1h', 'sun_last1h_glob', 'hour', 'day_of_month',
         'day_of_week', 'is_weekend']]

# split data
n = len(df)
train = df[0:int(n*0.9)]
test = df[int(n*0.9):]

# explore data 
result = seasonal_decompose(df['cross_cafe'], model='additive')
result.plot()
pyplot.show()

#Stationarity test
import statsmodels.tsa.stattools as sts 
dftest = sts.adfuller(test.iloc[:,:].cross_cafe)
print('ADF Statistic: %f' % dftest[0])
print('p-value: %f' % dftest[1])
print('Critical Values:')
for key, value in dftest[4].items():
  print('\t%s: %.3f' % (key, value))

# conclusion: cross_cafe series is STATIONARY <---------------

# ============================ prepare SARIMAX ==============================

# Load functions
def get_sarima_params(data):
  p = d = q = range(0, 2)
  pdq = list(itertools.product(p, d, q))
  seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]
  result_table = pd.DataFrame(columns=['pda','seasonal_pda','aic'])

  for param in pdq:
      for param_seasonal in seasonal_pdq:
          try:
            mod = sm.tsa.statespace.SARIMAX(data,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            result_table = result_table.append({'pda':param, 'seasonal_pda':param_seasonal, 'aic':results.aic},ignore_index=True)
          except:
            continue

  optimal_params = result_table[result_table['aic']==result_table.aic.min()]
  order = optimal_params.pda.values[0]
  seasonal_order = optimal_params.seasonal_pda.values[0]
  return (order,seasonal_order)

def apply_sarimax(train_data,train_exog, test_data, test_exog , order , seasonal_order):
  print('SARIMAX MODEL ORDERS ARE = {} {} '.format(order,seasonal_order))
   
  mod = sm.tsa.statespace.SARIMAX(train_data,exog=train_exog,order=order,seasonal_order=seasonal_order)
  results = mod.fit()
  
  pred = results.get_prediction(start=train_data.index[0],end=train_data.index[-1],exog=train_exog,dynamic=False)
  train_forecast = pred.predicted_mean.round()
  train_forecast[train_forecast<0] = 0


  pred1 = results.get_prediction(start=test_data.index[0],end=test_data.index[-1],exog=test_exog.iloc[:-1,:],dynamic=False)
  test_forecast = pred1.predicted_mean.round()
  test_forecast[test_forecast<0] = 0

  
  return (train_forecast,test_forecast)

def print_sarima_results(train_data,test_data,train_forecast,test_forecast):
  print('Train Mean Absolute Error:     ', mean_absolute_error(train_data , train_forecast))
  print('Train Root Mean Squared Error: ',np.sqrt(mean_squared_error(train_data , train_forecast)))
  print('Test Mean Absolute Error:      ', mean_absolute_error(test_data, test_forecast))
  print('Test Root Mean Squared Error:  ',np.sqrt(mean_squared_error(test_data, test_forecast)))

# normalize dataframe to make comparable with Neural Networks 
# normalize the data before training
train_mean = train.mean()
train_std = train.std()

train_df = (train - train_mean) / train_std
test_df = (test - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

# ================================ Apply SARIMAX  ===============================
#Applying Sarimax on demand data with exogenance variables on Cluster 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # get order and params for SARIMAX
    train_data = pd.DataFrame(train['cross_cafe'])
    train_exog = train.loc[:,['cloud_cover', 'humidity', 'pressure', 'temp_mean_past1h', 
         'wind_max_per10min_past1h', 'sun_last1h_glob', 'hour', 'day_of_month',
         'day_of_week', 'is_weekend']]
    test_data = pd.DataFrame(test['cross_cafe'])
    test_exog = test.loc[:,['cloud_cover', 'humidity', 'pressure', 'temp_mean_past1h', 
         'wind_max_per10min_past1h', 'sun_last1h_glob', 'hour', 'day_of_month',
         'day_of_week', 'is_weekend']]
    
    order,seasonal_order = get_sarima_params(train_data)
    train_forecast, test_forecast = apply_sarimax(train_data,train_exog,test_data,test_exog,order,seasonal_order)

    #order,seasonal_order = get_sarima_params(train_data)
    #train_forecast, test_forecast = apply_sarima(train.demand,train.iloc[:,0:-1],test.demand,test.iloc[:,0:-1],order,seasonal_order)
    print('SARIMA MODEL ORDERS ARE {} {} = '.format(order,seasonal_order))
    order = (1, 1, 1) 
    seasonal_order = (1, 0, 0, 24)
    mod = sm.tsa.statespace.SARIMAX(train_data,exog=train_exog,order=order,seasonal_order=seasonal_order)
    results = mod.fit()
  
    pred = results.get_prediction(start=train_data.index[0],end=train_data.index[-1],exog=train_exog,dynamic=False)
    train_forecast = pred.predicted_mean.round()
    train_forecast[train_forecast<0] = 0


    pred1 = results.get_prediction(start=test_data.index[0],end=test_data.index[-1],exog=test_exog,dynamic=False)
    test_forecast = pred1.predicted_mean.round()
    test_forecast[test_forecast<0] = 0


    print_sarima_results(train_data,test_data,train_forecast, test_forecast)


# ==================== visualise forecast results =========================
result_df = pd.DataFrame(test)
result_df['predicted_value'] = test_forecast

plt.plot(result_df['cross_cafe'], label='original')
plt.plot(result_df['predicted_value'], label='predicted')
plt.title("SARIMAX evaluation on test data")
plt.legend();

from plotly.offline import init_notebook_mode, iplot
iplot([{
    'x': result_df.index,
    'y': result_df['cross_cafe'],
    'name': cross_cafe
}  for cross_cafe in result_df.loc[:,['cross_cafe','predicted_value']].columns])


