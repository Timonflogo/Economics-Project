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
import tensorflow as tf

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

### Personal consumption data ----

# load personal consumption dataset
persCons = pd.read_csv('C:/Users/timon/Documents/GitHub/Economics-Project/Data/consumption-timon.csv')
persCons.info()
persCons.head()

# change danish to english 
persCons = persCons.rename(columns={"Mængde" : "Value", "Måleenhed" : "Measure"})

# add new column with hour of day and date
persCons["Day"] = persCons["Til dato"].str[:2]
persCons["Month"] = persCons["Til dato"].str[3:5]
persCons["Year"] = persCons["Til dato"].str[6:10]
persCons["Hour"] = persCons["Til dato"].str[-2:]

persCons["Date"] = persCons["Year"].astype(str) + "-" + persCons["Month"].astype(str) + "-" + persCons["Day"].astype(str)

# add minute and second format to hour column 
persCons["Hour"] = persCons["Hour"].astype(str) + ":00:00"

# combine date and hour 
persCons["Datetime"] = persCons["Date"].astype(str) + " " + persCons["Hour"].astype(str)

# subset persCons to include Datetime and measurements in Kwh only
persCons = persCons[["Datetime", "Value"]]

# Set Datetime as type datetime
persCons['Datetime'] = pd.to_datetime(persCons['Datetime'], infer_datetime_format=True)
# persCons = persCons.set_index('Datetime')
persCons.info()

# sort descending by date
persCons.sort_values(by=['Datetime'], inplace=True, ascending=True)

### Meteorological Observation data ----

# load metObs dataset
metObs = pd.read_csv('C:/Users/timon/Documents/GitHub/Economics-Project/Data/Weather-Data.csv')

# subset dataframe and only include values with high frequency
# delete the 3 first observations as they are only 1/2 of an hour
metObs = metObs.iloc[3:,0:14]
metObs = metObs.drop("StationID", axis=1)
metObs.info()

# change Datetime to datetype datetime
metObs['Time'] = pd.to_datetime(metObs['Time'], infer_datetime_format=True)

# set Datetime as index
# metObs = metObs.set_index('Time')
metObs.info()

# check for missing data 
metObs.isna().sum()
# cloud height has a lot of missing data and will be removed from the dataframe
metObs = metObs.drop("cloud_height", axis=1)

# replace NA values in dataframe 
# propagate last valid observation forward to next 
metObs.fillna(method='ffill', inplace = True)
metObs.isna().sum()

# sort descending by date
metObs.sort_values(by=['Time'], inplace=True, ascending=True)

# Aggregate data by hour
metObs = metObs.groupby(metObs.Time.floor('H')).mean()

# Add additional variables for seasonality
persCons['hour'] = persCons.Time.hour
# add day of month column 0 = first day of the month
persCons['day_of_month'] = persCons.index.day
# add day of week column 0 = Monday
persCons['day_of_week'] = persCons.index.dayofweek
# add month column
persCons['month'] = persCons.index.month
# add weekend column
persCons['is_weekend'] = ((persCons.index.dayofweek) // 5 == 1).astype(float)


# rename Time to Datetime 
metObs = metObs.rename(columns={"Time" : "Datetime"})

### Merge Meteorological Observation data with Personal consumption data ---

df = metObs.merge(metObs, how="outer")

df.isna().sum()

df = df.iloc[1398:,:]




