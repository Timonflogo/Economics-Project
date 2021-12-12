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
rcParams['figure.figsize'] = 22, 10

# Load data
df_input = pd.read_csv('C:/Users/timon/Documents/GitHub/Economics-Project/Data/weather-energy-data.csv')
df_input = df_input.iloc[:,1:]

# plot heatmap
# df_input = df_input.reset_index()
# get correlations
df_input_corr = df_input.corr()
# create mask
mask = np.triu(np.ones_like(df_input_corr, dtype=np.bool))

sns.heatmap(df_input_corr, mask=mask, annot=True, fmt=".2f", cmap='Blues',
            vmin=-1, vmax=1, cbar_kws={"shrink": .8})


# inspect time series 
sns.lineplot(x='Datetime', y='kWh', data=df_input)

    
# create daily, weekly, and yearly signals 
# Set datetime as index
df_input = df_input.set_index('Datetime')
date_time = df_input.pop('Datetime')
timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24*60*60
week = 24*60*60*7
year = (365.2425)*day

df_input['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df_input['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))


plt.plot(np.array(df_input['Day sin'])[:25])
plt.plot(np.array(df_input['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')