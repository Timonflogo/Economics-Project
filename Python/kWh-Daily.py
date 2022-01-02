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
rcParams['figure.figsize'] = 16, 6

# set random seed
random_seed = 20
np.random.seed(random_seed)

# load dataset
df_input = pd.read_csv('/Users/timongodt/Documents/GitHub/Economics-Project/Data/weather-energy-data-update.csv', index_col="Datetime", parse_dates=True).iloc[:,1:]
df_input = df_input['20181101':'20211031']

# prepare data
df = df_input[['kWh']]

df_input.isna().sum()
df_input.fillna(method='ffill', inplace = True)
df_input.isna().sum()


# resample by day
df_SD = df.resample('D').sum()

# Reduce dataframe to include only single variable 
df_SD = df_SD[['kWh']]
df_SD.index.freq = 'D'

# =========================== Exploratory Data Analysis ==============================
# plot heatmap
# df_input = df_input.reset_index()
# get correlations
# =============================================================================
# df_input_corr = df_input.corr()
# # create mask
# mask = np.triu(np.ones_like(df_input_corr, dtype=np.bool))
# 
# sns.heatmap(df_input_corr, mask=mask, annot=True, fmt=".2f", cmap='Blues',
#             vmin=-1, vmax=1, cbar_kws={"shrink": .8})
# =============================================================================

# # Regression
# X = df_input.iloc[:,1:]
# y = df_input['kWh']

# # encode categorical values
# X['hour'] = X['hour'].astype("category")
# X['day_of_week'] = X['day_of_week'].astype("category")
# X['is_weekend'] = X['is_weekend'].astype("category")
# X['day_of_month'] = X['day_of_month'].astype("category")
# X['month'] = X['month'].astype("category")
# X.info()

# model = sm.OLS(y, X).fit()
# predictions = model.predict(X)

# model.summary()

# dataframe for univariate 
df_input = df_SD

################ SARIMA Hourly forecasts  
# import libraries for time series analysis
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tools.eval_measures import mse,rmse     # for ETS Plots
from pmdarima import auto_arima  

# dataframe for SARIMAX
# resample by day
df_SD = df.resample('D').sum()

# Reduce dataframe to include only single variable 
df_SD = df_SD[['kWh']]
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

df = df_SD

# decompose series
result = seasonal_decompose(df['kWh'], model = "additive")
result.plot({'figure.figsize': (16,9)});
result.seasonal['20210801':'20211101'].plot() 

# add weekly dummy variables 
df['day_of_week'] = df.index.dayofweek
df_dummies = pd.get_dummies(df['day_of_week'])
df = pd.merge(df, df_dummies, how='left', left_index=True, right_index=True)
df.drop('day_of_week', inplace=True, axis=1)


# Train Test set split - we want to forecast 7 Days into the future so out test set should be at least one month 
len(df)
# we will go with a train-test split such that our test set represents 168 Hours worth of data
train1 =  df[:len(df)-7]
test1 = df[len(df)-7:]
len(df) == len(train1) + len(test1) # True

# forecast start and end
# obtain predicted results
start1 = len(train1)
end1 = len(train1)+len(test1)-1

# ------- WITHOUT Exogenous

# run Auto Arima to determine model without specified seasonality
#auto_arima(df['kWh'],seasonal=False).summary()
# SARIMAX(2, 0, 2)

# run Auto Arima to determine model without specified seasonality
#auto_arima(df['kWh'],seasonal=True,m=7).summary()
# SARIMAX(3, 0, 0)x(1, 0, [1], 7) 

df.plot()

# Fit ARIMA WITHOUT EXOGENOUS
model = SARIMAX(train1['kWh'],order=(2,0,2),enforce_invertibility=False)
results = model.fit()
results.summary()

# predict
predictions = results.predict(start=start1, end=end1).rename('ARIMA(2,0,2) Predictions')

# Fit ARIMA WITH EXOGENOUS
exog_train = train1.iloc[:,1:]
exog_forecast = test1.iloc[:,1:]
model3 = SARIMAX(train1['kWh'],exog=exog_train,order=(3,0,0),seasonal_order=(1,0,1,7),enforce_invertibility=False)
results3 = model3.fit()
results3.summary()

# predict
predictions3 = results3.predict(start=start1, end=end1, exog=exog_forecast).rename('ARIMAX(2,0,2)Predictions')


# Fit SARIMA WITHOUT EXOGENOUS
model1 = SARIMAX(train1['kWh'],order=(3,0,0),seasonal_order=(1,0,1,7),enforce_invertibility=False)
results1 = model1.fit()
results1.summary()

# predict
predictions1 = results1.predict(start=start1, end=end1).rename('SARIMA(3,0,0)(1,0,1,7) Predictions')

# Fit SARIMA WITH EXOGENOUS
exog_train = train1.iloc[:,1:]
exog_forecast = test1.iloc[:,1:]
model2 = SARIMAX(train1['kWh'],exog=exog_train,order=(3,0,0),seasonal_order=(1,0,1,7),enforce_invertibility=False)
results2 = model1.fit()
results2.summary()

# predict
predictions2 = results1.predict(start=start1, end=end1, exog=exog_forecast).rename('SARIMA(3,0,0)(1,0,1,7) Predictions')



# plot predictions
title='Electricity Demand Forecast Daily'
ylabel='kWh'
xlabel=''

ax = test1['kWh'].plot(legend=True,figsize=(12,4),title=title)
#predictions.plot(legend=True)
predictions1.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# Evaluate model performance
error1 = mse(test1['kWh'], predictions)
error2 = rmse(test1['kWh'], predictions)
error3 = mse(test1['kWh'], predictions1)
error4 = rmse(test1['kWh'], predictions1)
error5 = mse(test1['kWh'], predictions2)
error6 = rmse(test1['kWh'], predictions2)
error7 = mse(test1['kWh'], predictions3)
error8 = rmse(test1['kWh'], predictions3)

# MAPE
from sklearn.metrics import mean_absolute_percentage_error as maperror
error9 = maperror(test1['kWh'], predictions)
error10 = maperror(test1['kWh'], predictions1)

print(f'ARIMA(2,0,2) MSE Error: {error1:11.10}')
print(f'ARIMA(2,0,2) RMSE Error: {error2:11.10}')
print(f'ARIMA(2,0,2) MSE Error: {error7:11.10}')
print(f'ARIMA(2,0,2) RMSE Error: {error8:11.10}')
print(f'SARIMA(3,0,0)(1,0,1,7) MSE Error: {error3:11.10}')
print(f'SARIMA(3,0,0)(1,0,1,7) RMSE Error: {error4:11.10}')
print(f'SARIMAX(3,0,0)(1,0,1,7) MSE Error: {error5:11.10}')
print(f'SARIMAX(3,0,0)(1,0,1,7) RMSE Error: {error6:11.10}')


 
  
# ================================== Multi Step Autoregressive LSTM forecasting model ========================

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
train =  df[:len(df)-7]
test = df[len(df)-7:]
len(df) == len(train) + len(test)

# Scale data
scaler = MinMaxScaler()

# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
# scaled_train = train
# scaled_test = test


# Let's define to get 168 Days back wbich represents one week and then predict the next week out
n_input = 7
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
model.fit_generator(generator,epochs=10, shuffle=False)

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

# model evaluation 
from statsmodels.tools.eval_measures import mse,rmse 
error = mse(test['kWh'], test['Predictions'])
error








# ================================== Multi Step single shot forecasting models ========================

# ---------------------------------------------- split data -----------------------------
# split the data
# we will use a 70/20/10 split for training, validation, and test sets
column_indices = {name: i for i, name in enumerate(df_input.columns)}

n = len(df_input)
train_df = df_input[0:int(n*0.7)]
val_df = df_input[int(n*0.7):int(n*0.9)]
test_df = df_input[int(n*0.9):]

num_features = df_input.shape[1]
# ------------------------------------------ normalize data --------------------------

# # normalize the data before training
# train_mean = train_df.mean()
# train_std = train_df.std()

# train_df = (train_df - train_mean) / train_std
# val_df = (val_df - train_mean) / train_std
# test_df = (test_df - train_mean) / train_std

# # look at features after normalization
# df_std = (df_input - train_mean) / train_std
# df_std = df_std.melt(var_name='Column', value_name='Normalized')
# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
# _ = ax.set_xticklabels(df_input.keys(), rotation=90)
# # looks fine, but there are still some extreme values in our target variable

# ---------------------------- create window class for indexing and offsetting ---------------------
# create window class for 
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

def __repr__(self):
  return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

# --------------------- create function to handle label columns ----------------

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

# =============================================================================
# # =============================================================================
# # # ------------------- example of how the window generator function works ------------------
# # example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
# #                             np.array(train_df[100:100+w1.total_window_size]),
# #                             np.array(train_df[200:200+w1.total_window_size])])
# # 
# # 
# # example_inputs, example_labels = w1.split_window(example_window)
# # 
# # print('All shapes are: (batch, time, features)')
# # print(f'Window shape: {example_window.shape}')
# # print(f'Inputs shape: {example_inputs.shape}')
# # print(f'labels shape: {example_labels.shape}')
# # 
# # =============================================================================
# w1.example = example_inputs, example_labels
# =============================================================================

# -------------------------- define plot function for evaluation -------------

# plot window
def plot(self, model=None, plot_col='kWh', max_subplots=3, title = 'plot'):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.tight_layout(pad=1.0)
    plt.ylabel(f'{plot_col} [norm]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)
    plt.title(title)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot

# --------------- convert input dataframe to tf.data.dataset ---------

# create tf.data.dataset
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=4,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

# ---------------- add properties for accessing tf.data.dataset -----------

# The WindowGenerator object holds training, validation and test data. 
# Add properties for accessing them as tf.data.Datasets using the above make_dataset method. 
# Also add a standard example batch for easy access and plotting:
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# =============================================================================
# # Each element is an (inputs, label) pair
# w1.train.element_spec
# 
# for example_inputs, example_labels in w1.train.take(1):
#   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#   print(f'Labels shape (batch, time, features): {example_labels.shape}')
# =============================================================================
  
# ----------------------- define forecasting horizon --------------------------  
  
# define forecasting length
lags = 30
OUT_STEPS = 30
multi_window = WindowGenerator(input_width=lags,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=['kWh'])
# multi_window.plot()
# multi_window

# ------------------- create compile and fit -------------------------------
MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=10):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanSquaredError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping],
                      shuffle=False)
  return history


# =================== create multi-step baseline model ===========================
class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last_obs'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last_obs'] = last_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(last_baseline, title= 'Last observation (kWh) 30 Days')

# ----------------- instantiate and evaluate baseline model ------------------
class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat_win'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat_win'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(repeat_baseline, title = 'Previous window (kWh) 30 Days')

# ===================== Dense Neural Network =======================

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(16, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_dense_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_dense_model, title= 'Dense Neural Network (kWh) 30 Days')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Dense Loss 30 Days (kWh)")
plt.legend();

# ================== Conv NN model =============================
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(16, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model, title = "Convolutional Neural Network (kWh) 30 Days")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Conv loss 30 Days (kWh)")
plt.legend();

# =================== run LSTm model ===========================
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, activation = 'relu', return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.random_normal()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model, title = 'LSTM (kWh) 30 Days')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("LSTM loss 30 Days (kWh)")
plt.legend();

# -------------------plot evaluation metrics --------------

# =============================================================================
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='validation')
# plt.title("loss")
# plt.legend();
# 
# plt.plot(history.history['mean_absolute_error'], label='train')
# plt.plot(history.history['val_mean_absolute_error'],label='validation')
# plt.title('MAE')
# plt.legend();
# 
# =============================================================================

x = np.arange(len(multi_performance))
width = 0.3


metric_name = 'mean_absolute_error'
metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.title('MAE (kWh) 30 Days')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MSE (average over all times and outputs)')
_ = plt.legend()

for name, value in multi_performance.items():
  print(f'{name:8s}: {value[1]:0.4f}')
  
 