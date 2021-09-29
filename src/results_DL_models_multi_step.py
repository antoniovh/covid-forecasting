# #### Copyright (c) 2021 Spanish National Research Council

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Time series forecasting
#
# This tutorial is based on the Tensorflow tutorial [Time series forecasting]
# (https://www.tensorflow.org/tutorials/structured_data/time_series?hl=en#multi-step_models).
#
# The aim of this notebook is to forecast COVID-19 time series using TensorFlow. The models are Convolutional and
# Recurrent Neural Networks (CNNs and RNNs).
#
# Concepts:
# * **Univariate Time Series**: These are datasets where only a single variable is observed at each time, such as
# temperature each hour. The example in the previous section is a univariate time series dataset.
# * **Multivariate Time Series**: These are datasets where two or more variables are observed at each time.
# * **One-Step Forecast**: This is where the next time step (t+1) is predicted.
# * **Multi-Step Forecast**: This is where two or more future time steps are to be predicted.
#
# ## Index
#
# [Forecast multiple steps](#i1):
#   * [Single-shot](#i2): Make the predictions all at once.
#     1. [Linear model](#linear)
#     2. [Dense model](#dense)
#     3. [Feed Forward Neural Network](#ffnn)
#     4. [Convolutional Neural Network (CNN)](#CNN)
#     5. [Recurrent Neural Network (RNN)](#RNN)
#   * [Autoregressive](#i3): Make one prediction at a time and feed the output back to the model.
#     1. [Recurrent Neural Network (RNN)](#RNN)
#
# ## Setup

import os
from IPython.display import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import random
from time_series_tools import WindowGenerator, compile_and_fit
from models import linear_model, dense_model, ffnn, lstm_model, conv_model, FeedBack
from visualize import plot_history
from utils import Paths

random.seed(-123)
# Paths
PATHS = Paths()
data_path = PATHS.processed / 'covid' / 'region' / 'all_data_cantb.csv'
# To save figures
figures_path = PATHS.base / 'figures' / 'dl_model'
figures_path.exists() or os.makedirs(figures_path)
# To save trained models
models_path = PATHS.models

mpl.rcParams['figure.figsize'] = (14, 9)
mpl.rcParams['axes.grid'] = False


# ## COVID-19 dataset
#
# This project uses a covid-19 data from Cantabria (Spain) obtained from:
#
# 1. [Cantabria Health Service](https://www.scsalud.es/coronavirus).
# 2. [Cantabria Institute of Statistics](https://www.icane.es/covid19/dashboard/home/home) (ICANE).
# 3. [Ministry of Health, Consumer Affairs and Social Welfare](https://www.mscbs.gob.es).
#
# To generate and process regional COVID-19 data from scratch use:
# ```bash
# python src/data_tfm.py
# python src/data_tfm.py --files covid
# ```
# The dataset to be used is located in ./data/processed/covid/region/all_data_cantb.csv. This dataset contains 20
# different features such as daily positive cases, daily deaths, hospital and ICU bed occupancy, daily pcr tests,
# 7-day and 14-day cumulative incidence and the number of people vaccinated. These were collected every day, beginning
# in 29 February 2020.
#
# If you are running this notebook on Google Collab you must upload the file and execute the following code cell:

# Load data
df = pd.read_csv(data_path, sep=',').copy()
df = df.set_index('date')

# Here is the evolution of a few features over time:
plot_cols = ['daily_cases', 'daily_deaths', 'hospital_occ', 'icu_occ', 'daily_total_tests']
plot_features = df[plot_cols]
plot_features.index = df.index
_ = plot_features.plot(subplots=True)
plt.suptitle('Relevant variables')
fig_path = figures_path / 'relevant_variables.svg'
plt.savefig(fig_path)
plt.close()

# ## Inspect
# Next, look at the statistics of the dataset. All variables must be positive and `date` column should be date type.
# The processing of the data has been done before when the dataset has been generated.
df.describe().transpose()

# ## Split the data
# I will use a `(80%, 15%, 5%)` split for the training, validation, and test sets. The data is **not** being randomly
# shuffled before splitting because:
# 1. It ensures that chopping the data into windows of consecutive samples is still possible.
# 2. It ensures that the validation/test results are more realistic, being evaluated on the data collected after the
#    model was trained.
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.8)]
val_df = df[int(n*0.8):int(n*0.95)]
test_df = df[int(n*0.95):]
num_features = df.shape[1]

# ## Normalize the data
# Before training a neural network it is important to scale features. Normalization is a common way of doing this
# scaling: subtract the mean and divide by the standard deviation of each feature. The mean and standard deviation
# should only be computed using the training data so that the models have no access to the values in the validation
# and test sets.
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(30, 10))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.close()

# ## Data windowing
# Predictions will be based on a window of consecutive samples from the data.
#
# The main features of the input windows are:
#
# - The width (number of time steps) of the input and label windows.
# - The time offset between them.
# - Which features are used as inputs, labels, or both.
#
# The variety of models (including Linear, dense, CNN and RNN models) can be use for:
#
# - *Single-output*, and *multi-output* predictions.
# - *Single-time-step* and *multi-time-step* predictions.
#
# The `WindowGenerator` class handles the indexes and offsets, splits windows of features into `(features, labels)`
# pairs and converts to `tf.data.Dataset`s the input DataFrames for training, evaluation, and test. The  `__init__`
# method includes:
# * `input_width`: Size of the history window.
# * `label_width`: Number of consecutive instants to predict.
# * `shift`: Size of the time window from the history window to the last time instant you want to predict.

# Example of one prediction 15 days into the future, given 30 days of history of the target variable daily_cases and
# daily_deaths.
w1 = WindowGenerator(input_width=30,
                     label_width=1,
                     shift=15,
                     train_df=train_df,
                     val_df=val_df,
                     test_df=test_df,
                     label_columns=['daily_cases', 'daily_deaths'])

# Example: Stack three time slices:
example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                           np.array(train_df[100:100+w1.total_window_size]),
                           np.array(train_df[200:200+w1.total_window_size])])
w1.split_window(example_window)
w1.plot(plot_col=['daily_cases', 'daily_deaths'])
plt.close()
# The code above took a batch of three 45-time step windows with 18 features at each time step. It splits them into a
# batch of 30-time step 18-feature inputs, and a 1-time step 1-feature label.


# ## Multi-step models
# These models will make **multiple time step predictions**.
#
# In a multi-step prediction, the model needs to learn to predict a range of future values. Thus, unlike a single step
# model, where only a single future point is predicted, a multi-step model predicts a sequence of the future values.
#
# There are two rough approaches to this:
#
# 1. Single shot predictions where the entire time series is predicted at once.
# 2. Autoregressive predictions where the model only makes single step predictions and its output is fed back as its
# input.
#
# In this section all the models will predict all the features across all output time steps.


# The models will learn to predict 14 days into the future, given 30 days of the past.
OUT_STEPS = 14
multi_window = WindowGenerator(input_width=30,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df)
# Example for some predictions when the model is trained
r1 = random.randint(1, len(train_df) - multi_window.total_window_size)
r2 = random.randint(1, len(train_df) - multi_window.total_window_size)
r3 = random.randint(1, len(train_df) - multi_window.total_window_size)

example_window = tf.stack([np.array(train_df[r1:r1 + multi_window.total_window_size]),
                           np.array(train_df[r2:r2 + multi_window.total_window_size]),
                           np.array(train_df[r3:r3 + multi_window.total_window_size])])
multi_window._example = multi_window.split_window(example_window)
multi_window.plot(plot_col=['daily_cases', 'daily_deaths', 'icu_occ'])
plt.close()


# ### Single-shot models
# One high-level approach to this problem is to use a "single-shot" model, where the model makes the entire sequence
# prediction in a single step. This can be implemented efficiently as a `tf.keras.layers.Dense` with
# `out_step*features` output units. The model just needs to reshape that output to the required `(out_step, features)`.

multi_val_performance = {}
multi_performance = {}

# #### Model 1: Linear
# A simple linear model based on the last input time step does better than either baseline, but is underpowered.
# The model needs to predict `OUTPUT_STEPS` time steps, from a single input time step with a linear projection.
# It can only capture a low-dimensional slice of the behavior, likely based mainly on the time of month and year.
Image(url="https://www.tensorflow.org/tutorials/structured_data/images/multistep_dense.png")

# Create model
multi_linear_model = linear_model(out_steps=OUT_STEPS,
                                  num_features=num_features)
# Train and evaluate model
history = compile_and_fit(multi_linear_model, multi_window, max_epoch=100)
plot_history(history, model_name='Linear NN model')
fig_path = figures_path / 'linear_nn_history.svg'
plt.savefig(fig_path)
plt.close()
# IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
# Plot example predictions
multi_window.plot(multi_linear_model, plot_col=['daily_cases', 'icu_occ'])
fig_path = figures_path / 'linear_nn_forecast.svg'
plt.savefig(fig_path)
plt.close()
# Save model
m_path = models_path / 'linear_nn.h5'
multi_linear_model.save(m_path)


# #### Model 2: Dense
# Adding a `tf.keras.layers.Dense` between the input and output gives the linear model more power, but is still only
# based on a single input time step.

# Create model
multi_dense_model = dense_model(out_steps=OUT_STEPS,
                                dense_units=512,
                                num_features=num_features)
# Train and evaluate model
history = compile_and_fit(multi_dense_model, multi_window, max_epoch=100)
# Plot and save history
plot_history(history, model_name='Dense NN model')
fig_path = figures_path / 'dense_nn_history.svg'
plt.savefig(fig_path)
plt.close()
# IPython.display.clear_output()
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
# Plot example predictions
multi_window.plot(multi_dense_model, plot_col=['daily_cases', 'icu_occ'])
fig_path = figures_path / 'dense_nn_forecast.svg'
plt.savefig(fig_path)
plt.close()
# Save model
m_path = models_path / 'dense_nn.h5'
multi_dense_model.save(m_path)


# #### Model 2 Advance: Feed Forward Neural Network

# Create model
multi_ffnn_model = ffnn(hidden_layers=2,
                        hidden_cells=500,
                        out_steps=OUT_STEPS,
                        dropout_rate=0,
                        num_features=num_features)
# Train and evaluate model
history = compile_and_fit(multi_ffnn_model, multi_window, max_epoch=100)
# Plot and save history
plot_history(history, model_name='FFNN model')
fig_path = figures_path / 'ffnn_history.svg'
plt.savefig(fig_path)
plt.close()
# IPython.display.clear_output()
multi_val_performance['FFNN'] = multi_ffnn_model.evaluate(multi_window.val)
multi_performance['FFNN'] = multi_ffnn_model.evaluate(multi_window.test, verbose=0)
# Plot example predictions
multi_window.plot(multi_ffnn_model, plot_col=['daily_cases', 'icu_occ'])
fig_path = figures_path / 'ffnn_forecast.svg'
plt.savefig(fig_path)
plt.close()
# Save model
m_path = models_path / 'ffnn.h5'
multi_ffnn_model.save(m_path)


# #### CNN
# A convolutional model makes predictions based on a fixed-width history, which may lead to better performance than the
# dense model since it can see how things are changing over time.
Image(url="https://www.tensorflow.org/tutorials/structured_data/images/multistep_conv.png")
# For the CNN, 20 past days (`CONV_WIDTH`) and 30 epochs will be used to predict 14 future days.

# Create model
CONV_WIDTH = 20
multi_conv_model = conv_model(conv_width=CONV_WIDTH,
                              out_steps=OUT_STEPS,
                              num_features=num_features)
# Train and evaluate model
history = compile_and_fit(multi_conv_model, multi_window, max_epoch=100)
# Plot and save history
plot_history(history, model_name='CNN model')
fig_path = figures_path / 'cnn_history.svg'
plt.savefig(fig_path)
plt.close()
# IPython.display.clear_output()
multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
# Plot example predictions
multi_window.plot(multi_conv_model, plot_col=['daily_cases', 'icu_occ'])
fig_path = figures_path / 'cnn_forecast.svg'
plt.savefig(fig_path)
plt.close()
# Save model
m_path = models_path / 'cnn.h5'
multi_conv_model.save(m_path)


# #### RNN
# A recurrent model can learn to use a long history of inputs, if it's relevant to the predictions the model is making.
# Here the model will accumulate internal state for 30 days, before making a single prediction for the next 14 days.
#
# In this single-shot format, the LSTM only needs to produce an output at the last time step, so set
# `return_sequences=False` in `tf.keras.layers.LSTM`.
Image(url="https://www.tensorflow.org/tutorials/structured_data/images/multistep_lstm.png")

# Create model
multi_lstm_model = lstm_model(out_steps=OUT_STEPS,
                              cells_lstm=32,
                              hidden_layers=2,
                              hidden_cells=500,
                              num_features=num_features,
                              dropout_rate=0.05)
# Train and evaluate model
history = compile_and_fit(multi_lstm_model, multi_window, max_epoch=100)
# Plot and save history
plot_history(history, model_name='LSTM model')
fig_path = figures_path / 'lstm_history.svg'
plt.savefig(fig_path)
plt.close()
# IPython.display.clear_output()
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
# Plot example predictions
multi_window.plot(multi_lstm_model, plot_col=['daily_cases', 'icu_occ'])
fig_path = figures_path / 'lstm_forecast.svg'
plt.savefig(fig_path)
plt.close()
# Save model
m_path = models_path / 'lstm.h5'
multi_lstm_model.save(m_path)


# ### Advance: Autoregressive model
# The above models all predict the entire output sequence in a single step. In some cases it may be helpful for the
# model to decompose this prediction into individual time steps. Then, each model's output can be fed back into itself
# at each step and predictions can be made conditioned on the previous one.
# One clear advantage to this style of model is that it can be set up to produce output with a varying length.
# You could take any of the single-step multi-output models trained in the first half and run in an autoregressive
# feedback loop, but here you'll focus on building a model that's been explicitly trained to do that.
Image(url="https://www.tensorflow.org/tutorials/structured_data/images/multistep_autoregressive.png")
# ##### RNN
# Only an autoregressive RNN model will be built, but this pattern could be applied to any model that was designed to
# output a single time step. The model will have the same basic form as the single-step LSTM models from earlier: a
# `tf.keras.layers.LSTM` layer followed by a `tf.keras.layers.Dense` layer that converts the `LSTM` layer's outputs to
# model predictions. A `tf.keras.layers.LSTM` is a `tf.keras.layers.LSTMCell` wrapped in the higher level
# `tf.keras.layers.RNN` that manages the state and sequence result. In this case, the model has to manually manage the
# inputs for each step, so it uses `tf.keras.layers.LSTMCell` directly for the lower level, single time step interface.

# Create and train the model.
feedback_model = FeedBack(units=32,
                          out_steps=OUT_STEPS,
                          num_features=num_features)
history = compile_and_fit(feedback_model, multi_window, max_epoch=100)
# Plot and save history
plot_history(history, model_name='Autoregressive LSTM model')
fig_path = figures_path / 'ar_LSTM_history.svg'
plt.savefig(fig_path)
plt.close()
# IPython.display.clear_output()
multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
# Plot example predictions
multi_window.plot(feedback_model, plot_col=['daily_cases', 'icu_occ'])
fig_path = figures_path / 'ar_lstm_forecast.svg'
plt.savefig(fig_path)
plt.close()


# ### Performance
multi_val_df = pd.DataFrame.from_dict(data=multi_val_performance,
                                      orient='index',
                                      columns=['Loss', 'MAE'])
print("Validation MAE: \n", multi_val_df)
# print(multi_performance)

x = np.arange(len(multi_val_df))
width = 0.3
metric_name = 'mean_absolute_error'
plt.ylabel(f'{metric_name}', fontsize=13)
plt.xlabel('Models')
plt.title('Validation dataset', fontsize=16)

loss_metric = multi_val_df['Loss'].values
mae_metric = multi_val_df['MAE'].values
plt.bar(x - 0.17, loss_metric, width, label='loss')
plt.bar(x + 0.17, mae_metric, width, label='mae')
plt.xticks(ticks=x,
           labels=multi_val_performance.keys(),
           rotation=45)
_ = plt.legend()
fig_path = figures_path / 'mae_dl_models.svg'
plt.savefig(fig_path)
plt.close()
