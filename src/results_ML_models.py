# ##### Copyright (c) 2021 Spanish National Research Council
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


# # Machine Learning Models
# 
# The aim of this notebook is to forecast COVID-19 time series using ML models. These models are 
# ARIMA, Random Forest, Gradient Boosting for regression, SVR, KRR, KNN, GLM.
# 
# Target variables are:
# 1. Daily positive cases
# 2. Daily deaths
# 3. Hospital bed occupancy
# 4. ICU bed occupancy
# 
# The target variable used in this notebook will be `daily_cases`.

# ## Models
# 
# - [GLM](#GLM)
# - [Random Forest](#RF)
# - [Support Vector Regression](#SVR)
# - [Kernel Ridge Regression](#KRR)
# - [KNN](#KNN)
# - [Gradient Boosting Regressor](#GBR)
# - [ARIMA](#ARIMA)
# 
# All models except the ARIMA model are not autoregressive, so it necessary to modify the target variable to create
# lagged observations in order to predict with these models at $n$ future instants in time. These models will try to
# find dependent relationships between the target at time $t+n$ and the features at time $t$. In contrast ARIMA model
# will try to use the dependent relationship between an observation and some number of lagged observations of the
# target variable.

# ## Setup
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import time
import warnings
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
from itertools import product
from utils import Paths
from train import make_splits, normalize, model_summary, my_map
from models import evaluate_arima_model, arima_model, multi_step_arima, autoregressive_arima
from visualize import model_plot
import pickle

warnings.filterwarnings("ignore")


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
# python src/data.py
# python src/data.py --files covid
# ```
# The dataset to be used is located in ./data/processed/covid/region/all_data_cantb.csv.
# This dataset contains 20 different features such as daily positive cases, daily deaths, hospital and
# ICU bed occupancy, daily pcr tests, 7-day and 14-day cumulative incidence and the number of people vaccinated.
# These were collected every day, beginning in 29 February 2020.

# Paths.
PATHS = Paths()
data_path = PATHS.processed / 'covid' / 'region' / 'all_data_cantb.csv'
df = pd.read_csv(data_path, sep=',').copy()
df = df.set_index('date')
# To save figures
figures_path = PATHS.base / 'figures' / 'ml_models'
figures_path.exists() or os.makedirs(figures_path)
# To save trained models
models_path = PATHS.models

# Plot target variables.
mpl.rcParams['figure.figsize'] = (14, 9)
mpl.rcParams['axes.grid'] = False
plot_cols = ['daily_cases', 'daily_deaths', 'hospital_occ', 'icu_occ']
plot_features = df[plot_cols]
plot_features.index = df.index
_ = plot_features.plot(subplots=True)
fig_path = figures_path / 'target_variables.svg'
plt.savefig(fig_path)
plt.close()


# ## Inspect
# Next, look at the statistics of the dataset. All variables must be positive.

print(df.describe().transpose())

# The processing of the data has been done before when the dataset has been generated.

# ## Split and normalize the data
# The data is **not** being randomly shuffled before splitting because:
# 
# 1. It ensures that chopping the data into windows of consecutive samples is still possible.
# 2. It ensures that the validation/test results are more realistic, being evaluated on the data collected after the
# model was trained.
# 
# Normalization is a common way of doing this scaling: subtract the mean and divide by the standard deviation of each
# feature.

# Split and define target variable
split_df = make_splits(df, target='daily_cases', time_step=7, ntrain=0.85, nval=0.1, ntest=0.05)

# Normalize data
split_ndf, norm = normalize(split_df)

# Plot distribution of features
df_std = split_ndf['train']['X']
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(30, 10))
plt.title('Distribution of features')
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(split_ndf['train']['X'].keys(), rotation=90)
fig_path = figures_path / 'features_distribution.svg'
plt.savefig(fig_path)
plt.close()
# Metrics
metrics = {}

# ## Models
#
# ### GLM<a name="GLM"></a>
# 
# Generalized Linear Models currently supports estimations using the one-parameter exponential families.

# Data
split_df = make_splits(df, target='daily_cases', time_step=7, ntrain=0.85, nval=0.1, ntest=0.05)
split_ndf, _ = normalize(split_df)
# Features
X_train = split_ndf['train']['X']
# Target
y_train = split_ndf['train']['y']
# Train model
t0 = time.time()
glm_model = sm.GLM(y_train, X_train, family=sm.families.Gaussian())
glm_model = glm_model.fit()
glm_model.name = 'GLM'
print(f'Elapsed time to train {glm_model.name}: ', time.time() - t0, '(s)')
# glm_model.summary()
# Metrics
_, glm_metrics = model_summary(glm_model, split_ndf)
metrics['GLM'] = glm_metrics
print("GLM metrics:\n", glm_metrics)
# Graph
model_plot(glm_model, split_ndf)
fig_path = figures_path / 'glm.svg'
plt.savefig(fig_path)
plt.close()
# Save the trained model as a pickle string.
m_path = models_path / 'glm.sav'
pickle.dump(glm_model, open(m_path, 'wb'))


# ### Random Forest<a name="RF"></a>
# 
# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the
# dataset and uses averaging to improve the predictive accuracy and control over-fitting.

# Data
split_df = make_splits(df, target='daily_cases', time_step=7, ntrain=0.85, nval=0.1, ntest=0.05)
split_ndf, _ = normalize(split_df)
# Features
X_train = split_ndf['train']['X']
# Target
y_train = split_ndf['train']['y']
# Parameters
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 80, 90, 100, 110],
    'max_features': [1, 2, 3, 4],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [4, 6, 8, 10, 12],
    'n_estimators': [5, 10, 50, 100, 200, 300, 1000]}
# Create model
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           # n_jobs=-1,
                           cv=3,
                           verbose=1)
# Search best model
t0 = time.time()
models = grid_search.fit(X_train, y_train)
print('Elapsed time to search the best parameters: ', time.time() - t0, '(s)')
# Best parameters for our model
best_grid = models.best_params_
print("Best parameters for Random Forest model:\n", best_grid)
# Fit best model
t0 = time.time()
rf_model = RandomForestRegressor(n_estimators=5,
                                 criterion='mae', 
                                 max_depth=20, 
                                 max_features=3, 
                                 min_samples_leaf=3, 
                                 min_samples_split=10, 
                                 bootstrap=False)
rf_model = rf_model.fit(X_train, y_train)
rf_model.name = 'Random Forest'
print(f'Elapsed time to train {rf_model.name}: ', time.time() - t0, '(s)')
# Metrics
_, rf_metrics = model_summary(rf_model, split_ndf)
metrics['RF'] = rf_metrics
print("Random Forest metrics:\n", rf_metrics)
# Graph
model_plot(rf_model, split_ndf)
fig_path = figures_path / 'random_forest_model.svg'
plt.savefig(fig_path)
plt.close()
# Save the trained model as a pickle string.
m_path = models_path / 'random_forest_model.sav'
pickle.dump(rf_model, open(m_path, 'wb'))


# ### Support Vector Regression (SVR)<a name="SVR"></a>
# SVR is built based on the concept of Support Vector Machine (SVM). SVR gives us the flexibility to define how much
# error is acceptable in our model. The free parameters in the model are C and $\epsilon$ to choose how tolerant the
# model will be of errors, both through an acceptable error margin and through tuning our tolerance of falling outside
# that acceptable error rate.

# Data
split_df = make_splits(df, target='daily_cases', time_step=7, ntrain=0.8, nval=0.1, ntest=0.1)
split_ndf, _ = normalize(split_df)
# Features
X_train = split_ndf['train']['X']
# Target
y_train = split_ndf['train']['y']
# Parameters
param_grid = {"C": [1e-1, 1e0, 1e1, 1e2],
              "epsilon": [0.05, 0.1, 0.5, 1, 5, 10, 20],
              "gamma": np.logspace(-2, 2, 5),
              "kernel": ['rbf', 'sigmoid']}
# Create the model
svr = SVR(kernel='rbf')
grid_search = GridSearchCV(estimator=svr,
                           param_grid=param_grid,
                           # n_jobs=-1,
                           cv=5,
                           verbose=1)
# Search best model
t0 = time.time()
models = grid_search.fit(X_train, y_train)
print('Elapsed time to search the best parameters: ', time.time() - t0, '(s)')
# Best parameters for our model
best_grid = models.best_params_
print("Best parameters for SVR model:\n", best_grid)
# Fit best model
t0 = time.time()
svr_model = SVR(kernel='rbf',
                gamma=0.01,
                epsilon=5,
                C=100)
svr_model = svr_model.fit(X_train, y_train)
print('Elapsed time: ', time.time() - t0, '(s)')
svr_model.name = 'Support Vector Regression'
print(f'Elapsed time to train {svr_model.name}: ', time.time() - t0, '(s)')
# Metrics
_, svr_metrics = model_summary(svr_model, split_ndf)
metrics['SVR'] = svr_metrics
print("SVR metrics:\n", svr_metrics)
# Graph
model_plot(svr_model, split_ndf)
fig_path = figures_path / 'svr_model.svg'
plt.savefig(fig_path)
plt.close()
# Save the trained model as a pickle string.
m_path = models_path / 'svr_model.sav'
pickle.dump(svr_model, open(m_path, 'wb'))


# ### Kernel Ridge Regression (KRR)<a name="KRR"></a>
# 
# Kernel ridge regression (KRR) combines ridge regression (linear least squares with L2-norm regularization) with
# the kernel trick.
# 
# Both KRR and SVR learn a non-linear function by applying the kernel trick. They differ in the loss functions
# (ridge verse epsilon-insensitive loss). In contrast to SVR, fitting a KRR can be done in closed-form and is typically
# faster for medium-sized datasets. On the other hand, the learned model is non-sparse and thus slower than SVR at
# prediction-time.

# Data
split_df = make_splits(df, target='daily_cases', time_step=7, ntrain=0.8, nval=0.1, ntest=0.1)
split_ndf, _ = normalize(split_df)
# Features
X_train = split_ndf['train']['X']
# Target
y_train = split_ndf['train']['y']
# Parameters
param_grid = {"alpha": [30, 10, 1e0, 0.1, 1e-2, 1e-3],
              "gamma": np.logspace(-4, 2, 7),
              "kernel": ['rbf', 'sigmoid'],
              "coef0": [1, 2, 5, 10]}
# Create the model
krr = KernelRidge(kernel='sigmoid')
grid_search = GridSearchCV(estimator=krr,
                           param_grid=param_grid,
                           # n_jobs=-1,
                           cv=5,
                           verbose=1)
# Search best model
t0 = time.time()
models = grid_search.fit(X_train, y_train)
print('Elapsed time to search the best parameters: ', time.time() - t0, '(s)')
# Best parameters for our model
best_grid = models.best_params_
print("Best parameters for KRR model:\n", best_grid)
# Best model
t0 = time.time()
krr_model = KernelRidge(kernel='rbf',
                        alpha=0.1,
                        gamma=0.01)
krr_model = krr_model.fit(X_train, y_train)
krr_model.name = 'KRR'
print(f'Elapsed time to train {krr_model.name}: ', time.time() - t0, '(s)')
# Metrics
_, krr_metrics = model_summary(krr_model, split_ndf)
metrics['KRR'] = krr_metrics
print("KRR metrics:\n", krr_metrics)
# Graph
model_plot(krr_model, split_ndf)
fig_path = figures_path / 'krr_model.svg'
plt.savefig(fig_path)
plt.close()
# Save the trained model as a pickle string.
m_path = models_path / 'krr_model.sav'
pickle.dump(krr_model, open(m_path, 'wb'))


# ### KNN<a name="KNN"></a>
# 
# K-Nearest-Neighbor algorithm can be used for classification and regression. It uses feature similarity to predict the
# values of any new data points.

# Data
split_df = make_splits(df, target='daily_cases', time_step=7, ntrain=0.8, nval=0.1, ntest=0.1)
split_ndf, _ = normalize(split_df)
# Features
X_train = split_ndf['train']['X']
# Target
y_train = split_ndf['train']['y']
# Parameters
param_grid = {"n_neighbors": list(range(1, 30, 1))}
# Create the model
knn = KNeighborsRegressor()
grid_search = GridSearchCV(estimator=knn,
                           param_grid=param_grid,
                           # n_jobs=-1,
                           cv=5,
                           verbose=1)
# Search best model
t0 = time.time()
models = grid_search.fit(X_train, y_train)
print('Elapsed time to search the best parameters: ', time.time() - t0, '(s)')
# Best parameters for our model
best_grid = models.best_params_
print("Best parameters for K-NN model:\n", best_grid)
# Best model
t0 = time.time()
knn_model = KNeighborsRegressor(n_neighbors=19)
knn_model = knn_model.fit(X_train, y_train)
knn_model.name = 'K-NN'
print(f'Elapsed time to train {knn_model.name}: ', time.time() - t0, '(s)')
# Metrics
_, knn_metrics = model_summary(knn_model, split_ndf)
metrics['KNN'] = knn_metrics
print("K-NN metrics:\n", knn_metrics)
# Graph
model_plot(knn_model, split_ndf)
fig_path = figures_path / 'knn_model.svg'
plt.savefig(fig_path)
plt.close()
# Save the trained model as a pickle string.
m_path = models_path / 'knn_model.sav'
pickle.dump(knn_model, open(m_path, 'wb'))


# ### Gradient Boosting Regressor<a name="GBR"></a>
# 
# Gradient Boosting (GB) for regression builds an additive model in a forward stage-wise fashion. It allows for the
# optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative
# gradient of the given loss function.

# Data
split_df = make_splits(df, target='daily_cases', time_step=7, ntrain=0.8, nval=0.1, ntest=0.1)
split_ndf, _ = normalize(split_df)
# Features
X_train = split_ndf['train']['X']
# Target
y_train = split_ndf['train']['y']
# Parameters
param_grid = {"n_estimators": [5, 10, 50, 100, 200, 300, 1000],
              "learning_rate": [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
              "max_depth": [10, 20, 30, 40, 50, 80, 90, 100, 110]}
# Create the model
gb = GradientBoostingRegressor(random_state=0, loss='ls')
grid_search = GridSearchCV(estimator=gb,
                           param_grid=param_grid,
                           # n_jobs=-1,
                           cv=5,
                           verbose=1)
# Search best model
t0 = time.time()
models = grid_search.fit(X_train, y_train)
print('Elapsed time to search the best parameters: ', time.time() - t0, '(s)')
# Best parameters for our model
best_grid = models.best_params_
print("Best parameters for Gradient Boosting model:\n", best_grid)
# Best model
t0 = time.time()
gb_model = GradientBoostingRegressor(n_estimators=5,
                                     learning_rate=0.2,
                                     max_depth=10,
                                     random_state=0,
                                     loss='ls')
gb_model = gb_model.fit(X_train, y_train)
gb_model.name = 'Gradient Boosting'
print(f'Elapsed time to train {gb_model.name}: ', time.time() - t0, '(s)')
# Metrics
_, gb_metrics = model_summary(gb_model, split_ndf)
metrics['GB'] = gb_metrics
print("GB metrics:\n", gb_metrics)
# Graph
model_plot(gb_model, split_ndf)
fig_path = figures_path / 'gradient_boosting_model.svg'
plt.savefig(fig_path)
plt.close()
# Save the trained model as a pickle string.
m_path = models_path / 'gradient_boosting_model.sav'
pickle.dump(gb_model, open(m_path, 'wb'))


# ## Metrics of all models
it = 1
for metric in ['mse', 'mae']:
    plt.subplot(2, 1, it)
    x = np.arange(len(metrics))
    width = 0.3
    plt.ylabel(f'{metric} [{y_train.name}]', fontsize=13)
    plt.title('Models', fontsize=16)
    train_metric = [model.loc[metric, 'train'] for model in metrics.values()]
    val_metric = [model.loc[metric, 'val'] for model in metrics.values()]
    test_metric = [model.loc[metric, 'test'] for model in metrics.values()]
    
    plt.bar(x - width, train_metric, width, label='Train')
    plt.bar(x, val_metric, width, label='Validation')
    plt.bar(x + width, test_metric, width, label='Test')
    plt.xticks(ticks=x, labels=metrics.keys(), 
               rotation=45)
    _ = plt.legend()
    it += 1
fig_path = figures_path / 'metrics_ml_models.svg'
plt.savefig(fig_path)
plt.close()


# It seems that the Gradient Boosting model is the best to forecast the target variable with the features at $t-7$.

# ### ARIMA<a name="ARIMA"></a>

# An ARIMA model is a class of statistical models for analyzing and forecasting time series data. 
# * AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged
# observations.
# * I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at
# the previous time step) in order to make the time series stationary.
# * MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving
# average model applied to lagged observations.
# 
# The parameters of the ARIMA model are defined as follows:
# 
# * p: The number of lag observations included in the model, also called the lag order.
# * d: The number of times that the raw observations are differenced, also called the degree of differencing.
# * q: The size of the moving average window, also called the order of moving average.
# 
# A value of 0 can be used for a parameter, which indicates to not use that element of the model. This way, the ARIMA
# model can be configured to perform the function of an ARMA model, and even a simple AR, I, or MA model.

# Data with time_step=0
split_df = make_splits(df, target='daily_cases', time_step=0, ntrain=0.85, nval=0, ntest=0.15)
split_ndf, _ = normalize(split_df)
# Evaluated parameters
p = [0, 1, 2, 4, 6, 8, 10]
d = range(0, 3)
q = range(0, 3)
parameters = product(p, d, q)
parameters_list = list(parameters)
# Search best combination of parameters (p, d, q)
t0 = time.time()
result = list(my_map(evaluate_arima_model, split_ndf, tqdm(parameters_list)))
print('Elapsed time to search the best parameters: ', time.time() - t0, '(s)')
result = pd.DataFrame(result,
                      columns=['(p, d, p)', 'RMSE', 'MSE', 'MAE', 'MAPE', 'R2'])
result_sort = result.sort_values(by='RMSE', ascending=True).reset_index(drop=True)
result_sort = result_sort.dropna()
# Train best model
t0 = time.time()
arima_metrics = arima_model(split_ndf, (4, 1, 2), graph=True)
print(f'Elapsed time to train best ARIMA model', time.time() - t0, '(s)')
# Metrics
print("Best ARIMA model metrics:\n", arima_metrics)
# Graph
fig_path = figures_path / 'arima_model.svg'
plt.savefig(fig_path)
plt.close()

# The above predictions at each instant $t$ were calculated from the observations $(t=0,..., t-1)$ of the test dataset,
# i.e. a one-step forecast.

multi_arima_metrics, arima_m = multi_step_arima(split_ndf, (4, 1, 2), time_step=7, graph=True)
# Metrics
print("Multi step ARIMA metrics:\n", multi_arima_metrics)
# Graph
fig_path = figures_path / 'arima_forecast.svg'
plt.savefig(fig_path)
plt.close()
# Save the trained model as a pickle string.
m_path = models_path / 'arima_model.sav'
pickle.dump(arima_m, open(m_path, 'wb'))

# ARIMA model is useful for multi-step time series forecasting:
# * Single-shot: Make the predictions all at once.
# * Autoregressive: Make one prediction at a time and feed the output back to the model.

auto_arima_metrics = autoregressive_arima(split_ndf, (4, 1, 2), time_step=7, graph=True)
# Metrics
print("Autoregressive ARIMA metrics:\n", auto_arima_metrics)
# Graph
fig_path = figures_path / 'autoregressive_arima_forecast.svg'
plt.savefig(fig_path)
plt.close()
