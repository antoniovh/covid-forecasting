from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape, Lambda, Conv1D
import tensorflow as tf
import statsmodels.api as sm
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import warnings
warnings.filterwarnings("ignore")


def linear_model(out_steps, num_features=18):
    """
    Create a multi-step linear model with Neural Networks.
    The neural network without any activation function in any of its layers is called a linear neural network.
    The neural network which has action functions like relu, sigmoid or tanh in any of its layer or even in more than
    one layer is called non-linear neural network.

    Args:
        out_steps: Time instants to be predicted (t_final = t + out_steps).
                    Must coincide with WindowGenerator.label_width value
        num_features: Number of features (columns) to use in the predictions.

    Returns: Linear neural network.

    """
    model = Sequential(name='Linear_Neural_Network')
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    model.add(Lambda(lambda x: x[:, -1:, :]))
    model.add(Dense(out_steps*num_features,
                    kernel_initializer=tf.initializers.zeros()))
    # Shape => [batch, out_steps, features]
    model.add(Reshape([out_steps, num_features]))
    return model


def dense_model(out_steps, dense_units=512, num_features=18):
    """
    Create a multi-step Deep Learning Model with one hidden layer.

    Args:
        out_steps: Time instants to be predicted (t_final = t + out_steps).
                    Must coincide with WindowGenerator.label_width value
        dense_units:
        num_features: Number of features (columns) to use in the predictions.

    Returns: Linear neural network.

    """
    model = Sequential(name='Dense_Neural_Network')
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    model.add(Lambda(lambda x: x[:, -1:, :]))
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(dense_units, activation='relu'),
    # Shape => [batch, out_steps*features]
    model.add(Dense(out_steps*num_features,
                    kernel_initializer=tf.initializers.zeros()))
    # Shape => [batch, out_steps, features]
    model.add(Reshape([out_steps, num_features]))
    return model


def ffnn(out_steps, hidden_layers=2, hidden_cells=500, num_features=18, dropout_rate=0.05):
    """
    Create a custom Feed Forward Neural Network.
    Args:
        hidden_layers: Number of hidden layers.
        hidden_cells: Number of neurones in each hidden layer.
        out_steps: Time instants to be predicted (t_final = t + out_steps).
                    Must coincide with WindowGenerator.label_width value
        num_features: Number of features (columns) to use in the predictions.
        dropout_rate: (0-1). Percentage of randomly selected neurons or units (hidden and visible) that will be ignored
                    during training to prevent NN from overfitting.

    Returns: Feed Forward Neural Network model.

    """

    model = Sequential(name='Feed_Forward_Neural_Network')
    # # Take the last time step.
    # # Shape [batch, time, features] => [batch, 1, features]
    model.add(Lambda(lambda x: x[:, -1:, :]))
    for i in range(hidden_layers):
        model.add(Dense(hidden_cells,
                        activation="relu",
                        kernel_initializer="he_uniform",
                        name=f"dense{i}"))
        model.add(Dropout(dropout_rate))

    model.add(Dense(out_steps * num_features,
                    activation="relu"))
    model.add(Dense(out_steps * num_features,
                    kernel_initializer=tf.initializers.zeros()))
    # Shape => [batch, out_steps, features].
    model.add(Reshape([out_steps, num_features]))
    return model


def lstm_model(out_steps, cells_lstm=32, hidden_layers=0, hidden_cells=0, num_features=18, dropout_rate=0.05):
    """
    Long Short-Term Memory, a Recurrent Neural Network (RNN) model.
    This model makes multiple time step predictions, namely predicts a sequence of future values (out_steps).
    In this case to approach it, this model is a single-shot model: Single shot predictions where the entire time series
    is predicted at once.
    Args:
        out_steps: Time instants to be predicted (t_final = t + out_steps).
                    Must coincide with WindowGenerator.label_width value
        cells_lstm: Neurons in LSTM layer.
        hidden_layers: Number of hidden layers.
        hidden_cells: Number of neurones in each hidden layer.
        num_features: Number of features (columns) to use in the predictions.
        dropout_rate: (0-1). Percentage of randomly selected neurons or units (hidden and visible) that will be ignored
                    during training to prevent NN from overfitting.

    Returns: LSTM model

    """
    model = Sequential(name='Multi_step_LSTM')
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    model.add(LSTM(cells_lstm,
                   return_sequences=False))
    model.add(Dropout(dropout_rate))
    for i in range(hidden_layers):
        model.add(Dense(hidden_cells,
                        activation="relu",
                        kernel_initializer="he_uniform",
                        name=f"dense{i}"))
        model.add(Dropout(dropout_rate))
    # Shape => [batch, out_steps*features].
    model.add(Dense(out_steps * num_features,
                    kernel_initializer=tf.initializers.zeros()))
    # Shape => [batch, out_steps, features].
    model.add(Reshape([out_steps, num_features]))
    return model


def conv_model(conv_width, out_steps, num_features=18):
    """
    Convolutional Neural Network (CNN) model.
    This model makes multiple time step predictions, it predicts a sequence of future values (out_steps) in
    one single-shot.
    Args:
        conv_width: A convolutional model makes predictions based on a fixed-width history. It is the number of time
                    instants used to make predictions. It is also the size kernel argument of Conv1D layer.
        out_steps: Time instants to be predicted (t_final = t + out_steps).
                    Must coincide with WindowGenerator.label_width value
        num_features: Number of features (columns) to use in the predictions.

    Returns:

    """
    model = Sequential(name='Convolutional_Neural_Network')
    model.add(Lambda(lambda x: x[:, -conv_width:, :]))
    model.add(Conv1D(256, activation='relu', kernel_size=conv_width))
    model.add(Dense(out_steps*num_features,
                    kernel_initializer=tf.initializers.zeros()))
    model.add(Reshape([out_steps, num_features]))
    return model


class FeedBack(tf.keras.Model):
    """
    Autoregressive LSTM model.

    In some cases it may be helpful for the model to decompose the prediction into individual time steps.
    Then, each model's output can be fed back into itself at each step and predictions can be made conditioned on the
    previous one.

    Returns: Autoregressive predictions where the model only makes single step predictions and its output is feed back
            as its input.

    """
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True, name='Autoregressive_LSTM')
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x,
                                      states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


def evaluate_arima_model(splits, arima_order):
    """
    Evaluate an ARIMA model for a given order (p,d,q) and also forecast the next one time step.
    Train the model using t = (1, ..., t) and predict next time step (t+1).
    Then add (t+1) value from test dataset to history and fit again the model using t = (1, ..., t, t+1)
    to predict the next instant (t+2), and so on up to t=N where N=len(test)
    Finally, with the predictions and observables from the test dataset, the metrics MSE, MAE, MAPE y R2
    are calculated.
    Args:
        splits: Dictionary with training, validation and test data.
        arima_order: Tuple. Contains the argument p, d, q for ARIMA model.


    Returns: Errors from predictions using previous iterations.

    """
    try:
        # prepare training and test dataset (validation+test)
        train = splits['train']['y'].astype('float32')
        val_y = splits['val']['y'].astype('float32')
        test_y = splits['test']['y'].astype('float32')
        test = pd.concat([val_y, test_y])

        history = [x for x in train]
        predictions = list()

        for t in range(len(test)):
            model = sm.tsa.ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0, dis=-1)  # All cores
            yhat = model_fit.forecast()[0]  # Predict one step in future from the last value in history variable.
            predictions.append(yhat)
            history.append(test[t])

        # Metrics
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        mape = mean_absolute_percentage_error(test, predictions)
        r2 = r2_score(test, predictions)

        return arima_order, rmse, mse, mae, mape, r2

    except Exception as e:
        pass


def arima_model(splits, arima_order, graph=False):
    """
    Evaluate an ARIMA model for a given order (p,d,q) and also forecast the next one time step.
    Split data in train and test. Train the model using t = (1, ..., t) and predict next time step (t+1).
    Then add (t+1) value from test dataset to history and fit again the model using t = (1, ..., t, t+1).
    Then, it predicts for the next instant (t+2), and so on up to t=N where N=len(test)
    Finally, with the predictions and observables from the test dataset, the metrics MSE, MAE, MAPE y R2
    are calculated.
    Args:
        splits: Dictionary with training, validation and test data.
        arima_order: Tuple. Contains the argument p, d, q for ARIMA model.
        graph: Boolean. Plot the predictions and test dataset.


    Returns: Metrics.

    """
    try:
        # prepare training dataset
        train = splits['train']['y'].astype('float32')
        val_y = splits['val']['y'].astype('float32')
        test_y = splits['test']['y'].astype('float32')
        test = pd.concat([val_y, test_y])

        history = [x for x in train]
        predictions = list()

        for t in range(len(test)):
            model = sm.tsa.ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0, dis=-1)  # All cores
            yhat = model_fit.forecast()[0]  # Predict one step in future from the last value in history variable.
            predictions.append(yhat)
            history.append(test[t])

        # Metrics
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        mape = mean_absolute_percentage_error(test, predictions)
        r2 = r2_score(test, predictions)
        metrics = {"RMSE": rmse,
                   "MSE": mse,
                   "MAE": mae,
                   "MAPE": mape,
                   "R2": r2}

        if graph:
            # plot forecasts against actual outcomes
            test.plot()
            plt.plot(predictions, color='red')
            plt.title('ARIMA Fit')
            plt.ylabel(test.name)
            plt.xlabel('Time [days]')
            plt.legend(['Data test', 'Forecast'], loc='upper left')
            # plt.show()

        return metrics

    except Exception as e:
        pass


def multi_step_arima(splits, arima_order, time_step=7, graph=False):
    """
    Get a multi-step single-shot forecast for the last [time_step] days of test dataset.
    Calculate RMSE, MSE, MAE, R2.
    Args:
        splits: Dictionary with training, validation and test data.
        arima_order: Tuple. Contains the argument p, d, q for ARIMA model.
        time_step: Number of Step Out-of-Sample Forecast
        graph: Boolean. Plot the predictions and test dataset.

    Returns: Metrics and the trained model.
    """
    try:
        # prepare training dataset
        train_y = splits['train']['y'].astype('float32')
        val_y = splits['val']['y'].astype('float32')
        test_y = splits['test']['y'].astype('float32')
        target = pd.concat([train_y, val_y, test_y])

        train_size = int(len(target) - time_step)
        train, test = target[0:train_size], target[train_size:]
        history = [x for x in train]

        # Define model
        model = sm.tsa.ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0, dis=-1)
        # Forecast and metrics
        predictions = model_fit.forecast(steps=time_step)[0]

        # Metrics
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        mape = mean_absolute_percentage_error(test, predictions)
        r2 = r2_score(test, predictions)

        metrics = {"RMSE": rmse,
                   "MSE": mse,
                   "MAE": mae,
                   "MAPE": mape,
                   "R2": r2}
        # print(metrics)

        if graph:
            # plot forecasts against actual outcomes
            test.plot()
            plt.plot(predictions, color='red')
            plt.title('ARIMA Fit')
            plt.ylabel(target.name)
            plt.xlabel('Time [days]')
            plt.legend(['Data', 'Forecast'], loc='upper left')
            # plt.show()

        return metrics, model_fit

    except Exception as e:
        print(e)


def autoregressive_arima(splits, arima_order, time_step=7, graph=False):
    """
    Evaluate an ARIMA model for a given order (p,d,q) and forecast the next one time step.
    Autoregressive model: Train the model using t = (1, ..., t) and predict next time step (t+1).
    Then add (t+1) predicted value to history and fit again the model using t = (1, ..., t, t+1),
    and so on up to t=time_step.


    Args:
        splits: Dictionary with training, validation and test data.
        arima_order: Tuple. Contains the argument p, d, q for ARIMA model.
        time_step: Number of Step Out-of-Sample Forecast
        graph: Boolean. Plot the predictions and test dataset.

    Returns: Metrics.
    """
    try:
        # prepare training dataset
        train_y = splits['train']['y'].astype('float32')
        val_y = splits['val']['y'].astype('float32')
        test_y = splits['test']['y'].astype('float32')
        target = pd.concat([train_y, val_y, test_y])

        train_size = int(len(target) - time_step)
        train, test = target[0:train_size], target[train_size:]
        history = [x for x in train]
        predictions = list()

        for t in range(len(test)):
            model = sm.tsa.ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0, dis=-1)  # All cores
            yhat = model_fit.forecast(steps=1)[0]  # Predict one step in future from the last value in history variable.
            predictions.append(yhat)
            history.append(yhat)

        # Metrics
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        mape = mean_absolute_percentage_error(test, predictions)
        r2 = r2_score(test, predictions)
        metrics = {"RMSE": rmse,
                   "MSE": mse,
                   "MAE": mae,
                   "MAPE": mape,
                   "R2": r2}
        # print(metrics)

        if graph:
            # plot forecasts against actual outcomes
            test.plot()
            plt.plot(predictions, color='red')
            plt.title('ARIMA Fit')
            plt.ylabel(test.name)
            plt.xlabel('Time [days]')
            plt.legend(['Data test', 'Forecast'], loc='upper left')
            # plt.show()
        return metrics

    except Exception as e:
        pass
