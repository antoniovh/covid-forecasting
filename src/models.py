from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape, Lambda, Conv1D
import tensorflow as tf
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


def ffnn(out_steps, layer_num=2, layer_cells=500, num_features=18, dropout_rate=0.05):
    """
    Create a custom Feed Forward Neural Network.
    Args:
        layer_num: Number of hidden layers.
        layer_cells: Number of neurones in each hidden layer.
        out_steps: Time instants to be predicted (t_final = t + out_steps).
                    Must coincide with WindowGenerator.label_width value
        num_features: Number of features (columns) to use in the predictions.
        dropout_rate: (0-1). Percentage of randomly selected neurons or units (hidden and visible) that will be ignored
                    during training to prevent NN from overfitting.

    Returns: Feed Forward Neural Network model.

    """

    model = Sequential()
    # # Take the last time step.
    # # Shape [batch, time, features] => [batch, 1, features]
    # model.add(Lambda(lambda x: x[:, -1:, :]))
    for i in range(layer_num):
        model.add(Dense(layer_cells,
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


def multi_lstm_model(out_steps, cells_lstm=32, layer_num=0, layer_cells=0, num_features=18, dropout_rate=0.05):
    """
    Long Short-Term Memory, a Recurrent Neural Network (RNN) model.
    This model makes multiple time step predictions, namely predicts a sequence of future values (out_steps).
    In this case to approach it, this model is a single-shot model: Single shot predictions where the entire time series
    is predicted at once.
    Args:
        out_steps: Time instants to be predicted (t_final = t + out_steps).
                    Must coincide with WindowGenerator.label_width value
        cells_lstm: Neurons in LSTM layer.
        layer_num: Number of hidden layers.
        layer_cells: Number of neurones in each hidden layer.
        num_features: Number of features (columns) to use in the predictions.
        dropout_rate: (0-1). Percentage of randomly selected neurons or units (hidden and visible) that will be ignored
                    during training to prevent NN from overfitting.

    Returns: LSTM model

    """
    model = Sequential()
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    model.add(LSTM(cells_lstm,
                   return_sequences=False))
    model.add(Dropout(dropout_rate))
    for i in range(layer_num):
        model.add(Dense(layer_cells,
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


def multi_conv_model(conv_width, out_steps, num_features=18):
    """
    Convolutional Neural Network (RNN) model.
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
    model = Sequential()
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

    Returns: Autoregressive predictions where the model only makes single step predictions and its output is fed back
            as its input.

    """
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
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
            x, state = self.lstm_cell(x, states=state,
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
