import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class WindowGenerator:
    """
    Class to:
    1. Generate a time window.
    2. Split de data into different variables (input and labels) of each tensor. Necessary to plot an example.
    3. Make a dataset from dataframe (train, validation, test).
    4. Plot an example of predictions with the trained model using the inputs and the labels (target or expected values)
        obtained with split_window() method. Predict using the model and inputs and compare with the labels.
    5. Compile and fit the input model and return history object.
    """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """
        Constructs all the necessary attributes for the window.
        Args:
            input_width: Size of the history window. Number of instants used to train the models.
            label_width: Number of consecutive instants to predict.
            shift: Number of instants (lags) between the last lag of history window and the last instant to predict.
            train_df: Train DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            label_columns: None or a list of the name columns to predict.

        Attributes:
            self.label_columns_indices: Dictionary with the name and its index of the column targets.
            self.column_indices: Dictionary with the name and its index of the dataset.
            self.total_window_size: history window + shift.
            self.input_slice: (0, input_width, None)
            self.input_indices: List from 0 to input_width
            self.labels_slice: (total_window_size - label_width, None, None)
            self.label_indices: List from (total_window_size - label_width) to (total_window_size).
        """

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

    def split_window(self, data):
        """
        WidowGenerator class method. Split de data into different variables (input and labels) of each tensor.
        Args:
            self: Class method.
            data: List of rank-R tensors.txt.

        Returns:
            inputs: Data of history window used to make the predictions in the plotted example.
            labels: Data with the expected target values of the predictions.
        """
        inputs = data[:, self.input_slice, :]
        labels = data[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        # (batch, time, features)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        Make a dataset from input DataFrame (train, validation, test).
        Working with datasets has some advantages when training the model and using fit() arguments.
        (https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
        Args:
            data: DataFrame.

        Returns: DataSet.

        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)

        ds = ds.map(self.split_window)

        return ds

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

    def plot(self, model=None, plot_col=['daily_cases'], max_subplots=3):
        """
        Plot an example of the trained model using the inputs and the labels (target or expected values) obtained
        with split_window() method.
        Args:
            self: Class method.
            model: Model to make predictions.
                If it is None, it not makes predictions, only inputs and labels are plotted.
            plot_col: List of the columns names to predict.
            max_subplots: Maximum number of plots.

        Returns: Graph

        """
        inputs, labels = self.example
        # inputs = self.inputs
        # labels = self.labels

        # index of the column to plot
        plot_col_index = [self.column_indices.get(key) for key in plot_col]
        num_features = len(plot_col_index)
        # plt.figure(figsize=(12, 8))
        max_n = min(max_subplots, len(inputs))  # len(inputs) = n batch
        it = 1
        for n in range(max_n):
            for m in range(num_features):
                plt.subplot(max_n, num_features, it)  # select subplot
                plt.ylabel(f'{plot_col[m]} [normed]')
                plt.plot(self.input_indices, inputs[n, :, plot_col_index[m]],
                         label='Inputs', marker='.', zorder=-10)
                it += 1

                if self.label_columns:
                    label_col_index = self.label_columns_indices.get(plot_col[m], None)
                else:
                    label_col_index = plot_col_index[m]

                if label_col_index is None:
                    continue

                plt.scatter(self.label_indices, labels[n, :, label_col_index],
                            edgecolors='k', label='Labels', c='#2ca02c', s=64)
                if model is not None:
                    predictions = model(inputs)
                    plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)
                    plt.suptitle(f'{model.name}')
                if n == 0:
                    plt.legend()
                plt.xlabel('Time [days]')

        plt.tight_layout(pad=2.0)
        # plt.show()


def compile_and_fit(model, window, patience=3, max_epoch=30):
    """
    Compile and fit the models. This function is used with WindowGenerator Object, df.data Datasets and DL models.
    Some ML models need 'x' (features_train) and 'y' (target) to be trained, if you use a Dataset you get an error.
    Args:
        model: Model to fit.
        window: WindowGenerator Object that contains training, validation and test tf.data Dataset
        patience: Argument for early stopping to avoid overfitting.
            Number of epochs to wait before early stop if no progress on the validation set.
        max_epoch: Max epochs to train the model.

    Returns:
        History object that provides a clean observation of the performance of deep learning models over
        time during training. Metrics are stored in a dictionary in the history member of the object returned.
        It is useful to plot metrics (accuracy, loss) of Training and Validation Datasets over training time.

    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train,
                        epochs=max_epoch,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history
