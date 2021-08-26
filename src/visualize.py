import matplotlib.pyplot as plt


def plot_history(history, model_name):
    """
    Visualize Model Training History from NN.
    Args:
        history: History object that provides a clean observation of the performance of deep learning models over
                time during training. Metrics are stored in a dictionary in the history member of the object returned.
                It is useful to plot metrics (accuracy, loss) of Training and Validation Datasets over training time.
        model_name: Name of the model

    Returns: Plots from the collected history data.

    """
    n = int(len(history.history.keys())/2)
    it = 1
    for i in range(n):
        plt.subplot(2, 1, it)
        metric = list(history.history.keys())[i]
        val_metric = list(history.history.keys())[i+n]
        plt.plot(history.history[metric])
        plt.plot(history.history[val_metric])
        plt.title(f'{model_name}')
        plt.ylabel(f'{metric}')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        it += 1

    plt.tight_layout(pad=2.0)
    # plt.show()


def model_plot(model, splits):
    """
    Calculate predictions and evaluate the errors with some metrics.
    Calculate the Mean Absolute Error, Mean Absolute Percentage Error, Mean Square Erro and R^2 score.

    Args:
        model: Trained model used to predict.
        splits: Dictionary with training, validation and test data.

    Return: Dictionary with the metrics of each dataframe (train, val, test).
    """

    for index, i in enumerate(['train', 'val', 'test']):
        x_t = splits[i]['X']
        y_t = splits[i]['y']

        try:
            if len(x_t) != 0:
                y_p = model.predict(x_t)

                # Plots
                plt.subplot(3, 1, index + 1)
                y_t.plot(color='blue')
                plt.plot(y_p, color='red')
                plt.title(f'{model.name} fit {i}')
                plt.ylabel(y_t.name)
                plt.xlabel('Time, t [days]')
                plt.legend(['Data', 'Forecast'], loc='upper left')
                plt.tight_layout(pad=3.0)

                # plt.show()
        except Exception as e:
            print(e)