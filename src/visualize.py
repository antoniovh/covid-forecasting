import matplotlib.pyplot as plt


def plot_history(history):
    """
    Visualize Model Training History from NN.
    Args:
        history: History object that provides a clean observation of the performance of deep learning models over
                time during training. Metrics are stored in a dictionary in the history member of the object returned.
                It is useful to plot metrics (accuracy, loss) of Training and Validation Datasets over training time.

    Returns: Plots from the collected history data.

    """
    n = len(history.history.keys())/2
    for i in range(n):
        metric = history.history.keys()[i]
        val_metric = history.history.keys()[i+n]
        plt.plot(history.history[metric])
        plt.plot(history.history[val_metric])
        plt.title(f'model {metric}')
        plt.ylabel(f'{metric}')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

    plt.show()
