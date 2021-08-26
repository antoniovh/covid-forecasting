import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def make_splits(df,
                target,
                time_step=0,
                ntrain=0.8,
                nval=0.15,
                ntest=0.05):
    """
    Split the dataframe in train, validation and test dataframe.
    For models not autoregressive, it is necessary to modify the target variable to create lagged
    obersvations in order to predict the n future instants.
    These models will try to find dependent relationships between the target y(t+n) and the features x(t).

    Args:
        df: DataFrame
        target: Target variable.
        time_step: Number of lags.
        ntrain: Size for the training dataframe.
        nval: Size for the validation dataframe.
        ntest: Size for the test dataframe.

    Returns: Dict splited into training, validation and test dataframe. Also into features and target.
            {train:{X: , y: }, val:{X: , y: }, test:{X: , y: }}

    """
    # Target variable
    df = df.copy()
    y_target = df[target].shift(-int(time_step))
    target_t = f'{target}_(t+{time_step})'
    df[target_t] = y_target
    df = df.drop(columns=[target])
    # Keep only rows with non-NANs
    df = df[~df.isna().any(axis='columns')]

    # split into train, val, test datasets.
    n = len(df)
    train_df = df[0:int(n * ntrain)]
    val_df = df[int(n * ntrain):int(n * (ntrain + nval))]
    test_df = df[int(n * (1 - ntest)):]
    # split into features and target variables.
    x_train = train_df.drop(columns=target_t)
    x_val = val_df.drop(columns=target_t)
    x_test = test_df.drop(columns=target_t)
    y_train = train_df[target_t]
    y_val = val_df[target_t]
    y_test = test_df[target_t]

    splits_df = {'train': {}, 'val': {}, 'test': {}}
    splits_df['train']['X'] = x_train
    splits_df['train']['y'] = y_train
    splits_df['val']['X'] = x_val
    splits_df['val']['y'] = y_val
    splits_df['test']['X'] = x_test
    splits_df['test']['y'] = y_test

    return splits_df


def normalize(splits):
    """
    Normalize training, validation and test dataframe.
    Before training a deep learning model, it is important to scale features. Normalization is a common way of doing
    this scaling: subtract the mean and divide by the standard deviation of each feature.

    Note: The mean and standard deviation should only be computed using the training data so that the models have no
            access to the values in the validation and test sets.

    Args:
        splits: Dictionary with training, validation and test dataframe without normalization.

    Returns: Dictionary with training, validation and test dataframe after normalization.
            And the mean and std used in the process.

    """
    mean = splits['train']['X'].mean()
    std = splits['train']['X'].std()

    for i in ['train', 'val', 'test']:
        splits[i]['X'] = (splits[i]['X'] - mean) / std

    norm = pd.DataFrame([mean, std], index=['mean', 'std']).T

    return splits, norm


def denormalize(splits, norm):
    """
    Denormalize training, validation and test dataframe.
    Args:
        splits: Dictionary with training, validation and test normalized dataframe.
        norm: DataFrame with the parameters used in normalization process.

    Returns: Dictionary with training, validation and test dataframe before normalization.

    """
    mean = norm['mean']
    std = norm['std']

    for i in ['train', 'val', 'test']:
        splits[i]['X'] = splits[i]['X'] * std + mean

    return splits


def model_summary(model, splits):
    """
    Calculate predictions and evaluate the errors with some metrics .
    Calculate the Mean Absolute Error, Mean Absolute Percentage Error, Mean Square Erro and R^2 score.

    Args:
        model: Trained model used to predict.
        splits: Dictionary with training, validation and test data.

    Return: Dictionary with the metrics of each dataframe (train, val, test).
    """

    metrics = {'rmse': {}, 'mse': {}, 'mae': {}, 'mape': {}, 'r2': {}}

    for i in ['train', 'val', 'test']:
        x_t = splits[i]['X']
        y_t = splits[i]['y']

        try:
            if len(x_t) != 0:
                y_p = model.predict(x_t)

                # Metrics
                mse = mean_squared_error(y_true=y_t,
                                         y_pred=y_p,
                                         squared=False)
                rmse = sqrt(mse)
                mae = mean_absolute_error(y_true=y_t,
                                          y_pred=y_p)
                mape = mean_absolute_percentage_error(y_true=y_t,
                                                      y_pred=y_p)
                r2 = r2_score(y_t, y_p)

                metrics['rmse'][i] = rmse
                metrics['mae'][i] = mae
                metrics['mape'][i] = mape
                metrics['mse'][i] = mse
                metrics['r2'][i] = r2

                df_metrics = pd.DataFrame.from_dict(metrics, orient='index')

        except Exception as e:
            print(e)

    return metrics, df_metrics


def my_map(fun1, obj, iterlist):
    """
    Map() function with a non-iterable and iterable set of parameters.
    Args:
        fun1: Function to use with 2 arguments (obj, element)
        obj: Non iterable or constant object.
        iterlist: List of elements to iterate.

    Returns: Return list from map(func1, obj_cte, iterlist)

    """
    def fun2(x):
        """
        Apply function 1.
        Args:
            x: Element of a list
        Returns: The return from function 1
        """
        return fun1(obj, x)
    return map(fun2, iterlist)
