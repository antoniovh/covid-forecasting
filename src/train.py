import tensorflow as tf


def make_splits(df,
                ptrain=0.8,
                pval=0.15,
                ptest=0.05):
    """
    Split the dataframe in train, validation and test dataframe.
    Args:
        df: DataFrame
        ptrain: Size for the training dataframe.
        pval: Size for the validation dataframe.
        ptest: Size for the test dataframe.

    Returns: Dict with training, validation and test dataframe.

    """

    n = len(df)
    train_df = df[0:int(n * ptrain)]
    val_df = df[int(n * ptrain):int(n * (ptrain + pval))]
    test_df = df[int(n * ptest):]
    split_df = {'train': train_df, 'val': val_df, 'test': test_df}

    return split_df


def normalize(split_df):
    """
    Normalize training, validation and test dataframe.
    Before training a deep learning model, it is important to scale features. Normalization is a common way of doing
    this scaling: subtract the mean and divide by the standard deviation of each feature.

    Note: The mean and standard deviation should only be computed using the training data so that the models have no
            access to the values in the validation and test sets.

    Args:
        split_df: Dictionary with training, validation and test dataframe without normalization.

    Returns: Dictionary with training, validation and test dataframe after normalization. And the mean and std used in
            the process.

    """
    train_df = split_df['train']
    val_df = split_df['val']
    test_df = split_df['test']

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    split_ndf = {'train': train_df, 'val': val_df, 'test': test_df, 'mean': train_mean, 'std': train_std}

    return split_ndf


def denormalize(split_ndf):
    """
    Denormalize training, validation and test dataframe.
    Args:
        split_ndf: Dictionary with training, validation and test normalized dataframe.

    Returns: Dictionary with training, validation and test dataframe before normalization.

    """

    train_df = split_ndf['train']
    val_df = split_ndf['val']
    test_df = split_ndf['test']

    train_mean = split_ndf['mean']
    train_std = split_ndf['std']

    train_df = train_df * train_std + train_mean
    val_df = val_df * train_std + train_mean
    test_df = test_df * train_std + train_mean

    split_df = {'train': train_df, 'val': val_df, 'test': test_df}

    return split_df


def compile_and_fit_wg(model, window, patience=3, max_epoch=30):
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

