import tensorflow as tf

def train_validation_split(
        dataset: tf.data.Dataset, 
        val_size: float=0.2, 
        seed:int=42):
    """
    Uses tensorflow native shuffle and take/skip to split a dataset into train and validation sets.
    This function will shuffle the dataset prior to splitting to ensure randomn split. 
        However the index before and after shuffling will not be tracked so post splitting it will
        be very difficult to associate the train and val samples back to the original dataset/meta data.
    For this reason this function is only intended to be used for dividing the train data into train
        and validation sets to facilitate train and validation loss computation where there is no need
        to associate the train and validation samples back to the original dataset/meta data. 
    
    If you need to keep track of the identity of each sample across data splits that should be done
        manually outside of the BuDDI dataset pipeline (perhaps during the data preprocessing step).  

    :param dataset: tf.data.Dataset to split
    :param val_size: Fraction of dataset to allocate to validation set
    :param seed: Random seed for shuffling for the sake of reproducibility
    :return ds_train: tf.data.Dataset train set
    :return ds_val: tf.data.Dataset validation set
    """

    num_samples = dataset.cardinality().numpy()

    train_size = int((1-val_size) * num_samples)

    ds_shuffled = dataset.shuffle(
        buffer_size=num_samples, seed=seed, reshuffle_each_iteration=False)
    
    ds_train = ds_shuffled.take(train_size)
    ds_val = ds_shuffled.skip(train_size)

    return ds_train, ds_val