from typing import Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

def fit_buddi(
    supervised_model: tf.keras.Model, 
    unsupervised_model: tf.keras.Model, 
    dataset_supervised: tf.data.Dataset, 
    dataset_unsupervised: tf.data.Dataset,
    dataset_test_supervised: Optional[tf.data.Dataset]=None,
    dataset_test_unsupervised: Optional[tf.data.Dataset]=None,
    epochs: int=10, 
    batch_size:int =16, 
    shuffle_every_epoch: bool=True, 
    prefetch: bool=False) -> pd.DataFrame:
    """
    Train buddi model with supervised and unsupervised datasets, 
        alternating between the two datasets from batch to batch.

    This fit function assumes that the supervised dataset is larger than the unsupervised dataset.

    Every epoch, the supervised dataset fully iterates through, each batch of supervised data training 
        is followed by training on a batch of unsupervised data from dataset_unsupervised.repeat().
        This does not necessary guarantee that the unsupervised dataset is fully iterated through
        each epoch but it should most likely do that when unsupervised dataset is much smaller than 
        the supervised dataset.

    :param supervised_model: Supervised BUDDI model from `build_buddi4`
    :param unsupervised_model: Unsupervised BUDDI model from `build_buddi4`
    :param dataset_supervised: Supervised dataset from `buddi_v2.dataset.buddi4_dataset.get_supervised_dataset`
    :param dataset_unsupervised: Unsupervised dataset from `buddi_v2.dataset.buddi4_dataset.get_unsupervised_dataset`
    :param epochs: Number of epochs to train the model
    :param batch_size: Batch size for training
    :param shuffle_every_epoch: Whether to shuffle the dataset every epoch
    :param prefetch: Whether to prefetch the dataset
    :return: DataFrame containing the loss values for each batch and epoch
        The DataFrame will contain rows for both the unsupervised and supervised loss with column 
        `type` indicating which type of training the row of loss corresponds to.
    :rtype: pd.DataFrame
    """
    
    # Get the number of batches in the smaller dataset
    num_sup_samples = dataset_supervised.cardinality().numpy()
    num_unsup_samples = dataset_unsupervised.cardinality().numpy()
    
    _ds_supervised = dataset_supervised.shuffle(num_sup_samples, reshuffle_each_iteration=shuffle_every_epoch).batch(batch_size)
    _ds_unsupervised = dataset_unsupervised.shuffle(num_unsup_samples, reshuffle_each_iteration=shuffle_every_epoch).batch(batch_size)

    n_batches = np.ceil(max(num_unsup_samples, num_sup_samples) / batch_size).astype(int)    

    if num_unsup_samples < num_sup_samples:
        _ds_unsupervised = _ds_unsupervised.repeat()

    if prefetch:
        _ds_supervised = _ds_supervised.prefetch(tf.data.experimental.AUTOTUNE)
        _ds_unsupervised = _ds_unsupervised.prefetch(tf.data.experimental.AUTOTUNE)

    _ds_test_supervised = None
    _ds_test_unsupervised = None
    if dataset_test_supervised is not None:
        _ds_test_supervised = dataset_test_supervised.batch(batch_size)
    if dataset_test_unsupervised is not None:
        _ds_test_unsupervised = dataset_test_unsupervised.batch(batch_size)        

    sup_loss_df = pd.DataFrame()
    unsup_loss_df = pd.DataFrame()
    test_sup_loss_df = pd.DataFrame()
    test_unsup_loss_df = pd.DataFrame()

    for epoch in range(epochs):

        unsup_iter = iter(_ds_unsupervised)
        epoch_str = f"Epoch {epoch+1}/{epochs}"

        sup_batch_losses = []
        unsup_batch_losses = []
        # Loop through both datasets simultaneously
        for _, sup_batch in enumerate(tqdm(_ds_supervised, total=n_batches, desc=epoch_str)):

            # Supervised training step
            sup_x, sup_y = sup_batch
            sup_loss = supervised_model.train_on_batch(sup_x, sup_y)
            sup_batch_losses.append([x.item() for x in  sup_loss])
            
            # iterate over repeated unsupervised dataset
            unsup_batch = next(unsup_iter)

            # Unsupervised training step (No labels in unsupervised dataset)
            unsup_x, unsup_y = unsup_batch
            unsup_loss = unsupervised_model.train_on_batch(unsup_x, unsup_y)
            unsup_batch_losses.append([x.item() for x in  unsup_loss])

        sup_test_batch_losses = []
        unsup_test_batch_losses = []
        if _ds_test_supervised is not None:
            val_accumulators = {
                name: tf.keras.metrics.Mean(name=name + "_val")
                for name in supervised_model.metrics_names
            }
            supervised_model.reset_metrics()
            for test_sup_batch in tqdm(_ds_test_supervised, desc="Testing supervised model"):
                test_sup_x, test_sup_y = test_sup_batch
                test_sup_loss = supervised_model.test_on_batch(test_sup_x, test_sup_y)
                for name, value in zip(supervised_model.metrics_names, test_sup_loss):
                    val_accumulators[name].update_state(value)
                sup_test_batch_losses.append([accu.result().numpy() for _, accu in val_accumulators.items()])
            del val_accumulators
            supervised_model.reset_metrics()
        if _ds_test_unsupervised is not None:  
            val_accumulators = {
                name: tf.keras.metrics.Mean(name=name + "_val")
                for name in unsupervised_model.metrics_names
            }          
            unsupervised_model.reset_metrics()
            for test_unsup_batch in tqdm(_ds_test_unsupervised, desc="Testing unsupervised model"):
                test_unsup_x, test_unsup_y = test_unsup_batch
                test_unsup_loss = unsupervised_model.test_on_batch(test_unsup_x, test_unsup_y)
                for name, value in zip(unsupervised_model.metrics_names, test_unsup_loss):
                    val_accumulators[name].update_state(value)
                unsup_test_batch_losses.append([accu.result().numpy() for _, accu in val_accumulators.items()])
                # unsup_test_batch_losses.append([x.item() for x in  test_unsup_loss])
            del val_accumulators
            unsupervised_model.reset_metrics()

        sup_batch_loss_df = pd.DataFrame(sup_batch_losses)
        unsup_batch_loss_df = pd.DataFrame(unsup_batch_losses)
        sup_batch_loss_df.columns = supervised_model.metrics_names
        unsup_batch_loss_df.columns = unsupervised_model.metrics_names
        sup_batch_loss_df['epoch'] = epoch
        unsup_batch_loss_df['epoch'] = epoch
        sup_batch_loss_df['batch'] = np.arange(len(sup_batch_losses))
        unsup_batch_loss_df['batch'] = np.arange(len(unsup_batch_losses))
        sup_loss_df = pd.concat([sup_loss_df, sup_batch_loss_df], ignore_index=True)
        unsup_loss_df = pd.concat([unsup_loss_df, unsup_batch_loss_df], ignore_index=True)
        
        if len(sup_test_batch_losses) > 0:
            test_sup_batch_loss_df = pd.DataFrame(sup_test_batch_losses)
            test_sup_batch_loss_df.columns = supervised_model.metrics_names
            test_sup_batch_loss_df['epoch'] = epoch
            test_sup_batch_loss_df['batch'] = np.arange(len(sup_test_batch_losses))
            test_sup_loss_df = pd.concat([test_sup_loss_df, test_sup_batch_loss_df], ignore_index=True)
        if len(unsup_test_batch_losses) > 0:
            test_unsup_batch_loss_df = pd.DataFrame(unsup_test_batch_losses)
            test_unsup_batch_loss_df.columns = unsupervised_model.metrics_names
            test_unsup_batch_loss_df['epoch'] = epoch
            test_unsup_batch_loss_df['batch'] = np.arange(len(unsup_test_batch_losses))
            test_unsup_loss_df = pd.concat([test_unsup_loss_df, test_unsup_batch_loss_df], ignore_index=True)

    print("Training complete!")

    sup_loss_df.columns = [
        col if i != 1 else 'X_reconstruction_loss' for i, col in enumerate(sup_loss_df.columns)
    ]
    unsup_loss_df.columns = [
        col if i != 1 else 'X_reconstruction_loss' for i, col in enumerate(unsup_loss_df.columns)
    ]
    unsup_loss_df['type'] = 'unsupervised'
    sup_loss_df['type'] = 'supervised'
    all_loss_df = pd.concat([sup_loss_df, unsup_loss_df])
    all_loss_df['split'] = 'train'

    if test_sup_loss_df.shape[0] > 0:
        test_sup_loss_df.columns = [
            col if i != 1 else 'X_reconstruction_loss' for i, col in enumerate(test_sup_loss_df.columns)
        ]
        test_sup_loss_df['type'] = 'supervised'

    if  test_unsup_loss_df.shape[0] > 0:
        test_unsup_loss_df.columns = [
            col if i != 1 else 'X_reconstruction_loss' for i, col in enumerate(test_unsup_loss_df.columns)
        ]
        test_unsup_loss_df['type'] = 'unsupervised'    
    all_test_loss_df = pd.concat([test_sup_loss_df, test_unsup_loss_df])
    all_test_loss_df['split'] = 'val'

    return pd.concat([all_loss_df, all_test_loss_df])