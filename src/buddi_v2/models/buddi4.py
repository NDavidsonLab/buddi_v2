from typing import Tuple, Union, Callable, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]] # alias
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy

from .components.branches import *
from .components.layers import *
from .components.losses import *

def build_buddi4(
        n_x: int,
        n_y: int,
        n_labels: int,
        n_stims: int,
        n_samp_types: int,
        z_dim: int = 64,
        encoder_hidden_dim: int = 512,
        decoder_hidden_dim: int = 512,
        alpha_x: float = 1.0,
        alpha_label: float = 100.0,
        alpha_stim: float = 100.0,
        alpha_samp_type: float = 100.0,
        alpha_prop: float = 100.0,
        beta_kl_slack: float = 0.1,
        beta_kl_label: float = 100.0,
        beta_kl_stim: float = 100.0,
        beta_kl_samp_type: float = 100.0,
        reconstr_loss_fn: Callable = MeanSquaredError, # default loss fn
        classifier_loss_fn: Callable = CategoricalCrossentropy, # default loss fn
        label_classifier_loss_fn: Optional[Callable] = None,
        stim_classifier_loss_fn: Optional[Callable] = None,
        samp_type_classifier_loss_fn: Optional[Callable] = None,
        prop_estimator_loss_fn: Optional[Callable] = None,
        activation: ActivationFn = 'relu',
        optimizer = Adam(learning_rate=0.0005),
        return_decoder: bool = False
    ) -> Union[Tuple[Model, Model], Tuple[Model, Model, Model, Model]]:
    """
    Builds the BUDDI model.

    :param n_x: Number of features in the input data
    :param n_y: Number of features in the output data
    :param n_labels: Number of unique labels in the data
    :param n_stims: Number of unique stimulation conditions in the data
    :param n_samp_types: Number of unique sample types in the data
    :param z_dim: Dimension of the latent space
    :param encoder_hidden_dim: Dimension of the hidden layers in the encoder
    :param decoder_hidden_dim: Dimension of the hidden layers in the decoder
    :param alpha_x: Weight of the reconstruction loss
    :param alpha_label: Weight of the classifier loss for the label branch
    :param alpha_stim: Weight of the classifier loss for the stimulation branch
    :param alpha_samp_type: Weight of the classifier loss for the sample type branch
    :param alpha_prop: Weight of the classifier loss for the proportion estimator branch
    :param beta_kl_slack: Weight of the KL divergence loss for the slack branch
    :param beta_kl_label: Weight of the KL divergence loss for the label branch
    :param beta_kl_stim: Weight of the KL divergence loss for the stimulation branch
    :param beta_kl_samp_type: Weight of the KL divergence loss for the sample type branch
    :param reconstr_loss_fn: Reconstruction loss function
    :param classifier_loss_fn: Classifier loss function
        Defines the default loss function for all classifier branches.
        Overridden by the specific loss function for each branch.
    :param label_classifier_loss_fn: Classifier loss function for the label branch
        Overrides the default loss function for the label branch.
    :param stim_classifier_loss_fn: Classifier loss function for the stimulation branch
        Overrides the default loss function for the stimulation branch.
    :param samp_type_classifier_loss_fn: Classifier loss function for the sample type branch
        Overrides the default loss function for the sample type branch.
    :param prop_estimator_loss_fn: Classifier loss function for the proportion estimator branch
        Overrides the default loss function for the proportion estimator branch.
    :param return_decoder: Whether to return the decoder model
    :param activation: Activation function for the hidden layers
    :param optimizer: Optimizer for the model
    :return supervised_buddi: Supervised BUDDI model
        Compiled with the specified optimizer and loss functions.
    :return unsupervised_buddi: Unsupervised BUDDI model
        Compiled with the specified optimizer and loss functions.
    :return supervised_decoder: Decoder model for the supervised branch
    :return unsupervised_decoder: Decoder model for the unsupervised branch
    """

    if label_classifier_loss_fn is None:
        label_classifier_loss_fn = classifier_loss_fn
    if stim_classifier_loss_fn is None:
        stim_classifier_loss_fn = classifier_loss_fn
    if samp_type_classifier_loss_fn is None:
        samp_type_classifier_loss_fn = classifier_loss_fn
    if prop_estimator_loss_fn is None:
        prop_estimator_loss_fn = classifier_loss_fn

    X = Input(shape=(n_x,), name='X')
    Y = Input(shape=(n_y,), name='Y')

    # --------------------- Encoders ---------------------
    ## This encoder branch captures the variation in data that is
    ## explained by sample labels
    encoder_branch_label = build_encoder_branch(
        inputs = X,
        hidden_dim = encoder_hidden_dim,
        z_dim = z_dim,
        activation = activation,
        representation_name = 'label'
    )

    ## This encoder branch captures the variation in data that is
    ## explained by stimulation conditions
    encoder_branch_stim = build_encoder_branch(
        inputs = X,
        hidden_dim = encoder_hidden_dim,
        z_dim = z_dim,
        activation = activation,
        representation_name = 'stim'
    )

    ## This encoder branch captures the variation in data that is
    ## explained by sample types
    encoder_branch_samp_type = build_encoder_branch(
        inputs = X,
        hidden_dim = encoder_hidden_dim,
        z_dim = z_dim,
        activation = activation,
        representation_name = 'samp_type'
    )

    ## This encoder branch absorbs the additional variation in data 
    ## that is not captured by other branches
    encoder_branch_slack = build_encoder_branch(
        inputs = X,
        hidden_dim = encoder_hidden_dim,
        z_dim = z_dim,
        activation = activation,
        representation_name = 'slack'
    )

    ## Each of these outputs are z_mu and z_log_var concatenated
    z_params_label = encoder_branch_label(X)
    z_params_stim = encoder_branch_stim(X)
    z_params_samp_type = encoder_branch_samp_type(X)
    z_params_slack = encoder_branch_slack(X)

    # --------------------- Sampling ---------------------
    z_label = ReparameterizationLayer(name='z_label')(z_params_label)
    z_stim = ReparameterizationLayer(name='z_stim')(z_params_stim)
    z_samp_type = ReparameterizationLayer(name='z_samp_type')(z_params_samp_type)
    z_slack = ReparameterizationLayer(name='z_slack')(z_params_slack)

    ## Wrapping the intermediate latent space representations into inputs
    z_label_input = tf.keras.Input(shape=z_label.shape[1:], name="z_label_input")
    z_stim_input = tf.keras.Input(shape=z_stim.shape[1:], name="z_stim_input")
    z_samp_type_input = tf.keras.Input(shape=z_samp_type.shape[1:], name="z_samp_type_input")

    # --------------------- Classifier ---------------------
    ## This classifier network predicts sample labels from the latent space
    classifier_branch_label = build_latent_space_classifier(
        inputs = z_label_input,
        num_classes = n_labels,
        representation_name = 'label'
    )

    ## This classifier network predicts stimulation conditions from the latent space
    classifier_branch_stim = build_latent_space_classifier(
        inputs = z_stim_input,
        num_classes = n_stims,
        representation_name = 'stim'
    )

    ## This classifier network predicts sample types from the latent space
    classifier_branch_samp_type = build_latent_space_classifier(
        inputs = z_samp_type_input,
        num_classes = n_samp_types,
        representation_name = 'samp_type'
    )

    ## These are the predicted labels from the latent space
    pred_label = classifier_branch_label(z_label)
    pred_stim = classifier_branch_stim(z_stim)
    pred_samp_type = classifier_branch_samp_type(z_samp_type)

    ## The Proportion Estimator branch predicts cell type proportions from X
    prop_estimator = build_prop_estimator(
        inputs = X,
        num_classes = n_y,
        activation = activation,
        estimator_name = 'prop_estimator'
    )

    Y_hat = prop_estimator(X)

    # --------------------- Decoders ---------------------

    ## Supervised Decoder Input
    supervised_decoder_concat = Concatenate(name='supervised_ls_concat')(
        [Y, z_label, z_stim, z_samp_type, z_slack])
    supervised_decoder_input = tf.keras.Input(shape=supervised_decoder_concat.shape[1:], 
                                              name="supervised_decoder_input")
    
    unsupervised_decoder_concat = Concatenate(name='unsupervised_ls_concat')(
        [Y_hat, z_label, z_stim, z_samp_type, z_slack])
    unsupervised_decoder_input = tf.keras.Input(shape=unsupervised_decoder_concat.shape[1:], 
                                                name="unsupervised_decoder_input")
    
    supervised_decoder, unsupervised_decoder = build_semi_supervised_decoder(
        inputs_supervised = supervised_decoder_input,
        inputs_unsupervised = unsupervised_decoder_input,
        output_dim = n_x,
        hidden_dims = decoder_hidden_dim,
        activation = activation,
        output_activation = 'sigmoid', # Assuming MinMaxNormalized data
        output_name = 'X'
    )

    X_hat_supervised = supervised_decoder(supervised_decoder_concat)
    X_hat_unsupervised = unsupervised_decoder(unsupervised_decoder_concat)

    # --------------------- Losses ---------------------
    ## Note these are all functions with standard tensorflow signature accepting y_true and y_pred
    kl_loss_fn_label = kl_loss_generator(
        beta=1.0,#beta_kl_label, 
        agg_fn=K.sum, axis=-1)
    kl_loss_fn_stim = kl_loss_generator(
        beta=1.0,#beta_kl_stim, 
        agg_fn=K.sum, axis=-1)
    kl_loss_fn_samp_type = kl_loss_generator(
        beta=1.0,#beta_kl_samp_type, 
        agg_fn=K.sum, axis=-1)
    kl_loss_fn_slack = kl_loss_generator(
        beta=1.0,#beta_kl_slack, 
        agg_fn=K.sum, axis=-1)

    reconstr_loss_fn = reconstr_loss_generator(
        weight=1.0,
        reconstr_loss_fn=reconstr_loss_fn, 
        reduction='sum')

    classifier_loss_fn_label = classifier_loss_generator(
        loss_fn=label_classifier_loss_fn,
        weight=1.0,#alpha_label, 
        reduction='sum')
    classifier_loss_fn_stim = classifier_loss_generator(
        loss_fn=stim_classifier_loss_fn,
        weight=1.0,#alpha_stim, 
        reduction='sum')
    classifier_loss_fn_samp_type = classifier_loss_generator(
        loss_fn=samp_type_classifier_loss_fn,
        weight=1.0,#alpha_samp_type, 
        reduction='sum')
    # this is the loss function for the proportion estimator
    prop_estimator_loss_fn = classifier_loss_generator(
        loss_fn=prop_estimator_loss_fn,
        weight=1.0,#alpha_prop, 
        reduction='sum')

    # --------------------- Compile ---------------------

    ## Shared output and loss components
    shared_buddi_z_params = [
        z_params_label, z_params_stim, z_params_samp_type, z_params_slack
    ]
    shared_buddi_kl_losses = [
        kl_loss_fn_label, kl_loss_fn_stim, kl_loss_fn_samp_type, kl_loss_fn_slack
    ]
    shared_buddi_kl_loss_weights = [
        beta_kl_label, beta_kl_stim, beta_kl_samp_type, beta_kl_slack
    ]

    shared_buddi_pred_outputs= [
        pred_label, pred_stim, pred_samp_type, 
        # note that Y_hat is predicted regardless
        Y_hat 
    ]
    shared_buddi_pred_losses = [
        classifier_loss_fn_label, classifier_loss_fn_stim, classifier_loss_fn_samp_type
        # note Y_hat loss is not included as a shared loss, that is because
        # in the unsupervised branch we don't have the ground truth to train it    
    ]
    shared_buddi_pred_loss_weights = [
        alpha_label, alpha_stim, alpha_samp_type
    ]

    supervised_buddi_inputs = [X, Y]
    supervised_buddi_outputs = [X_hat_supervised] + shared_buddi_z_params + shared_buddi_pred_outputs    
    supervisied_buddi_pred_losses = shared_buddi_pred_losses + [prop_estimator_loss_fn]
    supervised_buddi_losses = [
        # standard reconstruction loss
        # corresponding to [X_hat_supervised]
        # note this is shared with the unsupervised branch
        reconstr_loss_fn, 
        # shared KL losses to the latent spaces
        # corresponding to shared_buddi_z_params
        *shared_buddi_kl_losses, 
        # prediction losses specific to supervised branches 
        # corresponding to shared_buddi_pred_outputs
        *supervisied_buddi_pred_losses 
    ]
    supervised_buddi_loss_weights = [
        alpha_x, # reconstruction loss
        *shared_buddi_kl_loss_weights, 
        *shared_buddi_pred_loss_weights,
        alpha_prop, # proportion estimator loss in supervised branch
    ]
    
    unsupervised_buddi_inputs = [X]
    unsupervised_buddi_ouputs = [X_hat_unsupervised] + shared_buddi_z_params + shared_buddi_pred_outputs    
    unsupervised_buddi_pred_losses = shared_buddi_pred_losses + [unsupervised_dummy_loss_fn] # dummy loss item here that returns 0
    unsupervised_buddi_losses = [
        # standard reconstruction loss
        # corresponding to [X_hat_unsupervised]
        # note this is shared with the supervised branch
        reconstr_loss_fn,
        # shared KL losses to the latent spaces 
        # corresponding to shared_buddi_z_params
        *shared_buddi_kl_losses, 
        # prediction losses specific to unsupervised branches 
        # corresponding to shared_buddi_pred_outputs
        *unsupervised_buddi_pred_losses 
    ]
    unsupervised_buddi_loss_weights = [
        alpha_x, # reconstruction loss
        *shared_buddi_kl_loss_weights,
        *shared_buddi_pred_loss_weights,
        0.0 # proportion estimator loss in unsupervised branch
    ]

    supervised_buddi = Model(
        inputs = supervised_buddi_inputs,
        outputs = supervised_buddi_outputs,
        name = 'supervised_buddi'
    )
    supervised_buddi.compile(
        optimizer=optimizer, 
        loss=supervised_buddi_losses,
        loss_weights=supervised_buddi_loss_weights
        )

    unsupervised_buddi = Model(
        inputs = unsupervised_buddi_inputs,
        outputs = unsupervised_buddi_ouputs,
        name = 'unsupervised_buddi'
    )
    unsupervised_buddi.compile(
        optimizer=optimizer, 
        loss=unsupervised_buddi_losses,
        loss_weights=unsupervised_buddi_loss_weights
        )

    if return_decoder:        
        return supervised_buddi, unsupervised_buddi, supervised_decoder, unsupervised_decoder
    else:
        return supervised_buddi, unsupervised_buddi

def fit_buddi4(
        supervised_model: tf.keras.Model, 
        unsupervised_model: tf.keras.Model, 
        dataset_supervised: tf.data.Dataset, 
        dataset_unsupervised: tf.data.Dataset, 
        epochs: int=10, 
        batch_size:int =16, 
        shuffle_every_epoch: bool=True, 
        prefetch: bool=False) -> pd.DataFrame:
    """
    Train buddi4 model with supervised and unsupervised datasets, 
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

    sup_loss_df = pd.DataFrame()
    unsup_loss_df = pd.DataFrame()

    for epoch in range(epochs):

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
            unsup_batch = next(iter(_ds_unsupervised))

            # Unsupervised training step (No labels in unsupervised dataset)
            unsup_x, unsup_y = unsup_batch
            unsup_loss = unsupervised_model.train_on_batch(unsup_x, unsup_y)
            unsup_batch_losses.append([x.item() for x in  unsup_loss])

        sup_batch_loss_df = pd.DataFrame(sup_batch_losses)
        unsup_batch_loss_df = pd.DataFrame(unsup_batch_losses)
        sup_batch_loss_df.columns = supervised_model.metrics_names
        unsup_batch_loss_df.columns = unsupervised_model.metrics_names
        sup_batch_loss_df['epoch'] = epoch
        unsup_batch_loss_df['epoch'] = epoch
        sup_batch_loss_df['batch'] = sup_batch_loss_df.index
        unsup_batch_loss_df['batch'] = unsup_batch_loss_df.index

        sup_loss_df = pd.concat([sup_loss_df, sup_batch_loss_df], ignore_index=True)
        unsup_loss_df = pd.concat([unsup_loss_df, unsup_batch_loss_df], ignore_index=True)

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

    return all_loss_df