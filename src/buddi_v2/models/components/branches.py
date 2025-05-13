from typing import List, Tuple, Union, Callable

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, Concatenate
from tensorflow.keras.models import Model

ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]

def build_encoder_branch(
        inputs: Input, 
        hidden_dim: int, 
        z_dim: int, 
        activation: ActivationFn = 'relu', 
        representation_name: str = ''
        ) -> Model:
    """
    Defines a single hidden layer encoder that outputs two encodings: mu and log_var.

    :param inputs: Input tensor
    :param hidden_dim: Dimension of the hidden layer
    :param z_dim: Dimension of the latent space
    :param activation: Activation function for the hidden layer
    :param representation_name: Name of the representation that this encoder should learn
        This representation will be enforced outside of this encoder branch in a supervised fasion
        by a classifier network that predicts from the latent space
    :return encoder: Encoder model
    """

    encoder_name = f'encoder_{representation_name}'
    hidden = Dense(hidden_dim, activation=activation, name= encoder_name + '_hidden_layer')(inputs)

    # for clarity, we separately define the mu and log_var layers
    mu = Dense(z_dim, name=encoder_name + '_z_mu')(hidden)
    log_var = Dense(z_dim, name=encoder_name + '_z_log_var')(hidden)
    # for convenience in loss computation, we concatenate the mu and log_var layers
    z_params = Concatenate(name=encoder_name + '_z_params')([mu, log_var])

    return Model(inputs, z_params, name=encoder_name)

def build_prop_estimator(
        inputs: Input, 
        num_classes: int, 
        hidden_dims: Union[int, List[int]] = [512,256], 
        activation: ActivationFn = 'relu',
        output_activation: ActivationFn = 'softmax',
        estimator_name: str = 'prop_estimator'
        ) -> Model:
    """
    Defines a classifier network that estimates cell type proportion from the input. 
    Meant to directly predict from expression data.

    :param inputs: Input tensor
    :param num_classes: Number of classes to predict in the output layer
    :param hidden_dims: Dimension of the hidden layers. 
        Number of hidden layers is determined by the length of this list.
        Defaults to 2 hidden layers with 512 and 256 units respectively.
    :param activation: Activation function for the hidden layers
    :param output_activation: Activation function for the output layer
    :param estimator_name: Name of the proportion estimator model
    :return prop_estimator: Proportion estimator model
    """

    x = inputs

    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    elif isinstance(hidden_dims, list):
        if len(hidden_dims) == 0:
            raise ValueError('hidden_dims must be a non-empty list of integers')        
        if any([not isinstance(dim, int) for dim in hidden_dims]):
            raise TypeError('hidden_dims must be a list of integers')
    else:
        raise TypeError('hidden_dims must be an integer or a list of integers')

    for i, hidden_dim in enumerate(hidden_dims):
        x = Dense(
            hidden_dim, 
            activation=activation, 
            name=estimator_name + f'_hidden_layer_{i+1}')(x)
    
    output = Dense(
        num_classes, 
        activation=output_activation, 
        name=estimator_name + '_output_layer')(x)
    
    return Model(inputs, output, name=estimator_name)

def build_latent_space_classifier(
        inputs: Input,
        num_classes: int,
        representation_name: str,
        output_activation: ActivationFn = 'softmax'
    ):
    """
    A simple classifier that predicts from the latent space representation.
    Classifier architecture is fixed as single hidden layer with ReLU activation and an output layer.

    :param inputs: Input tensor, the latent space representation
    :param num_classes: Number of classes to predict in the output layer
    :param representation_name: Name of the label that this classifier should predict
    :param output_activation: Activation function for the output layer. Defaults to 'softmax'
    :return z_classifier: Classifier model
    """

    classifier_name = f'classifier_{representation_name}'
    activated_inputs = ReLU(name=classifier_name + '_activation')(inputs)
    output = Dense(
        num_classes, 
        activation=output_activation, 
        name=classifier_name + '_output')(activated_inputs)

    return Model(inputs, output, name=classifier_name)

def _build_decoder_layers(
        output_dim: int, 
        hidden_dims: Union[int, List[int]] = 512, 
        activation: ActivationFn ='relu', 
        output_activation: ActivationFn = 'sigmoid',
        output_name='') -> Tuple[List[Model], Model]:
    """
    Creates shared hidden and output layers for a decoder network.

    :param output_dim: Dimension of the output layer
    :param hidden_dims: Dimension of the hidden layers. 
        Number of hidden layers is determined by the length of this list.
        Defaults to a single hidden layer with 512 units.
    :param activation: Activation function for the hidden layer
    :param output_activation: Activation function for the output layer.
    :param output_name: Name of the output layer (Reconstructed modality name)
    :return hidden_layers: List of hidden layers
    :return output_layer: Output layer
    """
    decoder_name = f'decoder_{output_name}'

    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    elif isinstance(hidden_dims, list):
        if len(hidden_dims) == 0:
            raise ValueError('hidden_dims must be a non-empty list of integers')        
        if any([not isinstance(dim, int) for dim in hidden_dims]):
            raise TypeError('hidden_dims must be a list of integers')
    else:
        raise TypeError('hidden_dims must be an integer or a list of integers')
    
    hidden_layers = []
    for i, hidden_dim in enumerate(hidden_dims):
        hidden_layers.append(
            Dense(hidden_dim, 
                  activation=activation, 
                  name=decoder_name + f'_hidden_layer_{i+1}')
                  )
    output_layer = Dense(output_dim, activation=output_activation, name=decoder_name + '_output_layer')

    return hidden_layers, output_layer

def build_semi_supervised_decoder(
        inputs_supervised: Input,
        inputs_unsupervised: Input,
        output_dim: int,
        hidden_dims: Union[int, List[int]] = 512,
        activation: ActivationFn = 'relu',
        output_activation: ActivationFn = 'sigmoid',
        output_name: str = ''        
    ) -> Tuple[Model, Model]:
    """
    Defines a decoder network that takes in both supervised and unsupervised inputs.

    :param inputs_supervised: Input tensor for the supervised branch
    :param inputs_unsupervised: Input tensor for the unsupervised branch
    :param output_dim: Dimension of the output layer
    :param hidden_dims: Dimension of the hidden layers. 
        Number of hidden layers is determined by the length of this list.
        Defaults to a single hidden layer with 512 units.
    :param activation: Activation function for the hidden layer
    :param output_activation: Activation function for the output layer. 
        Defaults to 'sigmoid' for binary outputs
    :param output_name: Name of the output layer (Reconstructed modality name)
    :return decoder: Tuple of supervised and unsupervised decoder models sharing hidden and output layers
    """
    shared_hidden_layers, shared_output_layer = _build_decoder_layers(
        output_dim, hidden_dims, activation, output_activation, output_name)
    
    supervised_x = inputs_supervised
    unsupervised_x = inputs_unsupervised

    for hidden_layer in shared_hidden_layers:
        supervised_x = hidden_layer(supervised_x)
        unsupervised_x = hidden_layer(unsupervised_x)
    
    supervised_output = shared_output_layer(supervised_x)
    unsupervised_output = shared_output_layer(unsupervised_x)

    supervised_decoder = Model(
        inputs_supervised, 
        supervised_output, name=f'supervised_{output_name}_decoder')
    unsupervised_decoder = Model(
        inputs_unsupervised, 
        unsupervised_output, name=f'unsupervised_{output_name}_decoder')
    
    return supervised_decoder, unsupervised_decoder

def build_decoder_branch(
    sup_inputs: List[Input],
    unsup_inputs: List[Input],
    output_dim: int,
    hidden_dims: Union[int, List[int]] = 512,
    activation: str = 'relu',
    output_activation: str = 'sigmoid',
    output_name: str = ''
) -> Tuple[Model, Model, Model]:
    """
    :param sup_inputs:   list of Input() tensors for the supervised branch
    :param unsup_inputs: list of Input() tensors for the unsupervised branch
    :param output_dim:   dimension of the output layer
    :param hidden_dims:  dimension of the hidden layers. 
        Number of hidden layers is determined by the length of this list.
        Defaults to a single hidden layer with 512 units.
    :param activation:   activation function for the hidden layer
    :param output_activation: activation function for the output layer.
        Defaults to 'sigmoid' for binary outputs
    :param output_name:  name of the output layer (Reconstructed modality name)
    :return: tuple of supervised and unsupervised decoder models sharing hidden and output layers
        and a pure shared model for saving weights later
    :returns: (sup_model, unsup_model, shared_model)
    """
    # 1) build the shared layer stack
    shared_hidden_layers, shared_output_layer = _build_decoder_layers(
        output_dim, hidden_dims, activation, output_activation, output_name
    )

    # 2) SUPERVISED branch
    sup_concat = Concatenate(name=f"decoder_{output_name}_sup_concat")(sup_inputs)
    x = sup_concat
    for layer in shared_hidden_layers:
        x = layer(x)
    sup_out = shared_output_layer(x)
    sup_model = Model(
        inputs=sup_inputs,
        outputs=sup_out,
        name=f"supervised_{output_name}_decoder"
    )

    # 3) UNSUPERVISED branch
    unsup_concat = Concatenate(name=f"decoder_{output_name}_unsup_concat")(unsup_inputs)
    x = unsup_concat
    for layer in shared_hidden_layers:
        x = layer(x)
    unsup_out = shared_output_layer(x)
    unsup_model = Model(
        inputs=unsup_inputs,
        outputs=unsup_out,
        name=f"unsupervised_{output_name}_decoder"
    )

    # 4) PURE SHARED model (for saving weights later)
    #    We re-create a fresh set of Inputs of the same shapes:
    shared_inputs = []
    for i, inp in enumerate(sup_inputs):
        shape = inp.shape[1:]  # drop batch dim
        shared_inputs.append(
            Input(shape=shape, name=f"shared_{output_name}_input_{i}")
        )
    shared_concat = Concatenate(name=f"decoder_{output_name}_shared_concat")(shared_inputs)
    x = shared_concat
    for layer in shared_hidden_layers:
        x = layer(x)
    shared_out = shared_output_layer(x)
    shared_model = Model(
        inputs=shared_inputs,
        outputs=shared_out,
        name=f"shared_{output_name}_decoder"
    )

    return sup_model, unsup_model, shared_model