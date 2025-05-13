from typing import List, Tuple, Union, Optional

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .branches import (
    build_encoder_branch,
    build_decoder_branch,
    build_prop_estimator,
    build_latent_space_classifier
)

# -------------------------------------------------------------------
# wrapped_build_encoder_branch
# -------------------------------------------------------------------
def wrapped_build_encoder_branch(
    input_shape: int,
    hidden_dim:  int,
    z_dim:       int,
    activation:  str = 'relu',
    representation_name: str = ''
) -> Tuple[Input, Model]:
    """
    Factory wrapper around build_encoder_branch.

    :param input_shape:          Shape-tuple (excluding batch dim) for the encoder input
    :param hidden_dim:           Dimension of the hidden layer in the encoder
    :param z_dim:                Dimension of the latent space to produce
    :param activation:           Activation function for the hidden layer
    :param representation_name:  Name suffix for layer/model naming
    :return: Tuple containing:
        - encoder_input:        Keras Input() tensor for the encoder
        - encoder_model:        Keras Model for the encoder branch
    """
    encoder_input = Input(shape=(input_shape,), name=f"{representation_name}_encoder_input")
    encoder_model = build_encoder_branch(
        inputs=encoder_input,
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        activation=activation,
        representation_name=representation_name
    )
    return encoder_model

# -------------------------------------------------------------------
# wrapped_build_decoder_branch
# -------------------------------------------------------------------
def wrapped_build_decoder_branch(
    sup_input_shapes:   List[int],
    unsup_input_shapes: List[int],
    output_dim:         int,
    hidden_dims:        Union[int, List[int]] = 512,
    activation:         str = 'relu',
    output_activation:  str = 'sigmoid',
    output_name:        str = '',
    sup_input_names: Optional[List[str]] = None,
    unsup_input_names: Optional[List[str]] = None
) -> Tuple[
     List[Input],  # sup placeholders
     List[Input],  # unsup placeholders
     Model,        # sup_decoder
     Model,        # unsup_decoder
     Model         # shared_decoder
]:
    """
    Factory wrapper around build_decoder_branch.

    :param sup_input_shapes:   List of shape-tuples (excluding batch dim) for supervised inputs
    :param unsup_input_shapes: List of shape-tuples for unsupervised inputs
    :param output_dim:         Dimension of the output layer
    :param hidden_dims:        Dimension of the hidden layers; number of layers determined by length of list
    :param activation:         Activation function for the hidden layers
    :param output_activation:  Activation function for the output layer
    :param output_name:        Name suffix for layer/model naming
    :return: Tuple containing:
        - sup_inputs:   List of Input() tensors for the supervised branch
        - unsup_inputs: List of Input() tensors for the unsupervised branch
        - sup_model:    Keras Model for supervised decoding
        - unsup_model:  Keras Model for unsupervised decoding
        - shared_model: Keras Model wrapping the shared decoder stack for saving weights
    """
    # 1) build fresh Input() layers
    if sup_input_names is None:
        sup_input_names = [f"input_{i}" for i in range(len(sup_input_shapes))]
    else:
        if len(sup_input_names) != len(sup_input_shapes):
            raise ValueError("Length of sup_input_names must match length of sup_input_shapes.")
    if unsup_input_names is None:
        unsup_input_names = [f"input_{i}" for i in range(len(unsup_input_shapes))]
    else:
        if len(unsup_input_names) != len(unsup_input_shapes):
            raise ValueError("Length of unsup_input_names must match length of unsup_input_shapes.")
        
    sup_inputs = [
        Input(shape=(shape,), name=f"supervised_decoder_{name}")
        for i, (shape, name) in enumerate(zip(sup_input_shapes, sup_input_names))
    ]
    unsup_inputs = [
        Input(shape=(shape,), name=f"unsupervised_decoder_{name}")
        for i, (shape, name) in enumerate(zip(unsup_input_shapes, unsup_input_names))
    ]

    # 2) delegate to existing decoder builder
    sup_model, unsup_model, shared_model = build_decoder_branch(
        sup_inputs,
        unsup_inputs,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
        output_name=output_name
    )

    return sup_model, unsup_model, shared_model


# -------------------------------------------------------------------
# wrapped_build_prop_estimator_branch
# -------------------------------------------------------------------
def wrapped_build_prop_estimator_branch(
    input_shape:       int,
    num_classes:       int,
    hidden_dims:       Union[int, List[int]] = [512, 256],
    activation:        str = 'relu',
    output_activation: str = 'softmax',
    estimator_name:    str = 'prop_estimator'
) -> Tuple[Input, Model]:
    """
    Factory wrapper around build_prop_estimator.

    :param input_shape:       Shape-tuple (excluding batch dim) for the prop estimator input
    :param num_classes:       Number of output classes for classification
    :param hidden_dims:       Dimension(s) for the hidden layers
    :param activation:        Activation function for the hidden layers
    :param output_activation: Activation function for the output layer
    :param estimator_name:    Name suffix for layer/model naming
    :return: Tuple containing:
        - prop_input: Keras Input() tensor for the prop estimator
        - prop_model: Keras Model for property estimation
    """
    prop_input = Input(shape=(input_shape,), name=f"{estimator_name}_input")
    prop_model = build_prop_estimator(
        prop_input,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
        estimator_name=estimator_name
    )
    return prop_model


# -------------------------------------------------------------------
# wrapped_build_latent_space_classifier_branch
# -------------------------------------------------------------------
def wrapped_build_latent_space_classifier_branch(
    latent_shape:      int,
    num_classes:       int,
    output_activation: str = 'softmax',
    representation_name:str = ''
) -> Tuple[Input, Model]:
    """
    Factory wrapper around build_latent_space_classifier.

    :param latent_shape:      Shape-tuple (excluding batch dim) for the latent representation input
    :param num_classes:       Number of output classes for classification
    :param output_activation: Activation function for the classification output layer
    :param representation_name:
                              Name suffix for layer/model naming
    :return: Tuple containing:
        - z_input:  Keras Input() tensor for the latent classifier
        - clf_model: Keras Model for latent-space classification
    """
    z_input = Input(shape=(latent_shape,), name=f"z_classifier_{representation_name}_input")
    clf_model = build_latent_space_classifier(
        z_input,
        num_classes=num_classes,
        output_activation=output_activation,
        representation_name=representation_name
    )
    return clf_model