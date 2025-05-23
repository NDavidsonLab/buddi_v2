from typing import Union, Callable, Optional, List, Dict, Tuple
import os
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError, CategoricalCrossentropy

from .components.wrapped_branches import (
    wrapped_build_encoder_branch,
    wrapped_build_decoder_branch,
    wrapped_build_prop_estimator_branch,
    wrapped_build_latent_space_classifier_branch
)
from .components.layers import ReparameterizationLayer
from .components.losses import (
    kl_loss_generator,
    kl_loss,
    reconstr_loss_generator,
    classifier_loss_generator,
    unsupervised_dummy_loss_fn
)
from .buddi_abstract_class import BuDDIAbstract

ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]

class BuDDI3(BuDDIAbstract):
    """
    BuDDI3 model class
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        z_dim: int = 64,
        encoder_hidden_dim: int = 512,
        decoder_hidden_dim: int = 512,
        activation: ActivationFn = 'relu',
        output_activation: ActivationFn = 'relu',
        encoder_branch_names: Optional[List[str]] = None,
        slack_branch_name: Optional[str] = None,
        **kwargs # to collect the n_{branch_name} arguments
    ):
        """
        BuDDI3 model class constructor

        :param n_x: Number of input features
        :param n_y: Number of output features
        :param z_dim: Latent space dimension
        :param encoder_hidden_dim: Hidden dimension of the encoder
        :param decoder_hidden_dim: Hidden dimension of the decoder
        :param activation: Activation function for the encoder and decoder
        :param output_activation: Output activation function for the decoder
        :param encoder_branch_names: List of encoder branch names
        :param slack_branch_name: Name of the slack branch
        :param kwargs: Meant for additional dimension arguments for the encoder branches
        """
        super().__init__(
            n_x=n_x,
            n_y=n_y,
            z_dim=z_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            activation=activation,
            output_activation=output_activation,
            encoder_branch_names=encoder_branch_names,
            slack_branch_name=slack_branch_name,
            **kwargs
        )

    # ─── Checker functions ───────────────────────────────────────────
    def default_encoder_branch_names(self):
        return ['sample_id', 'samp_type']
    
    def check_encoder_branch_names(
        self,
        encoder_branch_names: List[str]
    ):
        """
        Check if the encoder branch names are valid.
        Subclasses should modify this method for rigorous input checks

        :param encoder_branch_names: List of encoder branch names
        """
        if len(encoder_branch_names) != 2:
            raise ValueError(
                f"Expected 0 encoder branch names, got {len(encoder_branch_names)}"
            )
        elif not all(isinstance(name, str) for name in encoder_branch_names):
            raise ValueError(
                f"Expected encoder branch names to be strings, got {type(encoder_branch_names[0])}"
            )
    
    def check_slack_branch_name(
        self,
        slack_branch_name: str
    ):
        """
        Check if the slack branch name is valid.
        Subclasses should modify this method for rigorous input checks

        :param slack_branch_name: Name of the slack branch
        """
        if not isinstance(slack_branch_name, str):
            raise ValueError(
                f"Expected slack branch name to be a string, got {type(slack_branch_name)}"
            )
        if slack_branch_name in self.encoder_branch_names:
            raise ValueError(
                f"Slack branch name {slack_branch_name} cannot be in encoder branch names"
            )
        
    # ─── Fit ────────────────────────────────────────────────────

    def fit(self, **kwargs):
        raise NotImplementedError(
            "fit() not implemented."
        )
    
    # ─── Save / Load ────────────────────────────────────────────
    def save(self, directory: str):
        """
        Save the model to the specified directory.

        :param directory: Directory to save the model to
        """
        os.makedirs(directory, exist_ok=True)
        # save config only

        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(self.config, f)

        # save weights
        for name, model in self.encoders.items():
            model.save_weights(os.path.join(directory, f'{name}_encoder.weights.h5'))
        for name, model in self.classifiers.items():
            model.save_weights(os.path.join(directory, f'{name}_classifier.weights.h5'))
        self.decoder.save_weights(os.path.join(directory, 'decoder.weights.h5'))
        self.prop_estimator.save_weights(os.path.join(directory, 'prop_estimator.weights.h5'))

    @classmethod
    def load(cls, directory: str) -> 'BuDDI3':
        """
        Load the model from the specified directory and reconstruct the model class
        from the saved config

        :param directory: Directory to load the model from
        :return: BuDDI class object with loaded weights
        """
        config_file = os.path.join(directory, 'config.json')
        if not os.path.exists(config_file):
            # returns None if the config file does not exist         
            return None
        
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        obj = cls(**cfg)

        for name in obj.encoder_branch_names:
            obj._encoders[name].load_weights(os.path.join(directory, f'{name}_encoder.weights.h5'))
            obj._classifiers[name].load_weights(os.path.join(directory, f'{name}_classifier.weights.h5'))
        obj._encoders[
            obj._slack_branch_name
        ].load_weights(os.path.join(directory, f'{obj._slack_branch_name}_encoder.weights.h5'))

        obj._decoder.load_weights(os.path.join(directory, 'decoder.weights.h5'))
        obj._prop_estimator.load_weights(os.path.join(directory, 'prop_estimator.weights.h5'))
        
        return obj