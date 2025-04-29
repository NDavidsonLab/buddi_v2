from typing import Tuple, Union, Callable, Optional, Dict
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

from .components.branches import (
    build_encoder_branch,
    build_latent_space_classifier,
    build_prop_estimator,
    build_decoder_branch
)
from .components.layers import ReparameterizationLayer
from .components.losses import (
    kl_loss_generator,
    reconstr_loss_generator,
    classifier_loss_generator,
    unsupervised_dummy_loss_fn
)

ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]

class BuDDI4:
    """
    Thin wrapper around the original build_buddi4/fit_buddi4 but with
    a pure setter-based loss API and compile pulled out of __init__.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        n_labels: int,
        n_stims: int,
        n_samp_types: int,
        z_dim: int = 64,
        encoder_hidden_dim: int = 512,
        decoder_hidden_dim: int = 512,
        activation: Union[str, Callable] = 'relu',
        output_activation: Union[str, Callable] = 'sigmoid',
    ):
        
        self.__encoder_branch_names = [
            'label', 'stim', 'samp_type'
        ]
        self.__slack_branch_name = 'slack'
        
        # 1. Store architecture-only config
        self.__config = dict(
            n_x=n_x, n_y=n_y,
            n_labels=n_labels,
            n_stims=n_stims,
            n_samp_types=n_samp_types,
            z_dim=z_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            activation=activation,
            output_activation=output_activation
        )

        # 2. Default loss config registry; user can call setters before compile() to update
        self.__losses: Dict[str, Tuple[Callable, float]] = {
            'x_hat_sup': (MeanAbsoluteError(), 1.0),
            'x_hat_unsup': (MeanAbsoluteError(), 1.0),
            'kl_label': (kl_loss_generator(), 1.0),
            'kl_stim': (kl_loss_generator(), 1.0),
            'kl_samp_type': (kl_loss_generator(), 1.0),
            'kl_slack': (kl_loss_generator(), 1.0),
            'label_pred': (CategoricalCrossentropy(), 1.0),
            'stim_pred': (CategoricalCrossentropy(), 1.0),
            'samp_type_pred': (CategoricalCrossentropy(), 1.0),
            'y_hat': (CategoricalCrossentropy(), 1.0),
            'y_dummy': (unsupervised_dummy_loss_fn, 0.0),
        }

        # 3. Placeholder for model parts
        self.__input_layers = {}
        self.__output_layers = {}

        self.__encoders: Dict[str, Model] = {}
        self.__classifiers: Dict[str, Model] = {}
        self.__resamp_layers: Dict[str, Callable] = {}
        self.__prop_estimator: Optional[Model] = None
        self.__decoder: Optional[Model] = None

        # 3. Placeholders for compiled models + decoder + history
        self.__sup_model: Optional[Model] = None
        self.__unsup_model: Optional[Model] = None
        self.__history = None

        # 4. Build the model 
        self.__build_models()

    # ─── Properties ───────────────────────────────────────────

    @property
    def encoder_branch_names(self):
        return self.__encoder_branch_names
    
    @property
    def config(self):
        return self.__config
    
    @property
    def sup_model(self):
        if self.__sup_model is None:
            raise RuntimeError("You must call .compile() before .sup_model")
        return self.__sup_model
    
    @property
    def unsup_model(self):
        if self.__unsup_model is None:
            raise RuntimeError("You must call .compile() before .unsup_model")
        return self.__unsup_model   

    @property
    def history(self):
        return self.__history

    @property
    def decoder(self):
        return self.__decoder
    
    @property
    def prop_estimator(self):
        return self.__prop_estimator
    
    @property
    def encoders(self):
        return self.__encoders
    
    @property
    def reparam_layers(self):
        return self.__resamp_layers
    
    @property
    def classifiers(self):
        return self.__classifiers

    # ─── Loss setters ───────────────────────────────────────────

    def set_reconstruction_loss(self, fn: Callable, weight: float=1.0):
        self.__losses['x_hat_sup'] = (fn, weight)
        self.__losses['x_hat_unsup'] = (fn, weight)

    def set_encoder_loss(self, branch: str, fn: Callable, weight: float=1.0):
        self.__losses[f'kl_{branch}'] = (fn, weight)

    def set_all_encoder_losses(self, fn: Callable, weight: float=1.0):
        for branch in self.encoder_branch_names:
            self.set_encoder_loss(branch, fn, weight)

        self.set_encoder_loss(self.__slack_branch_name, fn, weight)

    def set_predictor_loss(self, branch: str, fn: Callable, weight: float=1.0):
        self.__losses[f'{branch}_pred'] = (fn, weight)

    def set_all_predictor_losses(self, fn: Callable, weight: float=1.0):
        for branch in self.encoder_branch_names:
            self.set_predictor_loss(branch, fn, weight)

    def set_prop_estimator_loss(self, fn: Callable, weight: float=1.0):
        self.__losses['y_hat'] = (fn, weight)

    # ─── Build Model ────────────────────────────────────────────

    def __build_models(self):

        n_x = self.config['n_x']
        n_y = self.config['n_y']
        n_labels = self.config['n_labels']
        n_stims = self.config['n_stims']
        n_samp_types = self.config['n_samp_types']
        z_dim = self.config['z_dim']
        encoder_hidden_dim = self.config['encoder_hidden_dim']
        decoder_hidden_dim = self.config['decoder_hidden_dim']
        activation = self.config['activation']
        output_activation = self.config['output_activation']

        X = Input(shape=(n_x,), name='X')
        Y = Input(shape=(n_y,), name='Y')
        self.__input_layers['X'] = X
        self.__input_layers['Y'] = Y

        # 1) encoder branch building
        for name in self.encoder_branch_names + [self.__slack_branch_name]:
            
            self.__encoders[name] = build_encoder_branch(
                inputs = X,
                hidden_dim = encoder_hidden_dim,
                z_dim = z_dim,
                activation = activation,
                representation_name = name
            )

            z_param = self.__encoders[name](X)
            self.__output_layers[f'z_{name}_param'] = z_param

            self.__resamp_layers[name] = ReparameterizationLayer(
                name=f'z_{name}_resamp')
            
            z = self.__resamp_layers[name](z_param)
            self.__output_layers[f'z_{name}'] = z

            if name != self.__slack_branch_name:
                self.__classifiers[name] = build_latent_space_classifier(
                    inputs = z,
                    num_classes = self.config[f'n_{name}s'],
                    representation_name = name,
                    output_activation = output_activation
                )
                pred = self.__classifiers[name](z)
                self.__output_layers[f'{name}_pred'] = pred
                
        # 2) prop estimator building
        self.__prop_estimator = build_prop_estimator(
            inputs = X,
            num_classes=n_y,
            activation=activation,
            estimator_name='prop_estimator',
        )

        y_hat = self.__prop_estimator(X)
        self.__output_layers['y_hat'] = y_hat

        # 3) decoder building

        self.__decoder = build_decoder_branch(
            y = Y,
            z_label = self.__output_layers['z_label'],
            z_stim = self.__output_layers['z_stim'],
            z_samp_type = self.__output_layers['z_samp_type'],
            z_slack = self.__output_layers['z_slack'],
            output_dim=n_x,
            decoder_hidden_dim=decoder_hidden_dim,
            activation=activation,
            output_activation=output_activation,
            name='decoder'
        )

        x_hat_sup = self.__decoder(
            (Y, 
            self.__output_layers['z_label'],
            self.__output_layers['z_stim'],
            self.__output_layers['z_samp_type'],
            self.__output_layers['z_slack'])
        )
        self.__output_layers['x_hat_sup'] = x_hat_sup

        x_hat_unsup = self.__decoder(
            (self.__output_layers['y_hat'], 
            self.__output_layers['z_label'],
            self.__output_layers['z_stim'],
            self.__output_layers['z_samp_type'],
            self.__output_layers['z_slack'])
        )
        self.__output_layers['x_hat_unsup'] = x_hat_unsup

    # ─── Compile ────────────────────────────────────────────────

    def compile(
            self, 
            optimizer: Optimizer = None
            ):
        if optimizer is None:
            optimizer = Adam(5e-4)

        # wire supervised model for training
        inputs_sup = [
            self.__input_layers['X'],
            self.__input_layers['Y'],
        ]
        outputs_sup = [
            self.__output_layers['x_hat_sup'],
            self.__output_layers['z_label_param'],
            self.__output_layers['z_stim_param'],
            self.__output_layers['z_samp_type_param'],
            self.__output_layers['z_slack_param'],
            self.__output_layers['label_pred'],
            self.__output_layers['stim_pred'],
            self.__output_layers['samp_type_pred'],
            self.__output_layers['y_hat'],
        ]
        model_sup = Model(
            inputs=inputs_sup,
            outputs=outputs_sup,
            name='supervised_buddi4'
        )       

        sup_loss_keys = [
            'x_hat_sup',
            'kl_label', 
            'kl_stim', 
            'kl_samp_type', 
            'kl_slack',
            'label_pred', 
            'stim_pred',
            'samp_type_pred',
            'y_hat'
        ]
        model_sup.compile(
            optimizer=optimizer,
            loss=[self.__losses[k][0] for k in sup_loss_keys],
            loss_weights=[self.__losses[k][1] for k in sup_loss_keys]
        )        
        
        # wire unsupervised model for training
        inputs_unsup = [
            self.__input_layers['X'],
        ]
        outputs_unsup = [
            self.__output_layers['x_hat_unsup'],
            self.__output_layers['z_label_param'],
            self.__output_layers['z_stim_param'],
            self.__output_layers['z_samp_type_param'],
            self.__output_layers['z_slack_param'],
            self.__output_layers['label_pred'],
            self.__output_layers['stim_pred'],
            self.__output_layers['samp_type_pred'],
            self.__output_layers['y_hat'],
        ]
        model_unsup = Model(
            inputs=inputs_unsup,
            outputs=outputs_unsup,
            name='unsupervised_buddi4'
        )

        unsup_loss_keys = [
            'x_hat_unsup',
            'kl_label', 
            'kl_stim', 
            'kl_samp_type', 
            'kl_slack',
            'label_pred', 
            'stim_pred',
            'samp_type_pred',
            'y_dummy'
        ]
        model_unsup.compile(
            optimizer=optimizer,
            loss=[self.__losses[k][0] for k in unsup_loss_keys],
            loss_weights=[self.__losses[k][1] for k in unsup_loss_keys]
        )

        # save as attributes
        self.__sup_model   = model_sup
        self.__unsup_model = model_unsup

    # ─── Fit ────────────────────────────────────────────────────

    def fit(self, **kwargs):
        raise NotImplementedError(
            "fit() not implemented. Use fit_buddi4() instead."
        )

    # ─── Save / Load ────────────────────────────────────────────

    def save(self, directory: str):
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
    def load(cls, directory: str) -> 'BuDDI4':
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            cfg = json.load(f)
        obj = cls(**cfg)

        for name in obj.encoder_branch_names:
            obj.__encoders[name].load_weights(os.path.join(directory, f'{name}_encoder.weights.h5'))
            obj.__classifiers[name].load_weights(os.path.join(directory, f'{name}_classifier.weights.h5'))
        obj.__encoders['slack'].load_weights(os.path.join(directory, 'slack_encoder.weights.h5'))

        obj.__decoder.load_weights(os.path.join(directory, 'decoder.weights.h5'))
        obj.__prop_estimator.load_weights(os.path.join(directory, 'prop_estimator.weights.h5'))
        
        return obj