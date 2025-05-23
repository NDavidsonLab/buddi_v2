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

ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]

class BuDDI3:
    """
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
        _encoder_branch_names: Optional[List[str]] = None,
        _slack_branch_name: Optional[str] = None,
        **kwargs # to collect the n_{branch_name} arguments
    ):
        """
        """

        if _encoder_branch_names is None:
            # default encoder branch names
            self.__encoder_branch_names = [
                'sample_id', 'samp_type'
            ]
        else:
            if len(_encoder_branch_names) != 2:
                raise ValueError(
                    f"Expected 2 encoder branch names, got {len(_encoder_branch_names)}"
                )
            elif not all(isinstance(name, str) for name in _encoder_branch_names):
                raise ValueError(
                    f"Expected encoder branch names to be strings, got {type(_encoder_branch_names[0])}"
                )
            self.__encoder_branch_names = _encoder_branch_names

        if _slack_branch_name is None:
            self.__slack_branch_name = 'slack'
        else:
            if not isinstance(_slack_branch_name, str):
                raise ValueError(
                    f"Expected slack branch name to be a string, got {type(_slack_branch_name)}"
                )
            self.__slack_branch_name = _slack_branch_name

        # 1. Store architecture config
        self.__config = dict(
            n_x=n_x, n_y=n_y,
            z_dim=z_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            activation=activation,
            output_activation=output_activation,
            _encoder_branch_names = self.__encoder_branch_names,
            _slack_branch_name = self.__slack_branch_name,
        )
        ## dynamically collect the n_{branch_name} arguments and add to config
        branch_dim_dict = {}
        for branch_name in self.__encoder_branch_names:
            if f'n_{branch_name}s' not in kwargs:
                raise ValueError(
                    f"Missing n_{branch_name}s in kwargs for encoder branch {branch_name}"
                )
            else:
                branch_dim_dict[f'n_{branch_name}s'] = kwargs[f'n_{branch_name}s']
        self.__config.update(branch_dim_dict)

        # 2. Default loss config registry; 
        # user can call setters before compile() to update
        self.__losses: Dict[str, Tuple[Callable, float]] = {
            # reconstruction loss of expression
            'x_hat_sup': (MeanAbsoluteError(reduction='sum'), 1.0), # supervised
            'x_hat_unsup': (MeanAbsoluteError(reduction='sum'), 1.0), # unsupervised
            # estimation loss of cell type proportion
            'y_hat': (CategoricalCrossentropy(reduction='sum'), 1.0),
            # dummy loss that returns 0 always for unsupervised branch training
            'y_dummy': (unsupervised_dummy_loss_fn, 0.0),
        }
        ## add kl and classifier loss configuration for each supervised encoder branch
        for branch_name in self.__encoder_branch_names:
            self.__losses[f'kl_{branch_name}'] = (kl_loss, 1.0)
            self.__losses[f'{branch_name}_pred'] = (
                CategoricalCrossentropy(reduction='sum'), 1.0
            )
        ## add slack branch kl loss
        # note slack kl has smaller weight to allow for better learning 
        # of additional variation
        self.__losses[f'kl_{self.__slack_branch_name}'] = (kl_loss, 0.1) 
        
        # 3. Container for model parts
        ## encoder branches goes here indexed by name
        self.__encoders: Dict[str, Model] = {} 
        ## classifiers corresponding to supervised encoder branches goes here
        #       indexed by name
        self.__classifiers: Dict[str, Model] = {}
        ## resampling layers for each encoder branch goes here
        #       indexed by name
        self.__resamp_layers: Dict[str, Callable] = {}
        ## proportion estiamtor
        self.__prop_estimator: Optional[Model] = None
        ## decoder branch
        self.__decoder: Optional[Model] = None

        ## Placeholder for compiled models for training
        #   and loss dataframe
        self.__sup_model: Optional[Model] = None
        self.__unsup_model: Optional[Model] = None
        self.__history: pd.DataFrame = None 

        # 4. Build the model 
        self.__build_models()

    # ─── Properties ───────────────────────────────────────────

    @property
    def encoder_branch_names(self):
        return self.__encoder_branch_names
    
    @property
    def losses(self):
        return self.__losses

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
        """
        Set the reconstruction loss function for the decoder.

        :param fn: Reconstruction loss function
        :param weight: Weight of the reconstruction loss
        """
        self.__losses['x_hat_sup'] = (fn, weight)
        self.__losses['x_hat_unsup'] = (fn, weight)

    def set_encoder_loss(
            self, 
            branch_name: Union[str, List[str]], 
            fn: Callable, 
            weight: float=1.0
        ):
        """
        Set the KL divergence loss function for the encoder branch.
        :param branch_name: Name of the encoder branch
        :param fn: KL divergence loss function
        :param weight: Weight of the KL divergence loss
        """
        if isinstance(branch_name, str):
            branch_name = [branch_name]
        
        for _branch_name in branch_name:

            if _branch_name not in self.__encoder_branch_names + [self.__slack_branch_name]:
                raise ValueError(
                    f"Branch name {_branch_name} not in encoder branch names: {self.__encoder_branch_names}"
                )

            self.__losses[f'kl_{_branch_name}'] = (fn, weight)

    def set_all_encoder_losses(self, fn: Callable, weight: float=1.0):
        """
        Set the KL divergence loss function for all encoder branches.

        :param fn: KL divergence loss function
        :param weight: Weight of the KL divergence loss
        """
        self.set_encoder_loss(
            self.encoder_branch_names + [self.__slack_branch_name], 
            fn, weight)

    def set_predictor_loss(
            self, 
            branch_name: Union[str, List[str]], 
            fn: Callable, 
            weight: float=1.0
        ):
        """
        Set the predictor loss function for the encoder branch.

        :param branch: Name of the encoder branch
        :param fn: Predictor loss function
        :param weight: Weight of the predictor loss
        """
        if isinstance(branch_name, str):
            branch_name = [branch_name]
        for _branch_name in branch_name:
            if _branch_name not in self.__encoder_branch_names:
                raise ValueError(
                    f"Branch name {_branch_name} not in encoder branch names: {self.__encoder_branch_names}"
                )

            self.__losses[f'{_branch_name}_pred'] = (fn, weight)

    def set_all_predictor_losses(self, fn: Callable, weight: float=1.0):
        """
        Set the predictor loss function for all encoder branches.

        :param fn: Predictor loss function
        :param weight: Weight of the predictor loss
        """
        self.set_predictor_loss(self.encoder_branch_names, fn, weight)

    
    def set_prop_estimator_loss(self, fn: Callable, weight: float=1.0):
        """
        Set the loss function for the prop estimator.
        
        :param fn: Loss function for the prop estimator
        :param weight: Weight of the prop estimator loss
        """
        self.__losses['y_hat'] = (fn, weight)

    # ─── Deterministic behavior Switch ────────────────────────────────────────────
    
    def set_reparam_deterministic(
            self, 
            deterministic: bool, 
            seed: Optional[int] = None):
        """
        Set the reparameterization layer to deterministic mode.
        :param deterministic: Boolean indicating if the layer should be deterministic.
        :param seed: Seed for the random number generator
        """

        for name, layer in self.__resamp_layers.items():
            try:
                layer.set_deterministic(deterministic, seed)
            except Exception as e:
                print(f"Error setting deterministic mode for {name}: {e}")

    # ─── Build Model ────────────────────────────────────────────

    def __build_models(self):
        """
        Internal helper method to build model parts
        """

        n_x = self.config['n_x']
        n_y = self.config['n_y']
        z_dim = self.config['z_dim']
        encoder_hidden_dim = self.config['encoder_hidden_dim']
        decoder_hidden_dim = self.config['decoder_hidden_dim']
        activation = self.config['activation']
        output_activation = self.config['output_activation']

        # 1) building encoder branches
        for branch_name in self.__encoder_branch_names + [self.__slack_branch_name]:
            
            self.__encoders[branch_name] = wrapped_build_encoder_branch(
                input_shape=n_x,
                hidden_dim=encoder_hidden_dim,
                z_dim=z_dim,
                activation=activation,
                representation_name=branch_name
            )

            self.__resamp_layers[branch_name] = ReparameterizationLayer(
                name=f'z_{branch_name}_resamp')
            
            ## only build classifier for supervised branches (non slack)
            if branch_name != self.__slack_branch_name:
                self.__classifiers[branch_name] = wrapped_build_latent_space_classifier_branch(
                    latent_shape = z_dim,
                    num_classes = self.config[f'n_{branch_name}s'],
                    representation_name = branch_name,
                    # output activation of classifier should always be softmax
                    # for classifiers (currently assumes exclusive classes)
                    # TODO: support for multi-class classification?
                    output_activation = 'softmax'
                )

        # 2) building proportion estimator
        self.__prop_estimator = wrapped_build_prop_estimator_branch(
            input_shape = n_x,
            num_classes=n_y,
            hidden_dims = [512, 256], # hard-coded value for now
            activation = activation,
            # output activation of prop estimator should always be softmax
            output_activation = 'softmax', 
            estimator_name = 'prop_estimator'
        )

        # 3) building decoder branch
        decoder_input_shapes = \
        [n_y] + \
        [
            z_dim for _ in range(len(self.encoder_branch_names) + 1)
        ]
        (
            self.__sup_decoder,
            self.__unsup_decoder,
            self.__decoder # decoder model containing the shared layers
        ) = wrapped_build_decoder_branch(
            sup_input_shapes=decoder_input_shapes,
            unsup_input_shapes=decoder_input_shapes,
            output_dim=n_x,
            hidden_dims=decoder_hidden_dim,
            activation=activation,
            output_activation=output_activation,
            output_name='X'
        )

        # ─── Compile ────────────────────────────────────────────────

    def compile(
        self, 
        optimizer: Optimizer = None
    ):
        """
        Compile the model with the specified optimizer and loss functions.

        :param optimizer: Optimizer to use for training
        """
        if optimizer is None:
            optimizer = Adam(5e-4)

        n_x = self.config['n_x']
        n_y = self.config['n_y']

        # 1a) wire supervised model for training
        sup_X = Input(shape=(n_x,), name='sup_X')
        sup_Y = Input(shape=(n_y,), name='sup_Y')
        sup_z_params = []
        sup_zs = []
        sup_z_preds = []
        for name in self.encoder_branch_names + [self.__slack_branch_name]:
            z_param = self.__encoders[name](sup_X)
            sup_z_params.append(z_param)
            z = self.__resamp_layers[name](z_param)
            sup_zs.append(z)
            if name in self.__classifiers:
                z_pred = self.__classifiers[name](z)
                sup_z_preds.append(z_pred)

        sup_Y_hat = self.__prop_estimator(sup_X)

        sup_X_hat = self.__sup_decoder(
            [sup_Y] + sup_zs
        )

        model_sup = Model(
            inputs=[sup_X, sup_Y],
            outputs=[sup_X_hat] + sup_z_params + sup_z_preds + [sup_Y_hat],
        )

        # 1b) stitch together the list of loss function keys 
        # corresponding to the outputs of supervised model
        sup_loss_keys = ['x_hat_sup'] + \
            [f'kl_{name}' for name in self.encoder_branch_names] + \
            [f'kl_{self.__slack_branch_name}'] + \
            [f'{name}_pred' for name in self.encoder_branch_names] + \
            ['y_hat']
        model_sup.compile(
            optimizer=optimizer,
            loss=[self.__losses[k][0] for k in sup_loss_keys],
            loss_weights=[self.__losses[k][1] for k in sup_loss_keys]
        )

        # 2a) wire unsupervised model for training
        unsup_X = Input(shape=(n_x,), name='unsup_X')
        unsup_z_params = []
        unsup_zs = []
        unsup_z_preds = []
        for name in self.encoder_branch_names + [self.__slack_branch_name]:
            z_param = self.__encoders[name](unsup_X)
            unsup_z_params.append(z_param)
            z = self.__resamp_layers[name](z_param)
            unsup_zs.append(z)
            if name in self.__classifiers:
                z_pred = self.__classifiers[name](z)
                unsup_z_preds.append(z_pred)
        
        unsup_Y_hat = self.__prop_estimator(unsup_X)
        unsup_X_hat = self.__unsup_decoder(
            [unsup_Y_hat] + unsup_zs
        )

        model_unsup = Model(
            inputs=[unsup_X],
            outputs=[unsup_X_hat] + unsup_z_params + unsup_z_preds + [unsup_Y_hat],
        )

        # 2b) stitch together the list of loss function keys
        # corresponding to the outputs of unsupervised model
        unsup_loss_keys = ['x_hat_unsup'] + \
            [f'kl_{name}' for name in self.encoder_branch_names] + \
            [f'kl_{self.__slack_branch_name}'] + \
            [f'{name}_pred' for name in self.encoder_branch_names] + \
            ['y_dummy']
        model_unsup.compile(
            optimizer=optimizer,
            loss=[self.__losses[k][0] for k in unsup_loss_keys],
            loss_weights=[self.__losses[k][1] for k in unsup_loss_keys]
        )

        # update as class attributes
        self.__sup_model   = model_sup
        self.__unsup_model = model_unsup

    # ─── Convenient Information Access ──────────────────────────────────────

    def get_loss_summary(self):
        """
        """

        fn_names = []
        w_strs = []
        for name, (fn, w) in self.__losses.items():
            fn_name = type(fn).__name__
            if fn_name == 'function':
                fn_name = fn.__name__
            fn_names.append(fn_name)
            w_strs.append(str(w))

        return pd.DataFrame(
            {
                'Loss Name': list(self.__losses.keys()),
                'Function': fn_names,
                'Weight': w_strs
            }
        )

    def print_loss_table(self):
        """
        Print a table of the loss functions and their weights.
        """
        # compute column widths
        fn_names = []
        w_strs = []
        for name, (fn, w) in self.__losses.items():
            fn_name = type(fn).__name__
            if fn_name == 'function':
                fn_name = fn.__name__
            fn_names.append(fn_name)
            w_strs.append(str(w))
        
        name_w   = max(len(k) for k in self.__losses) + 2
        fn_w     = max(len(name) for name in fn_names) + 2
        weight_w = max(len(w) for w in w_strs) + 2

        # header
        header = f"{'Loss Name':<{name_w}}{'Function':<{fn_w}}{'Weight':<{weight_w}}"
        sep    = '-' * len(header)

        print(header)
        print(sep)
        # rows
        for name, (fn, weight) in self.__losses.items():
            fn_name = type(fn).__name__
            if fn_name == 'function':
                fn_name = fn.__name__
            print(f"{name:<{name_w}}{fn_name:<{fn_w}}{weight:<{weight_w}}")

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
        :return: BuDDI3 class object with loaded weights
        """
        config_file = os.path.join(directory, 'config.json')
        if not os.path.exists(config_file):
            # returns None if the config file does not exist         
            return None
        
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        obj = cls(**cfg)

        for name in obj.encoder_branch_names:
            obj.__encoders[name].load_weights(os.path.join(directory, f'{name}_encoder.weights.h5'))
            obj.__classifiers[name].load_weights(os.path.join(directory, f'{name}_classifier.weights.h5'))
        obj.__encoders[
            obj.__slack_branch_name
        ].load_weights(os.path.join(directory, f'{obj.__slack_branch_name}_encoder.weights.h5'))

        obj.__decoder.load_weights(os.path.join(directory, 'decoder.weights.h5'))
        obj.__prop_estimator.load_weights(os.path.join(directory, 'prop_estimator.weights.h5'))
        
        return obj