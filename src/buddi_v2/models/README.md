# BuDDI model
Currently, the only supported module architecture is buddi4 with 4 latent spaces + slack. 

## Overview

### Module file structure
```
.
├── buddi4.py              # Main model building and training
├── buddi4_class.py        # Class version of buddi 4 model for convenient save/load/retraining
├── components/            # Model components
│   ├── branches.py        # Functions for encoder and classifier branches
│   ├── layers.py          # Custom layers like reparameterization
│   ├── losses.py          # KL, reconstruction, and classifier loss functions
├── __init__.py            # Module initializer
└── README.md              # This file
```

### Important Functions/Class

- **`build_buddi4`**: the model instantiation function, with the following parameters:
    - `n_x`: number of genes (features in the input)
    - `n_y`: number of cell types (features in the output or target Y)
    - `n_labels`: number of unique sample identifiers (used in label classification branch)
    - `n_stims`: number of stimulation conditions
    - `n_samp_types`: number of sample types
    - `z_dim`: latent space dimension (default: 64)
    - `encoder_hidden_dim`: number of units in encoder hidden layers (default: 512)
    - `decoder_hidden_dim`: number of units in decoder hidden layers (default: 512)
    - `alpha_x`: weight of reconstruction loss
    - `alpha_label`: weight of label classification loss
    - `alpha_stim`: weight of stimulation classification loss
    - `alpha_samp_type`: weight of sample type classification loss
    - `alpha_prop`: weight of proportion estimator loss
    - `beta_kl_slack`: KL loss weight for the slack branch
    - `beta_kl_label`: KL loss weight for the label branch
    - `beta_kl_stim`: KL loss weight for the stimulation branch
    - `beta_kl_samp_type`: KL loss weight for the sample type branch
    - `reconstr_loss_fn`: reconstruction loss function (default: `MeanSquaredError`)
    - `classifier_loss_fn`: default classifier loss function (default: `CategoricalCrossentropy`)
    - `label_classifier_loss_fn`: optional custom loss function for the label classifier
    - `stim_classifier_loss_fn`: optional custom loss function for the stimulation classifier
    - `samp_type_classifier_loss_fn`: optional custom loss function for the sample type classifier
    - `prop_estimator_loss_fn`: optional custom loss function for the proportion estimator
    - `activation`: activation function to use (default: `'relu'`)
    - `optimizer`: Keras optimizer instance (default: `Adam(learning_rate=0.0005)`)
    - `return_decoder`: if `True`, also return the decoder models

    **Returns**:
    - If `return_decoder=False`: `(supervised_model, unsupervised_model)`
    - If `return_decoder=True`: `(supervised_model, unsupervised_model, supervised_decoder, unsupervised_decoder)`


- **`fit_buddi4`**: the BUDDI4 model training function
    - Trains both the supervised and unsupervised models in alternating batches.
    - Assumes the supervised dataset is larger than the unsupervised dataset.
    - Unsupervised dataset is repeated to match the number of supervised batches per epoch.
    - Returns a concatenated DataFrame of loss values from both supervised and unsupervised models.

    **Parameters**:
    - `supervised_model`: the supervised model returned from `build_buddi4`
    - `unsupervised_model`: the unsupervised model returned from `build_buddi4`
    - `dataset_supervised`: tf.data.Dataset with `(X, Y)` tuples (used for supervised learning)
    - `dataset_unsupervised`: tf.data.Dataset with `(X, Y)` where Y can be dummy or placeholder (used for unsupervised learning)
    - `epochs`: number of epochs to train (default: 10)
    - `batch_size`: batch size (default: 16)
    - `shuffle_every_epoch`: whether to reshuffle datasets every epoch (default: True)
    - `prefetch`: whether to prefetch batches using TensorFlow autotuning (default: False)

    **Returns**:
    - `pd.DataFrame`: a DataFrame of all losses per batch per epoch, with a `type` column indicating `supervised` or `unsupervised`

- **`BuDDI4`**: a class version of `build_buddi4/fit_buddi4` that exposes a pure setter-based loss API and separates model building from compilation.

    - `n_x`: number of genes (features in the input)
    - `n_y`: number of cell types (features in the output or target Y)
    - `n_labels`: number of unique sample identifiers (used in label classification branch)
    - `n_stims`: number of stimulation conditions
    - `n_samp_types`: number of sample types
    - `z_dim`: latent space dimension (default: 64)
    - `encoder_hidden_dim`: number of units in encoder hidden layers (default: 512)
    - `decoder_hidden_dim`: number of units in decoder hidden layers (default: 512)
    - `activation`: activation function for encoder and classifier hidden layers (default: `'relu'`)
    - `output_activation`: activation function for decoder and classifier output layers (default: `'sigmoid'`)

    **Returns**:
    - A `BuDDI4` class instance with model parts (`encoders`, `classifiers`, `prop_estimator`, `decoder`) constructed but not yet compiled.

- **`compile`**: compile the supervised and unsupervised models for training.

    **Parameters**:
    - `optimizer`: a Keras optimizer instance (default: `Adam(learning_rate=0.0005)`)

    **Returns**:
    - Compiled supervised (`sup_model`) and unsupervised (`unsup_model`) Keras models accessible via properties.

- **Loss setters**: methods for flexible loss assignment before compiling.
    
    - **`set_reconstruction_loss(fn, weight)`**: set reconstruction loss for decoder.
    - **`set_encoder_loss(branch, fn, weight)`**: set KL divergence loss for a specific latent space encoder branch.
    - **`set_all_encoder_losses(fn, weight)`**: set the same KL divergence loss for all encoder branches.
    - **`set_predictor_loss(branch, fn, weight)`**: set classification loss for a specific latent space classifier.
    - **`set_all_predictor_losses(fn, weight)`**: set the same classification loss for all latent space classifiers.
    - **`set_prop_estimator_loss(fn, weight)`**: set loss function for the bulk proportion estimator.

- **Properties**: attributes accessible after building or compiling.

    - `encoder_branch_names`: list of names of encoder branches (`['label', 'stim', 'samp_type']`)
    - `config`: dictionary of model hyperparameters.
    - `sup_model`: compiled supervised Keras model (available after `compile()`).
    - `unsup_model`: compiled unsupervised Keras model (available after `compile()`).
    - `decoder`: the shared decoder model.
    - `prop_estimator`: the bulk proportion estimator model.
    - `encoders`: dictionary of encoder branches (models).
    - `classifiers`: dictionary of latent space classifier branches (models).
    - `reparam_layers`: dictionary of reparameterization layers.
    - `history`: placeholder for training history (currently not used).

- **`save(directory)`**: saves the model weights and configuration to the specified directory.

    **Parameters**:
    - `directory`: path to save model configuration and weights.

- **`load(directory)`**: loads a saved BuDDI4 model from disk.

    **Parameters**:
    - `directory`: path to load model configuration and weights from.

    **Returns**:
    - `BuDDI4` instance reconstructed with loaded weights and configuration.

- **Notes**:
    - `fit()` is intentionally not implemented inside the class. Use external `fit_buddi4()` function for training.
    - `save()` and `load()` methods handle only model architecture (`config.json`) and weights (`.h5` files), not optimizer states.

## Usage
Function version
```python
from buddi_v2.models.buddi4 import build_buddi4, fit_buddi4

# See module `dataset` for details in dataset construction
dataset_supervised = ... 
dataset_unsupervised = ...

# Build model
supervised_model, unsupervised_model = build_buddi4(
    n_x=5000,
    n_y=10,
    n_labels=4,
    n_stims=3,
    n_samp_types=2
)

# Fit model
loss_df = fit_buddi4(
    supervised_model,
    unsupervised_model,
    dataset_supervised,
    dataset_unsupervised,
    epochs=20, batch_size=16
)
```
Class version
```python
from tensorflow.keras.losses import CategoricalCrossentropy, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

from buddi_v2.models.components.losses import unsupervised_dummy_loss_fn
from buddi_v2.models.buddi4 import fit_buddi4
from buddi_v2.models.buddi4_class import BuDDI4

# See module `dataset` for details in dataset construction
dataset_supervised = ... 
dataset_unsupervised = ...

obj = BuDDI4(
    n_x=5000,
    n_y=10,
    n_labels=4,
    n_stims=3,
    n_samp_types=2,
)

# configure reconstruction loss
obj.set_reconstruction_loss(
    fn=MeanAbsoluteError(),
    weight=1.0,
)

# configure kl loss
obj.set_encoder_loss(
    branch='label',
    fn=unsupervised_dummy_loss_fn,
    weight=100.0,    
)
obj.set_encoder_loss(
    branch='slack',
    fn=unsupervised_dummy_loss_fn,
    weight=1000.0,    
)

# configure classifier loss
obj.set_predictor_loss(
    branch='label',
    fn=CategoricalCrossentropy,
    weight=100.0
)

# configure prop estimator loss
obj.set_prop_estimator_loss(
    fn=MeanAbsoluteError,
    weight=100.0,
)

model.compile(optimizer=Adam(learning_rate=0.0005))

# Fit model by
loss_df = fit_buddi4(
    obj.sup_model, obj.unsup_model, 
    dataset_supervised,
    dataset_unsupervised,
    epochs=20, batch_size=16
)
```