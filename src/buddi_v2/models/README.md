# BuDDI model
Currently, the only supported module architecture is buddi4 with 4 latent spaces + slack. 

## Overview

### Module file structure
```
.
├── buddi4.py              # Main model building and training
├── components/            # Model components
│   ├── branches.py        # Functions for encoder and classifier branches
│   ├── layers.py          # Custom layers like reparameterization
│   ├── losses.py          # KL, reconstruction, and classifier loss functions
├── __init__.py            # Module initializer
└── README.md              # This file
```

### Important Functions

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

## Usage
```python
from buddi4 import build_buddi4, fit_buddi4

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