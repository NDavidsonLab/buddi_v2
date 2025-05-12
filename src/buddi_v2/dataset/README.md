# BuDDI dataset

This module provides utilities for constructing supervised and unsupervised TensorFlow `datasets` tailored to training the BuDDI models with defined return signature and support for built-in tensorflow dataset operations such as shuffling and down-sampling.

(For now, only datasets for buddi4 can be easily generated with one-line function call).

**Note** that this module is intended for use with **training** only. Visualization utilities expect the raw data modalities (e.g. expression, proportion as numpy 2d arrays and metadata as pandas dataframes).

## Overview

### Module file structure
```
.
├── buddi4_dataset.py       # High-level dataset builders for buddi4
├── dataset_generator.py    # Core dataset generator and signature logic
├── data_split.py           # (Optional) dataset splitting utilities
├── __init__.py             # Module initializer
└── README.md               # This file
```

### Important functions

#### `dataset_generator.py`
- **`get_dataset`**: low-level customizable dataset constructor
    - `input_tuple_order`: list of keys defining the order of model inputs (default: `['X', 'Y_prop']`)
    - `output_tuple_order`: list of keys defining order of model outputs (default includes latent and label outputs)
    - `dtypes`: optional dictionary mapping each key to a NumPy dtype (defaults to float32 if not provided)
    - `**kwargs`: key-value pairs of NumPy arrays, where keys match the elements in `input_tuple_order` and `output_tuple_order`. The use may specify dummy targets that will be returned as a dimension 1 scalar of 0 by specifying a name in `output_tuple_order` and not including it in `**kwargs`

    **Returns**: `tf.data.Dataset` object with shape and type signatures matching the provided arrays


#### `buddi4_dataset.py`
- **`get_supervised_dataset`**: prepares a ready to use dataset for buddi4 from pseudobulk data with known cell-type proportions
    - `X_known_prop`: expression matrix (n_samples × n_genes)
    - `Y_known_prop`: ground-truth proportions (n_samples × n_cell_types)
    - `label_known_prop`: one-hot encoded sample IDs (n_samples × n_unique_labels)
    - `stim_known_prop`: one-hot encoded stimulation IDs (n_samples × n_stimulations)
    - `samp_type_known_prop`: one-hot sample type labels (n_samples × 2), typically [1, 0] for single-cell and [0, 1] for bulk

- **`get_unsupervised_dataset`**: prepares a ready to use dataset for buddi4 from pseudobulk samples **without** known cell-type proportions
    - `X_unknown_prop`: expression matrix (n_samples × n_genes)
    - `label_unknown_prop`: one-hot encoded sample IDs (n_samples × n_unique_labels)
    - `stim_unknown_prop`: one-hot encoded stimulation IDs (n_samples × n_stimulations)
    - `samp_type_unknown_prop`: one-hot sample type labels (n_samples × 2), typically [0, 1] for bulk

## Usage
```python
import numpy as np
from buddi_v2.dataset.buddi4_dataset import get_supervised_dataset, get_unsupervised_dataset

## prepare and format data as numpy 2d arrays for the supervised module
n_samples = 10000
n_genes = 5000
n_cells = 8
n_labels = 5
n_stims = 2
n_samp_types = 2

X_kp = np.random.rand(n_samples, n_genes) # normalized sample by gene matrix
y_kp = np.random.rand(n_samples, n_genes) # row normalized sample by cell type proportion

# one hot encoded metadata
label_kp = np.eye(n_labels)[np.random.randint(0, n_labels, size=n_samples)]
drug_kp = np.eye(n_stims)[np.random.randint(0, n_stims, size=n_samples)]
bulk_kp = np.eye(n_samp_types)[np.random.randint(0, n_samp_types, size=n_samples)]

ds_sup = get_supervised_dataset(
    X_known_prop=X_kp,
    Y_known_prop=y_kp,
    label_known_prop=label_kp,
    stim_known_prop=drug_kp,
    samp_type_known_prop=bulk_kp,
)

## prepare and format data as numpy 2d arrays for the unsupervised module

n_samples = 500 # the unsupervised data can have a different number of samples
n_genes = 5000
n_labels = 5
n_stims = 2
n_samp_types = 2

X_kp = np.random.rand(n_samples, n_genes) # normalized sample by gene matrix

# one hot encoded metadata
label_kp = np.eye(n_labels)[np.random.randint(0, n_labels, size=n_samples)]
drug_kp = np.eye(n_stims)[np.random.randint(0, n_stims, size=n_samples)]
bulk_kp = np.eye(n_samp_types)[np.random.randint(0, n_samp_types, size=n_samples)]

ds_unsup = get_unsupervised_dataset(
    X_unknown_prop=X_unkp,
    label_unknown_prop=label_unkp,
    stim_unknown_prop=drug_unkp,
    samp_type_unknown_prop=bulk_unkp,
)

## Check dataset size by
print(f"Number of entries in supervised dataset: {ds_sup.cardinality().numpy()}")
print(f"Number of entries in unsupervised dataset: {ds_unsup.cardinality().numpy()}")

## Check the output signature by
ds_sup_batch_input, ds_sup_batch_target = next(iter(ds_sup.batch(16).take(1)))
ds_sup_batch_x, ds_sup_batch_y = ds_sup_batch_input
print(f"Supervised batch x shape: {ds_sup_batch_x.shape}")
print(f"Supervised batch y shape: {ds_sup_batch_y.shape}")
ds_sup_batch_target_x, _, _, _, _, ds_sup_batch_label, ds_sup_batch_stim, ds_sup_batch_samp_type, ds_sup_batch_target_y = ds_sup_batch_target
print(f"Supervised batch target x shape: {ds_sup_batch_target_x.shape}")
print(f"Supervised batch target y shape: {ds_sup_batch_target_y.shape}")
print(f"Supervised batch label shape: {ds_sup_batch_label.shape}")
print(f"Supervised batch stim shape: {ds_sup_batch_stim.shape}")
print(f"Supervised batch samp_type shape: {ds_sup_batch_samp_type.shape}")

ds_unsup_batch_input, ds_unsup_batch_target = next(iter(ds_unsup.batch(16).take(1)))
ds_unsup_batch_x, = ds_unsup_batch_input
print(f"Unsupervised batch x shape: {ds_unsup_batch_x.shape}")
ds_unsup_batch_target_x, _, _, _, _, ds_unsup_batch_label, ds_unsup_batch_stim, ds_unsup_batch_samp_type, _ = ds_unsup_batch_target
print(f"Unsupervised batch target x shape: {ds_unsup_batch_target_x.shape}")
print(f"Unsupervised batch label shape: {ds_unsup_batch_label.shape}")
print(f"Unsupervised batch stim shape: {ds_unsup_batch_stim.shape}")
print(f"Unsupervised batch samp_type shape: {ds_unsup_batch_samp_type.shape}")
```