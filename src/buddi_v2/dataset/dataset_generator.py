from typing import Optional, List, Dict

import numpy as np
import tensorflow as tf

def get_dataset(
    input_tuple_order: List[str] = ['X', 'Y_prop'],
    output_tuple_order: List[str] = ['X', 'z_label', 'z_stim', 'z_samp_type', 'z_slack', 'label', 'stim', 'samp_type', 'Y_prop'],
    dtypes: Optional[Dict[str, np.dtype]] = {'label': np.int32, 'stim': np.int32, 'samp_type': np.int32},
    **kwargs: Dict[str, np.ndarray] # inputs
):
    """
    Create a TensorFlow dataset from numpy arrays for training the BuDDI model.
    Uses _dataset_generator to create the dataset from the input numpy arrays
    Uses _dataset_output_signature to create the output signature for the dataset
    Handles the input checking for these helper functions

    :param input_tuple_order: List of keys to specify order of input tuple
    :param output_tuple_order: List of keys to specify order of output tuple
    :param dtypes: Dictionary of numpy dtypes for different modalities of data    
        Optional, if not provided, will use tf.float32 as default
        if partial dtypes are provided, will use tf.float32 for the rest
    :param kwargs: Dictionary of numpy arrays for different modalities of data
    :return: TensorFlow dataset
    """

    ## Input checking
    if not all (key in kwargs for key in input_tuple_order):
        raise ValueError("Required keys not found in kwargs")
    
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            pass
        else:
            raise TypeError(f"Expected numpy array, got {type(value)}")
    
    ## Dataset size checking
    num_samples = kwargs[input_tuple_order[0]].shape[0]
    if not all(kwargs[input].shape[0] == num_samples for input in input_tuple_order):
        raise ValueError("Inputs have different number of samples")
    
    ## Fill in missing dtypes with default tf.float32
    for key in input_tuple_order + output_tuple_order:
        if key in dtypes:
            continue
        else:
            dtypes[key] = tf.float32 # default datatype

    ## Create dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: _dataset_generator(input_tuple_order, output_tuple_order, num_samples, **kwargs),
        output_signature=_dataset_output_signature(input_tuple_order, output_tuple_order, dtypes, **kwargs),
    ).apply(tf.data.experimental.assert_cardinality(num_samples)) # set dataset size

    return dataset

def _dataset_generator(
    input_tuple_order: List[str],
    output_tuple_order: List[str],
    num_samples: int,
    **kwargs: Dict[str, np.ndarray] # inputs
):
    """
    Customizable dataset generator function for BuDDI
    Uses kwargs to receive the different modalities of data as numpy arrays
        and yields (input_tuple, output_tuple) for model training 
        according to `input_tuple_order` and `output_tuple_order` lists
    Currently requires all `input_tuple_order` to be present in kwargs and will return 
        a input_tuple according to that order specified by `input_tuple_order`
    `output_tuple_order` can include keys that are not present in kwargs, in which case
        the generator will return a placeholder tensor of shape (1,) for that key
        in the output_tuple for compatibility with loss functions that do not require
        a ground truth value (such as kl divergence loss and classifier loss on unsupervised data)

    :param input_tuple_order: List of keys to specify order of input tuple
    :param output_tuple_order: List of keys to specify order of output tuple
    :param num_samples: Number of samples in dataset
    :param kwargs: Dictionary of numpy arrays for different modalities of data
    :yield: Tuple of (inputs, outputs) for model training
    """
    for i in range(num_samples):
        yield (
            tuple(kwargs[key][i] for key in input_tuple_order),
            tuple(
                kwargs[key][i] if key in kwargs else tf.zeros((1,), dtype=tf.float32) \
                    for key in output_tuple_order
            )
        )

def _dataset_output_signature(
    input_tuple_order: List[str],
    output_tuple_order: List[str],
    dtypes: Optional[Dict[str, tf.DType]],
    **kwargs: Dict[str, np.ndarray] # inputs
):   
    """
    Create output signature for tensor BuDDI dataset to match 
    the generator input and output dimension, datatype and order.

    :param input_tuple_order: List of keys to specify order of input tuple
    :param output_tuple_order: List of keys to specify order of output tuple
    :param dtypes: Dictionary of tf.DType for different modalities of data
    :param kwargs: Dictionary of numpy arrays for different modalities of data
    :return: Tuple of input and output signature for dataset
    """
    output_signature = (tuple(
        tf.TensorSpec(shape=kwargs[key].shape[1:], dtype=dtypes[key]) for key in input_tuple_order
    ),
    tuple(
        tf.TensorSpec(shape=kwargs[key].shape[1:], dtype=dtypes[key]) \
            if key in kwargs else tf.TensorSpec(shape=(1,), dtype=dtypes[key]) \
                for key in output_tuple_order
    ))

    return output_signature