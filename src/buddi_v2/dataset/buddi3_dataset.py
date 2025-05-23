import numpy as np
import tensorflow as tf

from .dataset_generator import get_dataset

def get_supervised_dataset(
    X_known_prop: np.ndarray,
    Y_known_prop: np.ndarray,
    label_known_prop: np.ndarray,
    samp_type_known_prop: np.ndarray,
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from numpy arrays catered to the buddi3 model specifically.
    Meant to be constructed with dataset from pseudobulks where expression data have associated
        ground truth proportions of cell types.

    :param X_known_prop: numpy array of normalized expression of shape (n, num_genes)
    :param Y_known_prop: numpy array of ground truth pseudobulk proportions of shape (n, num_cell_types)
    :param label_known_prop: numpy array of one hot encoded sample labels of shape (n, num_unique_samples)
    :param samp_type_known_prop: numpy array of one hot encoded sample type (sequencing tech) labels of shape (n, num_unique_sample_types)
        Usually for supervised dataset this should be a 2darray of shape (n, 2), 
        with one column should be all 1s (encoding for single cell) and the other all 0s (encoding for bulk)
    :return: TensorFlow dataset
    :rtype: tf.data.Dataset
    """

    dataset = get_dataset(
        input_tuple_order=['X', 'Y_prop'],
        output_tuple_order=['X', 'z_label', 'z_samp_type', 'z_slack', 'label', 'samp_type', 'Y_prop'],
        X=X_known_prop,
        Y_prop=Y_known_prop,
        label=label_known_prop,
        samp_type=samp_type_known_prop,
    )

    return dataset

def get_unsupervised_dataset(
    X_unknown_prop: np.ndarray,
    label_unknown_prop: np.ndarray,
    samp_type_unknown_prop: np.ndarray
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from numpy arrays catered to the buddi3 model specifically.
    :param X_unknown_prop: numpy array of normalized expression of shape (n, num_genes)
    :param label_unknown_prop: numpy array of one hot encoded sample labels of shape (n, num_unique_samples)
    :param samp_type_unknown_prop: numpy array of one hot encoded sample type (sequencing tech) labels of shape (n, num_unique_sample_types)
        Usually for supervised dataset this should be a 2darray of shape (n, 2), 
        with one column should be all 1s (encoding for bulk) and the other all 0s (encoding for single cell)
    :return: TensorFlow dataset
    :rtype: tf.data.Dataset
    """
    dataset = get_dataset(
        input_tuple_order=['X'],
        output_tuple_order=['X', 'z_label', 'z_samp_type', 'z_slack', 'label', 'samp_type', 'Y_dummy'],
        X=X_unknown_prop,
        label=label_unknown_prop,
        samp_type=samp_type_unknown_prop,
    )

    return dataset