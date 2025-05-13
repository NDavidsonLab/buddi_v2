from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class ReparameterizationLayer(Layer):
    """
    Custom layer that applies the reparameterization trick.
    Takes concatenated [z_mu, z_log_var] as input and samples z.
    """

    def __init__(self, name="reparameterization_layer", **kwargs):
        """
        Initializes the reparameterization layer.
        
        :param name: Name of the layer.
        :param kwargs: Additional keyword arguments for the base Layer class.
        """
        super(ReparameterizationLayer, self).__init__(name=name, **kwargs)
        self.__deterministic: bool = False
        self.__seed: Optional[int] = None # really the base seed for deterministic behavior
        self.__call_counter: int = 0 # doubles as secondary seed, allows for deterministic but non-identical epsilon

    def call(self, z_params, training=None):
        """
        Performs the reparameterization trick.

        :param z_params: Tensor of shape (batch_size, 2 * z_dim), where:
            - First half contains z_mu (mean)
            - Second half contains z_log_var (log variance)
        :param training: Boolean, whether in training mode.
        :return: Sampled latent variable z of shape (batch_size, z_dim)
        """
        # Split input into mu and log_var
        n_z_dim = tf.shape(z_params)[-1] // 2
        z_mu, z_log_var = z_params[:, :n_z_dim], z_params[:, n_z_dim:]

        # Compute standard deviation
        z_sigma = K.exp(0.5 * z_log_var)

        if self.__deterministic:
            seed_tensor = tf.constant([self.__seed, self.__call_counter], dtype=tf.int32)
            epsilon = tf.random.stateless_normal(
                shape=tf.shape(z_mu),
                seed=seed_tensor
            )
            self.__call_counter += 1
        else:
            # Sample from standard normal distribution
            epsilon = K.random_normal(shape=tf.shape(z_mu))

        # Reparameterization trick: z = mu + sigma * epsilon
        return z_mu + z_sigma * epsilon
    
    def set_deterministic(self, deterministic: bool, seed: Optional[int] = None):
        """
        Set the layer to deterministic mode.

        :param deterministic: Boolean indicating if the layer should be deterministic.
        """
        if deterministic:
            if seed is None:
                raise ValueError("Seed must be provided for deterministic mode.")
            elif not isinstance(seed, int):
                raise ValueError("Seed must be an integer.")
            self.__seed = seed
        else:
            self.__seed = None       
        
        self.__deterministic = deterministic
        self.__call_counter = 0 # reset call counter as secondary seed component