import numpy as np
import tensorflow as tf

from deepgroebner.networks import TransformerPMLP

class Score(tf.keras.Model):

    """Score function to estimate value."""

    def __init__(self, hidden_layers:list, activation = 'relu'):
        super(Score, self).__init__()
        self.encode = tf.keras.layers.GRU(128)
        self.ff = [tf.keras.layers.Dense(dim, activation = activation) for dim in hidden_layers]
        self.v = tf.keras.layers.Dense(1)
    
    def call(self, batch):
        X = self.encode(batch)
        for layer in self.ff:
            X = layer(X)
        additions_left = self.v(X)
        return additions_left

class TransformerPMLP_Score_MHA(TransformerPMLP):
    """A parallel multilayer perceptron network with a transformer layer.

    This model expects an input with shape (batch_dim, padded_dim, feature_dim), where
    entries are non-negative integers and padding is by -1. It returns a tensor
    of shape (batch_dim, padded_dim) where each batch is a softmaxed distribution over the rows
    with zero probability on any padded row.

    Parameters
    ----------
    dim : int
        Positive integer dimension of the transformer attention layer.
    hidden_dim : int
        Positive integer dimension of the transformer hidden feedforward layer.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation for the embedding.
    final_activation : {'log_softmax', 'softmax'}, optional
        Activation for the final output layer.

    """

    def __init__(self, score_layers:list, dim, hidden_dim, num_layers = 1):
        super().__init__(dim, hidden_dim, num_layers)
        self.score = Score(score_layers)

    def call(self, batch):
        X = self.embedding(batch)
        X = self.attn(X)
        Y = self.score(X)
        X = self.deciding(X)
        return X, -(Y+1)