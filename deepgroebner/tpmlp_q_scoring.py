import numpy as np
import tensorflow as tf

from deepgroebner.networks import ParallelEmbeddingLayer, ParallelDecidingLayer, \
                                    SelfAttentionLayer, TransformerLayer, TransformerPMLP

class SelfAttentionLayer_Score_Q(SelfAttentionLayer):
    """A multi head self attention layer.

    Adapted from https://www.tensorflow.org/tutorials/text/transformer.

    Parameters
    ----------
    dim : int
        Positive integer dimension.
    n_heads : int, optional
        Positive integer number of heads (must divide `dim`).

    """

    def __init__(self, dim, n_heads=1, context = False):
        super().__init__(dim, n_heads)
        if context:
            self.qval_learner = tf.keras.layers.GRU(dim)
        else:
            self.qval_learner = tf.keras.layers.Dense(dim) # Weird
        self.context = context
        layers = [tf.keras.layers.Dense(128, activation = 'sigmoid'),
                        tf.keras.layers.Dense(64, activation = 'sigmoid'),
                        tf.keras.layers.Dense(32, activation = 'sigmoid'),
                        tf.keras.layers.Dense(12, activation = 'sigmoid'),
                        tf.keras.layers.Dense(1, activation = 'relu')]
        self.scorer = tf.keras.Sequential(layers=layers)

    def call(self, batch, mask=None):
        """Return the processed batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, dim)
            Input batch with attached mask indicating valid rows.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, dim)
            Processed batch with mask passed through.

        """
        batch_size = tf.shape(batch)[0]
        Q = self.split_heads(self.Wq(batch), batch_size)
        K = self.split_heads(self.Wk(batch), batch_size)
        V = self.split_heads(self.Wv(batch), batch_size)
        Q_val = self.split_heads(self.get_qval(batch_size, batch = batch), batch_size)

        mask = mask[:, tf.newaxis, tf.newaxis, :]

        X = self.finish_attn(Q,K,V,batch_size,mask=mask)
        Y = self.finish_attn(Q_val,K,V,batch_size,mask=mask)

        output = self.dense(X)
        score = self.scorer(Y)

        return output, -(score[0]+1)
    
    def get_qval(self, batch_size, batch = None):
        if self.context:
            return self.qval_learner(batch)
        else:
            return self.qval_learner(tf.ones([batch_size, 1, 1]))

    def finish_attn(self, Q, K, V, batch_size, mask = None):
        X,_ = self.scaled_dot_product_attention(Q, K, V, mask=mask) # returns rewritten vectors and the softmax(QK^T)
        X = tf.transpose(X, perm=[0, 2, 1, 3])
        X = tf.reshape(X, (batch_size, -1, self.dim))
        return X

class TransformerLayer_Score_Q(TransformerLayer):
    """A transformer encoder layer.

    Parameters
    ----------
    dim : int
        Positive integer dimension of the attention layer and output.
    hidden_dim : int
        Positive integer dimension of the feed forward hidden layer.
    n_heads : int, optional
        Positive integer number of heads in attention layer (must divide `dim`).
    dropout : float, optional
        Dropout rate.

    """

    def __init__(self, dim, hidden_dim, n_heads=1, dropout=0.1):
        super().__init__(dim, hidden_dim, n_heads, dropout)
        self.attention = SelfAttentionLayer_Score_Q(dim, n_heads=n_heads)

    def call(self, batch, mask=None, training=False):
        """Return the processed batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, dim)
            Input batch with attached mask indicating valid rows.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, dim)
            Processed batch with mask passed through.

        """
        X1, score = self.attention(batch, mask=mask)
        X1 = self.dropout1(X1, training=training)
        X1 = self.layer_norm1(batch + X1)
        X2 = self.dense2(self.dense1(X1))
        X2 = self.dropout2(X2, training=training)
        output = self.layer_norm2(X1 + X2)
        return output, score

class TransformerPMLP_Score_Q(TransformerPMLP):
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

    def __init__(self, dim, hidden_dim, activation='relu', final_activation='log_softmax'):
        super().__init__(dim, hidden_dim, activation, final_activation)
        self.attn = TransformerLayer_Score_Q(dim, hidden_dim, n_heads=4)

    def call(self, batch):
        X = self.embedding(batch)
        X, score = self.attn(X)
        X = self.deciding(X)
        return X, score