"""Neural networks for agents.

The two network classes are designed to be fast wrappers around tf.keras models.
In particular, they store their weights in NumPy arrays and do predict calls in
pure NumPy, which in testing is at least on order of magnitude faster than
TensorFlow when called repeatedly.
"""

import numpy as np
import scipy.special as sc
import tensorflow as tf


class MultilayerPerceptron(tf.keras.Model):
    """A basic multilayer perceptron network.

    Parameters
    ----------
    output_dim : int
        The output positive integer dimension of the network.
    hidden_layers : list
        The list of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        The activation used for the hidden layers.
    final_activation : {'log_softmax', 'softmax', 'linear', 'exponential'}
        The activation used for the final output layer.

    Examples
    --------
    >>> import tensorflow as tf
    >>> mlp = MultilayerPerceptron(2, [128])
    >>> states = tf.random.uniform((64, 4))
    >>> logprobs = mlp(states)
    >>> logprobs.shape
    TensorShape([64, 2])
    >>> actions = tf.random.categorical(logprobs, 1)
    >>> actions.shape
    TensorShape([64, 1])

    """

    def __init__(self, output_dim, hidden_layers, activation='relu', final_activation='log_softmax'):
        super(MultilayerPerceptron, self).__init__()
        final_activation = tf.nn.log_softmax if final_activation == 'log_softmax' else final_activation
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(output_dim, activation=final_activation)

    def call(self, X):
        for layer in self.hidden_layers:
            X = layer(X)
        X = self.final_layer(X)
        return X


class ParallelMultilayerPerceptron:
    """A parallel multilayer perceptron network with fast predict calls."""

    def __init__(self, input_dim, hidden_layers):
        self.network = self._build_network(input_dim, hidden_layers)
        self.weights = self.get_weights()
        self.trainable_variables = self.network.trainable_variables

    def predict(self, X, **kwargs):
        for i, (m, b) in enumerate(self.weights):
            X = np.dot(X, m) + b
            if i == len(self.weights)-1:
                X = sc.log_softmax(X, axis=1).squeeze(axis=-1)
            else:
                X = np.maximum(X, 0, X)
        return X

    def __call__(self, inputs):
        return self.network(inputs)[0]

    def get_logits(self, inputs):
        return self.network(inputs)[1]

    def save_weights(self, filename):
        self.network.save_weights(filename)

    def load_weights(self, filename):
        self.network.load_weights(filename)
        self.weights = self.get_weights()

    def get_weights(self):
        network_weights = self.network.get_weights()
        self.weights = []
        for i in range(len(network_weights)//2):
            m = network_weights[2*i].squeeze(axis=0)
            b = network_weights[2*i + 1]
            self.weights.append((m, b))
        return self.weights

    def _build_network(self, input_dim, hidden_layers):
        inputs = tf.keras.Input(shape=(None, input_dim))
        x = inputs
        for hidden in hidden_layers:
            x = tf.keras.layers.Conv1D(hidden, 1, activation='relu')(x)
        x = tf.keras.layers.Conv1D(1, 1, activation='linear')(x)
        outputs = tf.nn.log_softmax(x, axis=1)
        logprobs = tf.keras.layers.Flatten()(outputs)
        return tf.keras.Model(inputs=inputs, outputs=[logprobs, outputs])


class PairsLeftBaseline:
    """A Buchberger value network that returns discounted pairs left."""

    def __init__(self, gam=0.99):
        self.gam = gam
        self.trainable_variables = []

    def predict(self, X, **kwargs):
        states, pairs, *_ = X.shape
        if self.gam == 1:
            fill_value = - pairs
        else:
            fill_value = - (1 - self.gam ** pairs) / (1 - self.gam)
        return np.full((states, 1), fill_value)

    def __call__(self, inputs):
        return self.predict(inputs)

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass

    def get_weights(self):
        pass


class AgentBaseline:
    """A Buchberger value network that returns an agent's performance."""

    def __init__(self, agent, gam=0.99):
        self.agent = agent
        self.gam = gam
        self.trainable_variables = []

    def predict(self, env):
        env = env.copy()
        R = 0.0
        discount = 1.0
        state = (env.G, env.P) if hasattr(env, 'G') else env._matrix()
        done = False
        while not done:
            action = self.agent.act(state)
            state, reward, done, _ = env.step(action)
            R += reward * discount
            discount *= self.gam
        return R

    def __call__(self, inputs):
        return self.predict(inputs)

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass

    def get_weights(self):
        pass
