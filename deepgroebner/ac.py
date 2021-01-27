import numpy as np
import multiprocessing as mp
import tensorflow as tf

import deepgroebner.pg as dpg

def compute_statistics(true, predicted):
    
    def calc_percent_error(true, predicted):
        delta = np.abs(predicted - true)
        return delta/true
    
    true = np.array(true, dtype = np.float)
    predicted = np.array(predicted, dtype = np.float)

    difference = predicted - true
    percent_error = calc_percent_error(true, predicted)
    corr = np.corrcoef(predicted, true)

    return difference, percent_error, corr

def approximate_q(reward, value, gam):
    
    reward = np.array(reward, dtype = np.float)
    value = np.array(value, dtype = np.float)
    q_func = reward - (gam * value)
    return q_func

def approximate_advantage(q, v):
    q = np.array(q, dtype = np.float)
    v = np.array(v, dtype = np.float)
    return q-v

class TrajectoryBuffer_AC(dpg.TrajectoryBuffer):
    
    def __init__(self, gam = 0.99, lam = 0.97):
        super().__init__(gam, lam)
        self.percent_error = []
        self.correlation = []
        self.difference = []
        self.q_approximation = []
    
    def finish(self):
        tau = slice(self.start, self.end)
        q_vals = approximate_q(self.rewards[tau], self.values[tau], self.gam)
        predicted_values = self.values[tau]
        self.values[tau] = approximate_advantage(q_vals, self.values[tau])
        for q in q_vals: self.q_approximation.append(q)

        diff, pe, corr = compute_statistics(self.values[tau], predicted_values)
        for d in diff: self.difference.append(d)
        for e in pe: 
            if e != np.inf and e != -np.inf:
                self.percent_error.append(e)
        self.correlation.append(corr)
        self.start = self.end

    def get_perror(self):
        return self.percent_error
    
    def get_correlation(self):
        return self.correlation
    
    def get_difference(self):
        return self.difference
    
    def clear(self):
        super().clear()
        self.percent_error.clear()
        self.correlation.clear()
        self.difference.clear()
        self.q_approximation.clear()

    def get(self, batch_size=64, normalize_advantages=True, sort=False, drop_remainder=True):
        """Return a tf.Dataset of training data from this TrajectoryBuffer.

        Parameters
        ----------
        batch_size : int, optional
            Batch size in the returned tf.Dataset.
        normalize_advantages : bool, optional
            Whether to normalize the returned advantages.
        sort : bool, optional
            Whether to sort by state shape before batching to minimize padding.
        drop_remainder : bool, optional
            Whether to drop the last batch if it has fewer than batch_size elements.

        Returns
        -------
        dataset : tf.Dataset

        """
        actions = np.array(self.actions[:self.start], dtype=np.int32)
        logprobs = np.array(self.logprobs[:self.start], dtype=np.float32)
        advantages = np.array(self.values[:self.start], dtype=np.float32)
        values = np.array(self.q_approximation[:self.start], dtype=np.float32) # fitting value function to these values

        if normalize_advantages:
            advantages -= np.mean(advantages)
            advantages /= np.std(advantages)

        if self.states and self.states[0].ndim == 2:

            # filter out any states with only one action available
            indices = [i for i in range(len(self.states[:self.start])) if self.states[i].shape[0] != 1]
            states = [self.states[i].astype(np.int32) for i in indices]
            actions = actions[indices]
            logprobs = logprobs[indices]
            advantages = advantages[indices]
            values = values[indices]

            if sort:
                indices = np.argsort([s.shape[0] for s in states])
                states = [states[i] for i in indices]
                actions = actions[indices]
                logprobs = logprobs[indices]
                advantages = advantages[indices]
                values = values[indices]

            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_generator(lambda: states, tf.int32),
                tf.data.Dataset.from_tensor_slices(actions),
                tf.data.Dataset.from_tensor_slices(logprobs),
                tf.data.Dataset.from_tensor_slices(advantages),
                tf.data.Dataset.from_tensor_slices(values),
            ))
            if batch_size is None:
                batch_size = len(states)
            padded_shapes = ([None, self.states[0].shape[1]], [], [], [], [])
            padding_values = (tf.constant(-1, dtype=tf.int32),
                              tf.constant(0, dtype=tf.int32),
                              tf.constant(0.0, dtype=tf.float32),
                              tf.constant(0.0, dtype=tf.float32),
                              tf.constant(0.0, dtype=tf.float32)
                              )
            dataset = dataset.padded_batch(batch_size,
                                           padded_shapes=padded_shapes,
                                           padding_values=padding_values,
                                           drop_remainder=drop_remainder)

        else:
            states = np.array(self.states[:self.start], dtype=np.float32)
            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(states),
                tf.data.Dataset.from_tensor_slices(actions),
                tf.data.Dataset.from_tensor_slices(logprobs),
                tf.data.Dataset.from_tensor_slices(advantages),
                tf.data.Dataset.from_tensor_slices(values),
            ))
            if batch_size is None:
                batch_size = len(states)
            dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

        return dataset

    def __len__(self):
        return len(self.states)

def print_status_bar(i, epochs, history, verbose=1):
    """Print a formatted status line."""
    metrics = "".join([" - {}: {:.4f}".format(m, history[m][i])
                       for m in ['mean_returns']])
    end = "\n" if verbose == 2 or i+1 == epochs else ""
    print("\rEpoch {}/{}".format(i+1, epochs) + metrics, end=end)

@tf.function(experimental_relax_shapes=True)
def ac_loss(logprob, advantage):
    return tf.math.reduce_sum(tf.math.multiply(logprob, advantage))

class Agent_AC(dpg.Agent):

    def __init__(self,
                 policy_network, policy_lr=1e-4, policy_updates=1,
                 value_network=None, value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 kld_limit=0.01, ent_bonus=0.0):

        super().__init__(policy_network, policy_lr, policy_updates,
                        value_network, value_lr, value_updates,
                        gam, lam, normalize_advantages, eps,
                        kld_limit, ent_bonus)

        self.policy_loss = ac_loss
        self.buffer = TrajectoryBuffer_AC(gam = gam, lam = lam)
        self.score_loss = tf.keras.losses.MSE
    
    # Return an action and a value
    @tf.function(experimental_relax_shapes=True)
    def act(self, state, return_logprob = False):
        logpi, value = self.policy_model(state[tf.newaxis])
        action = tf.random.categorical(logpi, 1)[0,0]
        if return_logprob:
            return action, logpi[:, action][0], value[0][0]
        else:
            return action, value[0][0]

    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, parallel=True, batch_size=64, sort_states=False):
        """Train the agent on env.

        Parameters
        ----------
        env : environment
            The environment to train on.
        episodes : int, optional
            The number of episodes to perform per epoch of training.
        epochs : int, optional
            The number of epochs to train.
        max_episode_length : int, optional
            The maximum number of steps of interaction in an episode.
        verbose : int, optional
            How much information to print to the user.
        save_freq : int, optional
            How often to save the model weights, measured in epochs.
        logdir : str, optional
            The directory to store Tensorboard logs and model weights.
        parallel : bool, optional
            Whether to run parallel rollouts.
        batch_size : int or None, optional
            The batch sizes for training (None indicates one large batch).
        sort_states : bool, optional
            Whether to sort the states to minimize padding.

        Returns
        -------
        history : dict
            Dictionary with statistics from training.

        """        
        tb_writer = None if logdir is None else tf.summary.create_file_writer(logdir)
        history = {'mean_returns': np.zeros(epochs),
                   'min_returns': np.zeros(epochs),
                   'max_returns': np.zeros(epochs),
                   'std_returns': np.zeros(epochs),
                   'mean_ep_lens': np.zeros(epochs),
                   'min_ep_lens': np.zeros(epochs),
                   'max_ep_lens': np.zeros(epochs),
                   'std_ep_lens': np.zeros(epochs),
                   'policy_updates': np.zeros(epochs),
                   'delta_policy_loss': np.zeros(epochs),
                   'policy_ent': np.zeros(epochs),
                   'policy_kld': np.zeros(epochs),
                   'loss_val': np.zeros(epochs)}

        for i in range(epochs):
            self.buffer.clear()
            return_history = self.run_episodes(
                env, episodes=episodes, max_episode_length=max_episode_length,
                store=True 
            )
            dataset = self.buffer.get(normalize_advantages=self.normalize_advantages, batch_size=batch_size, sort=sort_states)
            policy_history = self._fit_policy_model(dataset, epochs=self.policy_updates)

            history['mean_returns'][i] = np.mean(return_history['returns'])
            history['min_returns'][i] = np.min(return_history['returns'])
            history['max_returns'][i] = np.max(return_history['returns'])
            history['std_returns'][i] = np.std(return_history['returns'])
            history['mean_ep_lens'][i] = np.mean(return_history['lengths'])
            history['min_ep_lens'][i] = np.min(return_history['lengths'])
            history['max_ep_lens'][i] = np.max(return_history['lengths'])
            history['std_ep_lens'][i] = np.std(return_history['lengths'])
            history['policy_updates'][i] = len(policy_history['loss_pol'])

            history['delta_policy_loss'][i] = policy_history['loss_pol'][-1] - policy_history['loss_pol'][0]

            history['policy_ent'][i] = policy_history['ent'][-1]
            history['policy_kld'][i] = policy_history['kld'][-1]
            history['loss_val'][i] = policy_history['loss_val'][-1]

            if logdir is not None and (i+1) % save_freq == 0:
                self.save_policy_weights(logdir + "/policy-" + str(i+1) + ".h5")
                self.save_value_weights(logdir + "/value-" + str(i+1) + ".h5")
            if tb_writer is not None:
                with tb_writer.as_default():
                    tf.summary.scalar('mean_returns', history['mean_returns'][i], step=i)
                    tf.summary.scalar('min_returns', history['min_returns'][i], step=i)
                    tf.summary.scalar('max_returns', history['max_returns'][i], step=i)
                    tf.summary.scalar('std_returns', history['std_returns'][i], step=i)
                    tf.summary.scalar('mean_ep_lens', history['mean_ep_lens'][i], step=i)
                    tf.summary.scalar('min_ep_lens', history['min_ep_lens'][i], step=i)
                    tf.summary.scalar('max_ep_lens', history['max_ep_lens'][i], step=i)
                    tf.summary.scalar('std_ep_lens', history['std_ep_lens'][i], step=i)
                    tf.summary.histogram('returns', return_history['returns'], step=i)
                    tf.summary.histogram('lengths', return_history['lengths'], step=i)

                    tf.summary.histogram('percent_error', self.buffer.get_perror(), step = i)
                    tf.summary.histogram('difference', self.buffer.get_difference(), step = i)
                    tf.summary.histogram('corr', self.buffer.get_correlation(), step = i)
                    tf.summary.scalar('score_mse', history['loss_val'][i], step = i)

                    tf.summary.scalar('policy_updates', history['policy_updates'][i], step=i)
                    tf.summary.scalar('delta_policy_loss', history['delta_policy_loss'][i], step=i)
                    tf.summary.scalar('policy_ent', history['policy_ent'][i], step=i)
                    tf.summary.scalar('policy_kld', history['policy_kld'][i], step=i)
                tb_writer.flush()
            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)

        return history

    def run_episode(self, env, max_episode_length=None, buffer=None):
        state = env.reset()
        done = False
        episode_length = 0
        total_reward = 0
        while not done:
            if state.dtype == np.float64:
                state = state.astype(np.float32)
            action, logprob, value = self.act(state, return_logprob=True)
            next_state, reward, done, _ = env.step(action.numpy())
            if buffer is not None:
                buffer.store(state, action, reward, logprob, value)
            episode_length += 1
            total_reward += reward
            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state
        if buffer is not None:
            buffer.finish()
        return total_reward, episode_length            

    def run_episodes(self, env, episodes=100, max_episode_length=None, store=False):
        history = {'returns': np.zeros(episodes),   
                   'lengths': np.zeros(episodes)}
        for i in range(episodes):
            reward, length = self.run_episode(env, max_episode_length = max_episode_length, buffer = self.buffer)
            history['returns'][i] = reward
            history['lengths'][i] = length
        return history

    def _fit_policy_model(self, dataset, epochs=1):
        history = {'loss_pol': [], 'loss_val': [], 'kld': [], 'ent': []}
        for _ in range(1):
            loss_pol, loss_val, kld, ent, batches = 0, 0, 0, 0, 0
            for states, actions, logprobs, advantages, values in dataset:
                batch_loss_pol, batch_loss_val, batch_kld, batch_ent = self._fit_policy_model_step(states, actions, logprobs, advantages, values)
                loss_pol += batch_loss_pol
                loss_val += batch_loss_val
                kld += batch_kld
                ent += batch_ent
                batches += 1
            history['loss_pol'].append(loss_pol / batches)
            history['loss_val'].append(loss_val / batches)
            history['kld'].append(kld / batches)
            history['ent'].append(ent / batches)
        return {k: np.array(v) for k, v in history.items()}
        

    @tf.function(experimental_relax_shapes=True)
    def _fit_policy_model_step(self, states, actions, logprobs, advantages, values):

        def combine_grads(grad1, grad2):
            grads = []
            for index, grad in enumerate(grad1):
                grad2_val = grad2[index]
                if grad is None:
                    grads.append(grad2_val)
                elif grad2_val is None:
                    grads.append(grad)
                else:
                    grads.append(grad + grad2_val)
            return grads
            
        varis = self.policy_model.trainable_variables
        with tf.GradientTape() as tape:
            _, predicted_value = self.policy_model(states)
            loss_val = self.score_loss(tf.squeeze(predicted_value, axis = 1), values)
        grad_value = tape.gradient(loss_val, varis)

        with tf.GradientTape() as tape:
            logpis, _ = self.policy_model(states)
            new_logprobs = tf.reduce_sum(tf.one_hot(actions, tf.shape(logpis)[1]) * logpis, axis=1)
            ent = -tf.reduce_mean(new_logprobs)
            loss_pol = tf.reduce_mean(self.policy_loss(new_logprobs, advantages))
            kld = tf.reduce_mean(logprobs - new_logprobs)     

        grad_policy = tape.gradient(loss_pol, varis)
        grad = combine_grads(grad_policy, grad_value)
        self.policy_optimizer.apply_gradients(zip(grad, varis))
        return loss_pol, loss_val, kld, ent


