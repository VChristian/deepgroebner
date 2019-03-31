# pg.py
# Dylan Peifer
# 14 Mar 2019
"""Basic policy gradient agents."""

import numpy as np
import tensorflow as tf


class ExperienceBuffer:
    """A buffer to store trajectories."""
    
    def __init__(self, capacity, state_shape, action_dim, gamma=0.99, rewards_to_go=True, normalize=True):
        self.states = np.empty([capacity, state_shape])
        self.actions = np.empty([capacity, action_dim])
        self.rewards = np.empty(capacity)
        self.index = 0
        self.start = 0
    
    def store(self, state, action, reward):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.index += 1
    
    def finishTrajectory(self):
        if self.rewards_to_go:
            for i in reversed(range(self.start, self.index)):
                cumulative_reward = self.rewards[i] + self.gamma * cumulative_reward
                rewards[i] = cumulative_reward
        self.index += 1
        self.start = self.index
    
    def get(self):
        states = self.states[:index]
        rewards = self.rewards[:index]
        if self.normalize:
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        return states, rewards


class PGAgent:
    """A basic policy gradient agent."""

    def __init__(self, network, learning_rate=0.00025, gamma=0.99, rewards_to_go=True, normalize=True):
        self.policyModel = self._buildPolicyModel(network, learning_rate)
        self.gamma = gamma
        self.buffer = ExperienceBuffer(state_shape, action_dim, gamma, rewards_to_go, normalize)
        self.rewards_to_go = rewards_to_go
        self.normalize = normalize

    def act(self, state):
        """Choose an action for the given state."""
        probs = self.policyModel.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(len(probs), p=probs)

    def train(self, env, episodes):
        """Train the agent using policy gradients."""
        rewards = np.zeros(episodes)

        # generate rollouts
        for i in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.buffer.store(state, action, reward)
                state = next_state
                rewards[i] += reward
            self.buffer.finishTrajectory()

        # produce the advantage vectors
        states, advantages = self.buffer.get()

        # fit to advantages to update policy
        self.policyModel.fit(states, advantages, verbose=0)

        return rewards

    def test(self, env, episodes, render=False):
        """Test the agent for given episodes on given environment."""
        rewards = np.zeros(episodes)
        for i in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                rewards[i] += reward
                if render:
                    env.render()
        return rewards

    def savePolicyModel(self, name):
        self.policyModel.save_weights(name)

    def loadPolicyModel(self, name):
        self.policyModel.load_weights(name)

    def _buildPolicyModel(self, network, learning_rate):
        model = tf.keras.models.clone_model(network)
        loss = 'categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model


class VPGAgent:
    """A vanilla policy gradient agent."""

    def __init__(self, policy_network, value_network,
                 policy_learning_rate=0.00025, value_learning_rate=0.0025,
                 gamma=0.99, lam=0.95):
        self.action_size = policy_network.output_shape[1]
        self.policyModel = self._buildPolicyModel(policy_network, policy_learning_rate)
        self.valueModel = self._buildValueModel(value_network, value_learning_rate)
        self.gamma = gamma
        self.lam = lam

    def act(self, state):
        """Choose an action for the given state."""
        probs = self.policyModel.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(len(probs), p=probs)

    def train(self, env, episodes):
        """Train the agent using policy gradients."""
        rewards_out = np.zeros(episodes)

        total_states = []
        total_actions = []
        total_rewards = []
        total_deltas = []

        # generate rollouts and discounted rewards
        for i in range(episodes):
            state = env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                states += [state]
                actions += [action]
                rewards += [reward]
                state = next_state
                rewards_out[i] += reward
                
            # compute GAE
            values = self.valueModel.predict(np.array(states))
            delta = np.expand_dims(np.array(rewards), axis=1) - values
            delta[:-1] = delta[:-1] + self.gamma * values[1:]
            delta = discounted_rewards(delta, self.gamma * self.lam)
            
            rewards = discounted_rewards(rewards, self.gamma)

            total_states += states
            total_actions += actions
            total_rewards += rewards
            total_deltas += delta

        # produce the advantage vectors
        advantages = np.zeros((len(total_rewards), self.action_size))
        for i in range(len(total_rewards)):
            advantages[i][total_actions[i]] = total_deltas[i]

        # fit to advantages to update policy
        self.policyModel.fit(np.array(total_states), advantages, verbose=0)

        # fit to rewards to update value
        self.valueModel.fit(np.array(total_states), np.array(total_rewards), verbose=0)

        return rewards_out

    def test(self, env, episodes, render=False):
        """Test the agent for given episodes on given environment."""
        rewards = np.zeros(episodes)
        for i in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                rewards[i] += reward
                if render:
                    env.render()
                    time.sleep(0.05)
        return rewards

    def savePolicyModel(self, name):
        self.policyModel.save_weights(name)

    def loadPolicyModel(self, name):
        self.policyModel.load_weights(name)

    def _buildPolicyModel(self, network, learning_rate):
        model = tf.keras.models.clone_model(network)
        loss = 'categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def _buildValueModel(self, network, learning_rate):
        model = tf.keras.models.clone_model(network)
        loss = 'mse'
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model
