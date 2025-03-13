import gymnasium as gym
import gymnasium_env
from gymnasium import spaces
import pygame
import numpy as np
import random
import os
from collections import deque
import tensorflow as tf
from keras import layers, optimizers, losses

N_BS=5 
N_BC=10 
N_UE=5
BACTH_SIZE = 100

env = gym.make("gymnasium_env/CellularNetEnv-v0", nBS=N_BS, nBC=N_BC, nUE=N_UE, bSize=BACTH_SIZE)

class QNetwork(tf.keras.Model):
    def __init__(self, input_shape, output_shape):
        super(QNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(output_shape, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)


class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=BACTH_SIZE):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Q-Network
        self.model = QNetwork(input_shape=(self._get_state_shape(),), output_shape=self._get_action_shape())
        self.model(np.zeros((1, self._get_state_shape())))
        self.load_model()

        # Target network (for stable learning)
        self.target_model = QNetwork(input_shape=(self._get_state_shape(),), output_shape=self._get_action_shape())
        self.target_model(np.zeros((1, self._get_state_shape())))  
        self.target_model.set_weights(self.model.get_weights())

        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _get_state_shape(self):
        return (2 + N_BS) * N_UE
        #return self.env.observation_space[0][1].shape[0] * len(self.env.observation_space)

    def _get_action_shape(self):
        # Access the nvec attribute of the MultiDiscrete action space
        nBC, nBS = self.env.action_space[0].nvec  # nvec stores the size of each dimension
        return nBC * nBS * len(self.env.action_space)  # Total number of actions per UE
        #return self.env.action_space[0].shape * len(self.env.action_space)

    def _get_2d_idex(self, idx, rows, cols):
        i = idx // cols  # Row index
        j = idx % cols   # Column index
        return (i, j)

    def _preprocess_state(self, state):
        # Flatten each individual UE's state (MultiDiscrete and Box)
        flattened_state = []
        for ue_state in state:
            # Flatten the MultiDiscrete and Box separately and concatenate
            multi_discrete, box = ue_state
            flattened_state.extend(multi_discrete)  # Flatten MultiDiscrete (2 elements)
            flattened_state.extend(box)  # Flatten Box (shape=(nBS,))
        
        return np.array(flattened_state).reshape(1, -1)

    def remember(self, state, action, reward, next_state, done):
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print(f"========================== We are returning value at random ")
            random_action = self.env.action_space.sample()
            return random_action
        
        state = self._preprocess_state(state)
        q_values = self.model(state)
        print(f"++++++++++++++++++++++++++ We are returning value at normal ")
        greedy_action = []
        for i in range(N_UE):
            greedy_idx = np.argmax(q_values[:,i*(N_BS*N_BC):(i+1)*(N_BS*N_BC)])
            #print(f"Greedy index is : {greedy_idx}")
            index_x, index_y = self._get_2d_idex(greedy_idx, N_BC, N_BS)
            greedy_action.append(np.array([index_x, index_y]))

        return tuple(greedy_action)

    def _convert_batch_to_q_values(self, batch_q_values):
        future_q_values = []
        for itr in range(self.batch_size):
            future_q_values.append(np.array([np.max(batch_q_values[itr][0][i*(N_BS*N_BC):(i+1)*(N_BS*N_BC)]) for i in range(N_UE)]))
        return future_q_values    

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        #print(minibatch)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])
        # Predict Q-values for the current states and next states
        target_q_values = self.target_model(next_states)
        #future_q_values = np.max(target_q_values, axis=1)
        future_q_values = np.array([np.max(a, axis=1) for a in target_q_values])

        # Compute target Q-values using the Bellman equation
        target = rewards + (self.gamma * future_q_values.squeeze()  * (1 - dones)) 
   
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            squeeze_q_values = tf.squeeze(q_values,axis=1)
            action_indices = np.array([self._get_action_index(a) for a in actions])
            action_indices = tf.convert_to_tensor(action_indices, dtype=tf.int32)
            q_values = tf.gather(squeeze_q_values, action_indices, axis=1, batch_dims=1)
                        
            # Loss calculation (MSE)
            #loss = tf.reduce_mean(tf.square(target - q_values))
            q_values =  tf.reduce_sum(q_values, axis=1)
            loss = tf.reduce_mean(tf.square(target - q_values))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Check if gradients are None
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def _get_action_index(self, action):
        # Convert the action tuple (BC, BS) to a single index
        #return action[0] * self.env.nBS + action[1]
        #print(f"My actions: {action}")
        idx = []
        for i in range(N_UE):
            idx.append((i*(N_BC*N_BS)) + (action[i][0]*N_BS+action[i][1]))
   
        return idx

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self, state) :
        state = self._preprocess_state(state)
        return self.model(state)
    
    def save_model(self):
        # saving and loading the .h5 model
        fileName = "N_UE="+str(N_UE)+"-N_BC="+str(N_BC)+"-N_BS="+str(N_BS)+".weights.h5"
        self.model.save_weights(fileName)
        # save model
        print('Model Saved!')
 
    def load_model(self):
        # load model
        fileName = "N_UE="+str(N_UE)+"-N_BC="+str(N_BC)+"-N_BS="+str(N_BS)+".weights.h5"
        # load model if exists 
        if os.path.exists(fileName):
            self.model.load_weights(fileName)
            print('Model Loaded!')

def train_dqn_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        itr = 0
        while not done:
            print(f"============================ Iteration : {itr}") 
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            """
            print("Current State : ")
            print(state)
            print("Q Values : ")
            print(agent.get_q_values(state))
            print("Selected Action : ")
            print(action)
            print("Next State : ")
            print(next_state)
            print("Received Reward : ")
            print(reward)"
            """
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            agent.decrease_epsilon()
            state = next_state
            total_reward += reward
            itr += 1

        agent.update_target_model()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
    
    agent.save_model()

# Create the environment and agent
agent = DQNAgent(env)

# Train the agent
train_dqn_agent(env, agent, episodes=10)