import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class CellularNetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, nBS=5, nBC=10, nUE=5, bSize=100):
        self.BACTH_SIZE = bSize
        self.COUNTER = 0
        self.LTE_BS = 0
        self.max_snr = 60.0  # 60 mWatt
        self.MAX_REWARD = 100 # 100 Mbps
        self.nBS = nBS  # The total number BS within the environment
        self.nBC = nBC  # The total number of supported BC by all BSs
        self.nUE = nUE  # The total number UE within the LTE cell
        self.state = tuple()
        self.bc_pn_list = np.random.randint(0, 2, size=self.nBC, dtype=int)
        self.ue_bc_bs_reward = np.random.rand(self.nUE, self.nBC, self.nBS) * self.MAX_REWARD
        
        # Define the observation space for each UE as a tuple of MultiDiscrete and Box
        self.observation_space = spaces.Tuple((
            spaces.Tuple((
                spaces.MultiDiscrete([self.nBC, self.nBS]),
                spaces.Box(low=0, high=self.max_snr, shape=(self.nBS,))
            )) for _ in range(self.nUE)
        ))

        # Define the action space for each UE as a tuple of MultiDiscrete actions
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([self.nBC, self.nBS]) for _ in range(self.nUE)
        ))

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {"info": ""}

    def reset(self, seed=None, options=None):
        # Reset method to initialize the environment
        super().reset(seed=seed)
        self.COUNTER = 0
        state = []
        for _ in range(self.nUE):
            multi_discrete = np.random.randint(0, self.nBC), np.random.randint(0, self.nBS)
            box = np.random.uniform(0., self.max_snr, size=(self.nBS,))
            state.append(tuple([multi_discrete, box]))

        self.state = tuple(state)
        observation = self.state
        info = self._get_info()
        #print(observation)
        return observation, info
    
    def step(self, action):
        state = []
        reward = 0
        
        for i in range(self.nUE):
            #print(f"Action for UE {i}: {action[i]}")  # Debug: Print action for each UE
            bc_index, bs_index = action[i]
            #print(f"Using BC index: {bc_index}, BS index: {bs_index}")  # Debug: Check action indices
            
            # Check if indices are valid
            if not (0 <= bc_index < self.nBC):
                raise ValueError(f"BC index {bc_index} out of bounds (0 <= BC < {self.nBC})")
            if not (0 <= bs_index < self.nBS):
                raise ValueError(f"BS index {bs_index} out of bounds (0 <= BS < {self.nBS})")

            reward += self.ue_bc_bs_reward[i, bc_index, bs_index]
            
            # Create a new SNR state for this UE (randomized for simulation)
            box = np.random.uniform(0., self.max_snr, size=(self.nBS,))
            
            # Append the new state for this UE
            state.append((action[i], box))

        # Check if the episode is done (terminated)
        terminated = self.COUNTER == self.BACTH_SIZE
        self.state = tuple(state)
        observation = self.state
        info = self._get_info()
        self.COUNTER += 1

        return observation, reward, terminated, False, info


    def close(self):
        # Close the environment, if pygame is being used for rendering
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self, mode="human"):
        # Render the environment using pygame (optional visualization)
        if mode == "human":
            # Implement your rendering logic here using pygame
            pass