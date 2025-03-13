#import tensorflow as tf
from collections import defaultdict
import gymnasium
import gymnasium_env
#print("TensorFlow version:", tf.__version__)
env = gymnasium.make('gymnasium_env/CellularNetEnv-v0')
obs, info = env.reset()
print(obs)

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    #print("Reward : ", reward)
    episode_over = terminated or truncated

env.close()