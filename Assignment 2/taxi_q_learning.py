import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = gym.make("Taxi-v3")

state_size = env.observation_space.n
action_size = env.action_space.n

# Initialize Q-table
q_table = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 500

rewards = []

# Training loop
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Choose action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # Take step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update Q-table
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        state = next_state
        total_reward += reward
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

# Plot graph
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Taxi Q-Learning Training")
plt.show()


import time

env = gym.make("Taxi-v3", render_mode="human")

state, _ = env.reset()
done = False

print("🚕 Simulation Started\n")

while not done:
    action = np.argmax(q_table[state])
    
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    state = next_state
    
    time.sleep(0.5)

env.close()