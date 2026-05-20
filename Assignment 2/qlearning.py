import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# 1. CREATE ENVIRONMENT
# -----------------------------
env = gym.make("Taxi-v3")

state_size = env.observation_space.n
action_size = env.action_space.n

# -----------------------------
# 2. Q-TABLE
# -----------------------------
q_table = np.zeros((state_size, action_size))

# -----------------------------
# 3. HYPERPARAMETERS (IMPROVED)
# -----------------------------
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.999   # slower decay
epsilon_min = 0.01
episodes = 2000         # increased training

rewards = []

# -----------------------------
# 4. TRAINING
# -----------------------------
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Q-learning update
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        state = next_state
        total_reward += reward
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

# -----------------------------
# 5. TRAINING GRAPH
# -----------------------------
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Taxi Q-Learning Training")
plt.show()

# -----------------------------
# 6. SIMULATION (FIXED VERSION)
# -----------------------------
env = gym.make("Taxi-v3", render_mode="human")

print("🚕 Simulation Started\n")

success = False

for attempt in range(5):   # multiple attempts
    print(f"\n🔁 Attempt {attempt+1}")
    
    state, _ = env.reset()
    done = False
    step_count = 0
    
    while not done and step_count < 100:
        
        # 🔥 small exploration to avoid loops
        if np.random.rand() < 0.1:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Debug info
        taxi_row, taxi_col, pass_loc, dest = env.unwrapped.decode(state)
        print(f"Step:{step_count} Taxi:({taxi_row},{taxi_col}) Passenger:{pass_loc} Destination:{dest} Reward:{reward}")
        
        if pass_loc == 4:
            print("🚕 Passenger Picked Up")
        
        if reward == 20:
            print("✅ SUCCESS: Passenger dropped correctly!")
            success = True
            break
        
        state = next_state
        step_count += 1
        time.sleep(0.3)
    
    if success:
        break

# -----------------------------
# 7. FINAL RESULT
# -----------------------------
if success:
    print("\n🎯 Task Completed Successfully")
else:
    print("\n❌ Task Failed after multiple attempts")

env.close()