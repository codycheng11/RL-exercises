import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

mc_env = gym.make('MountainCar-v0')

def Qlearning(env, alpha, discount, epsilon, min_eps, episodes):

    # Initialize q-table
    buckets = (20, 60)
    actions = env.action_space.n
    q_table = np.zeros(buckets + (actions,))

    # Re-define observations into the buckets we defined eralier
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    def discretize_state(obs):
        ratio = (obs - state_low)/(state_high - state_low)
        new_state = np.floor(ratio * np.array(buckets))
        new_state = np.clip(new_state, 0, np.array(buckets) - 1)
        return tuple(new_state.astype(int))
    Qlearning.discretize_state = discretize_state

    # Initialize reward history for plotting, and epsilon decay rate
    reward_history = []
    eps_decay_rate = (epsilon - min_eps) / episodes
    fixed_ep = epsilon

    # Start an episode
    for i in range(episodes):
        observation, info = env.reset()
        state = discretize_state(observation)
        terminated = truncated = False
        total_reward = 0
        alpha_val = min(alpha, -3 * alpha * i / episodes + 3 * alpha)
        epsilon_val = min(epsilon, -3 * fixed_ep * i / episodes + 3 * fixed_ep)
        old_q = q_table.copy()

        while not terminated and not truncated:

            # Choose action by being epsilon-greedy
            if np.random.rand() < epsilon_val:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action, observe R and S'
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(observation)     

            # Update using Q-learning formula
            if terminated:
                q_table[state][action] += alpha_val * (reward - q_table[state][action])
            else:
                next_best = np.max(q_table[next_state])
                q_table[state][action] += alpha_val * (reward + discount * next_best - q_table[state][action])

            # Update state
            state = next_state
            total_reward -= 1

        # Append reward history, update epsilon
        reward_history.append(total_reward)
        epsilon -= eps_decay_rate
        print(np.max(q_table - old_q))
        
    # Return value function and reward history for plotting
    return q_table, reward_history

# Train the model
value_func, rewards = Qlearning(env = mc_env, alpha = 0.1, discount = 0.99, epsilon = 0.5, min_eps = 0.4, episodes = 5000)

# Test it out
mc_env = gym.make('MountainCar-v0', render_mode = "human")
observation, info = mc_env.reset()
state = Qlearning.discretize_state(observation)
for i in range(200):
    action = np.argmax(value_func[state])
    observation, reward, terminated, truncated, info = mc_env.step(action)
    state = Qlearning.discretize_state(observation)

# Plot rewards vs. episode number, as well as a sliding average version
rewards = np.array(rewards)
rewards_avg = np.convolve(rewards, np.ones(50) / 50, mode = "valid")

plt.plot(rewards, color = "b", label = "Reward")
plt.plot(rewards_avg, color = "r", label = "Reward (averaged over last 50)")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Mountain Car with Q-learning")
plt.legend()
plt.show()

mc_env.close()