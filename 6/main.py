import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

# observation space is an ndarray of shape (2,) where the elements
# are the position of the car along the x-axis [-1.2, 0.6] and its
# velocity [-0.07, 0.07]

# action space is an integer in the range [0, 2] where
# 0 = push left, 1 = no push, 2 = push right

# the reward is -1 for each time step and 0 when the goal is reached

# goal position is x_car >= 0.5

SEED = 17

class MountainCarAgent:
    def __init__(
            self,
            n_positions=20,
            n_velocities=20,
            n_actions=3,
            alpha=0.1,
            gamma=0.9,
            n_episodes=5000,
            max_steps=200,
            epsilon=1,
            epsilon_decay=0.001,
            epsilon_min=0.01,
            # epsilon-greedy parameters
            # As we can see in the while loop below, epsilon will decrease exponentially
            # This parameter is used to balance exploration and exploitation
            # epsilon is the threshold for the probability of choosing a random action instead of the best action
            # if epsilon is 1, then the agent will always choose a random action
            # if epsilon is 0, then the agent will always choose the best action
            # epsilon will decrease exponentially from 1 to 0.01
            # So, the agent will explore more in the beginning and exploit more in the end.
            # Which is one of the most common strategies in reinforcement learning
        ):
        # randomness
        np.random.seed(SEED)
        random.seed(SEED)

        self.N_POSITIONS = n_positions
        self.N_VELOCITIES = n_velocities
        self.N_STATES = np.array([self.N_POSITIONS, self.N_VELOCITIES])
        self.N_ACTIONS = n_actions
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.N_EPISODES = n_episodes
        self.MAX_STEPS = max_steps
        self.EPSILON = epsilon
        self.EPSILON_DECAY = epsilon_decay
        self.EPSILON_MIN = epsilon_min

        self.env = gym.make('MountainCar-v0', render_mode=None)
        self.env.reset()

        # The Q-table will have shape (N_POSITIONS, N_VELOCITIES, N_ACTIONS) and will
        # be initialized with random values between -2 and 0. At each cell of the Q-table
        # we will store the Q-value for the corresponding state-action pair, but we represent
        # the state as a tuple of the discretized position and velocity and action as an integer.
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.N_POSITIONS, self.N_VELOCITIES, self.N_ACTIONS))
        self.total_rewards = []
        self.episode_lengths = []
        self.convergencies = []

    def discretize_state(self, state):
        state_min = self.env.observation_space.low
        state_max = self.env.observation_space.high
        state_bins = (state_max - state_min) / self.N_STATES
        return tuple(((state - state_min) / state_bins).astype(int))

    def train(self):
        for episode in range(self.N_EPISODES):
            state, _ = self.env.reset()
            discrete_state = self.discretize_state(state)
            done = False
            total_reward = 0
            episode_length = 0

            while not done and episode_length < self.MAX_STEPS:
                episode_length += 1
                if np.random.rand() <= self.EPSILON: # explore
                    action = np.random.randint(0, self.N_ACTIONS)
                else: # exploit
                    action = np.argmax(self.q_table[discrete_state])

                next_state, reward, done, _, _ = self.env.step(action)
                next_discrete_state = self.discretize_state(next_state)

                if not done: # update the Q-value
                    max_future_q = np.max(self.q_table[next_discrete_state])
                    current_q = self.q_table[discrete_state][action]
                    new_q = (1 - self.ALPHA) * current_q + self.ALPHA * (reward + self.GAMMA * max_future_q)
                    self.q_table[discrete_state][action] = new_q
                elif next_state[0] >= self.env.goal_position: # goal reached
                    print(f"Goal reached in episode {episode}")
                    self.q_table[discrete_state][action] = 0

                discrete_state = next_discrete_state # update the state
                total_reward += reward # update the total reward

            self.EPSILON = max(self.EPSILON_MIN, np.exp(-self.EPSILON_DECAY * episode)) # decrease epsilon
            self.total_rewards.append(total_reward) # store the total reward
            self.episode_lengths.append(episode_length) # store the episode length
            self.convergencies.append(done) # store if the episode converged (goal reached or max steps reached)

        self.env.close()

    def plot_rewards(self, ax):
        ax.plot(self.total_rewards)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(f'(p,v,a,g,eps,s)=({self.N_POSITIONS},{self.N_VELOCITIES},{self.ALPHA},{self.GAMMA},{self.N_EPISODES},{self.MAX_STEPS})')

    def plot_episode_lengths(self, ax):
        ax.plot(self.episode_lengths)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title(f'(p,v,a,g,eps,s)=({self.N_POSITIONS},{self.N_VELOCITIES},{self.ALPHA},{self.GAMMA},{self.N_EPISODES},{self.MAX_STEPS})')

    def plot_convergencies(self, ax):
        ax.plot(self.convergencies)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Convergence')
        ax.set_title(f'(p,v,a,g,eps,s)=({self.N_POSITIONS},{self.N_VELOCITIES},{self.ALPHA},{self.GAMMA},{self.N_EPISODES},{self.MAX_STEPS})')

    def demo(self):
        render_env = gym.make('MountainCar-v0', render_mode='human')
        state, _ = render_env.reset()
        done = False

        while not done:
            render_env.render()
            discrete_state = self.discretize_state(state)
            action = np.argmax(self.q_table[discrete_state])
            state, _, done, _, _ = render_env.step(action)

        render_env.close()

# Paramterer greed
parameter_grid = {
    'alpha': [0.3, 0.5],
    'gamma' : [0.7],
    'n_episodes': [3000],
    'max_steps': [100, 500],
}

# Generate configurations that vary one parameter at a time for
# N_POSITIONS, N_VELOCITIES = 20, 20 or 40, 40
configuration_20 = {
    'n_positions': 20,
    'n_velocities': 20,
    'n_actions': 3,
    'alpha': 0.1,
    'gamma': 0.9,
    'n_episodes': 5000,
    'max_steps': 200,
    }

configuration_40 = {
    'n_positions': 40,
    'n_velocities': 40,
    'n_actions': 3,
    'alpha': 0.1,
    'gamma': 0.9,
    'n_episodes': 5000,
    'max_steps': 200,
}

configurations_20 = [configuration_20]
for param, values in parameter_grid.items():
    for value in values:
        config = configuration_20.copy()
        config[param] = value
        configurations_20.append(config)

configurations_40 = [configuration_40]
for param, values in parameter_grid.items():
    for value in values:
        config = configuration_40.copy()
        config[param] = value
        configurations_40.append(config)

agents = {}

for config in configurations_20:
    agent = MountainCarAgent(**config)
    agent.train()
    key = tuple(config.values())
    agents[key] = agent

for config in configurations_40:
    agent = MountainCarAgent(**config)
    agent.train()
    key = tuple(config.values())
    agents[key] = agent

# Plot the total rewards for each configuration
max_subplots_per_fig = 6
num_figs = len(agents) // max_subplots_per_fig + (len(agents) % max_subplots_per_fig > 0)

for fig_idx in range(num_figs):
    start_idx = fig_idx * max_subplots_per_fig
    end_idx = min(start_idx + max_subplots_per_fig, len(agents))
    
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    axs = axs.flatten()

    for plot_idx, result_idx in enumerate(range(start_idx, end_idx)):
        key = list(agents.keys())[result_idx]
        agent = agents[key]
        agent.plot_rewards(axs[plot_idx])

    plt.tight_layout()
plt.show()

for fig_idx in range(num_figs):
    start_idx = fig_idx * max_subplots_per_fig
    end_idx = min(start_idx + max_subplots_per_fig, len(agents))
    
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    axs = axs.flatten()

    for plot_idx, result_idx in enumerate(range(start_idx, end_idx)):
        key = list(agents.keys())[result_idx]
        agent = agents[key]
        agent.plot_episode_lengths(axs[plot_idx])

    plt.tight_layout()
plt.show()

for fig_idx in range(num_figs):
    start_idx = fig_idx * max_subplots_per_fig
    end_idx = min(start_idx + max_subplots_per_fig, len(agents))
    
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    axs = axs.flatten()

    for plot_idx, result_idx in enumerate(range(start_idx, end_idx)):
        key = list(agents.keys())[result_idx]
        agent = agents[key]
        agent.plot_convergencies(axs[plot_idx])

    plt.tight_layout()
plt.show()

# a demo at the end
demo_agent = agents[tuple(configuration_20.values())]
demo_agent.demo()