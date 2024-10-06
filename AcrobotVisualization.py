import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Создава околината
env = gym.make('Acrobot-v1', render_mode="human")

# Иницијализира, тренира и еваулира DQN агент
dqn_agent = DQN("MlpPolicy", env, verbose=1)
dqn_agent.learn(total_timesteps=10000)
mean_reward_dqn, std_reward_dqn = evaluate_policy(dqn_agent, env, n_eval_episodes=10)

# Иницијализира, тренира и еваулира PPO агент
ppo_agent = PPO("MlpPolicy", env, verbose=1)
ppo_agent.learn(total_timesteps=10000)
mean_reward_ppo, std_reward_ppo = evaluate_policy(ppo_agent, env, n_eval_episodes=10)

# Резултати
print(f"DQN Mean Reward: {mean_reward_dqn}, Std: {std_reward_dqn}")
print(f"PPO Mean Reward: {mean_reward_ppo}, Std: {std_reward_ppo}")