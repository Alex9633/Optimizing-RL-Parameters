import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import itertools

# Создава околината
env = gym.make('Acrobot-v1')

# Иницијализација на параметарските мрежи
dqn_params_grid = {
    'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    'gamma': [0.95, 0.96, 0.97, 0.98, 0.99],
    'batch_size': [32, 64, 128, 256]
}

ppo_params_grid = {
    'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    'gamma': [0.95, 0.96, 0.97, 0.98, 0.99],
    'batch_size': [32, 64, 128, 256]
}


# Функција за тренирање и евалуација на агентот
def evaluate_agent(agent, env, total_timesteps=10000, n_eval_episodes=10):
    agent.learn(total_timesteps=total_timesteps)
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=n_eval_episodes)
    return mean_reward, std_reward


# Функција за барање на оптимални параметри за DQN
def grid_search_dqn(env, param_grid):
    best_dqn_params = None
    best_mean_reward = float('-inf')

    # Итерација на сите параметарски комбинации
    for params in itertools.product(*param_grid.values()):
        dqn_params = dict(zip(param_grid.keys(), params))

        # Иницијализација на агентот со соодветни параметри
        dqn_agent = DQN("MlpPolicy", env, learning_rate=dqn_params['learning_rate'],
                        gamma=dqn_params['gamma'],
                        batch_size=dqn_params['batch_size'],
                        verbose=0)

        # Тестирање и евалуација на агентот
        print(f"Testing DQN with parameters: {dqn_params}")
        mean_reward, std_reward = evaluate_agent(dqn_agent, env)
        print(f"DQN Mean Reward: {mean_reward}, Standard Deviation: {std_reward}")

        # Ажурирање ги досега најоптималните параметри ако новите се подобри
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_dqn_params = dqn_params

    return best_dqn_params, best_mean_reward


# Функција за барање на оптимални параметри за PPO
def grid_search_ppo(env, param_grid):
    best_ppo_params = None
    best_mean_reward = float('-inf')

    # Итерација на сите параметарски комбинации
    for params in itertools.product(*param_grid.values()):
        ppo_params = dict(zip(param_grid.keys(), params))

        # Иницијализација на агентот со соодветни параметри
        ppo_agent = PPO("MlpPolicy", env, learning_rate=ppo_params['learning_rate'],
                        gamma=ppo_params['gamma'],
                        batch_size=ppo_params['batch_size'],
                        verbose=0)

        # Тестирање и евалуација на агентот
        print(f"Testing PPO with parameters: {ppo_params}")
        mean_reward, std_reward = evaluate_agent(ppo_agent, env)
        print(f"PPO Mean Reward: {mean_reward}, Standard Deviation: {std_reward}")

        # Ажурирање ги досега најоптималните параметри ако новите се подобри
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_ppo_params = ppo_params

    return best_ppo_params, best_mean_reward


# Извршува барањата на оптималните параметри
best_dqn_params, best_dqn_reward = grid_search_dqn(env, dqn_params_grid)
best_ppo_params, best_ppo_reward = grid_search_ppo(env, ppo_params_grid)

# Резултати
print(f"Best DQN Parameters: {best_dqn_params}, Mean Reward: {best_dqn_reward}")
print(f"Best PPO Parameters: {best_ppo_params}, Mean Reward: {best_ppo_reward}")

'''
Од претходни тестирања:

Best DQN Parameters: {'learning_rate': 0.01, 'gamma': 0.98, 'batch_size': 64}, Mean Reward: -109.2
Best PPO Parameters: {'learning_rate': 0.01, 'gamma': 0.97, 'batch_size': 64}, Mean Reward: -85.1

Best DQN Parameters: {'learning_rate': 0.001, 'gamma': 0.96, 'batch_size': 32}, Mean Reward: -127.7
Best PPO Parameters: {'learning_rate': 0.05, 'gamma': 0.99, 'batch_size': 32}, Mean Reward: -84.0

Best DQN Parameters: {'learning_rate': 0.05, 'gamma': 0.98, 'batch_size': 32}, Mean Reward: -104.5
Best PPO Parameters: {'learning_rate': 0.001, 'gamma': 0.99, 'batch_size': 32}, Mean Reward: -76.7

'''