import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gym
from gym import spaces
import threading
import time

class Config:
    MAX_FORCE = 100.0  # 最大安全拉力(N)
    MAX_TORQUE = 5.0  # 电机最大扭矩(Nm)
    HISTORY_WINDOW = 5  # 历史数据窗口大小
    SINGLE_STATE_NUM = 6

class RopeLiftEnv(gym.Env):
    def __init__(self):
        super(RopeLiftEnv, self).__init__()
        self.state_dim = Config.SINGLE_STATE_NUM + 3 * Config.HISTORY_WINDOW  # 基础6维 + (力/压力/动作各HISTORY_WINDOW步)
        self.reset()

    def reset(self):
        self.current_force = np.random.uniform(5, 15)
        self.pressure = np.random.uniform(0.5, 1.5)
        self.position = 0.0
        self.velocity = 0.0
        self.force_integral = 0.0
        self.pressure_integral = 0.0
        self.force_history = deque([self.current_force] * Config.HISTORY_WINDOW, maxlen=Config.HISTORY_WINDOW)
        self.pressure_history = deque([self.pressure] * Config.HISTORY_WINDOW, maxlen=Config.HISTORY_WINDOW)
        self.action_history = deque([0.0] * Config.HISTORY_WINDOW, maxlen=Config.HISTORY_WINDOW)

    def step(self, action):
        self.pressure += np.random.normal(0, 0.02)
        self.pressure = np.clip(self.pressure, 0, 2)
        self.velocity = np.random.normal(-1, 1)
        self.position = np.random.normal(-1, 1)
        self.current_force = np.random.normal(1, 3)
        self.force_integral += (self.current_force - 10)
        self.pressure_integral += (self.pressure - 1.0)
        self.force_history.append(self.current_force)
        self.pressure_history.append(self.pressure)
        self.action_history.append(action[0])
        reward = -0.1 * abs(action[0] - self.action_history[-1])
        done = self.current_force > Config.MAX_FORCE
        return self._get_state(), reward, done, {}

    def _get_state(self):
        state = np.zeros(Config.SINGLE_STATE_NUM + 3*Config.HISTORY_WINDOW, dtype=np.float32)
        state[0] = self.current_force / Config.MAX_FORCE
        state[1] = self.pressure
        state[2] = self.position / 2.0
        state[3] = self.velocity / 1.0
        state[4] = np.clip(self.force_integral / 10.0, -1, 1)
        state[5] = np.clip(self.pressure_integral / 5.0, -1, 1)
        for i, f in enumerate(self.force_history):
            state[Config.SINGLE_STATE_NUM+i] = (f - self.current_force)/Config.MAX_FORCE
        for i, p in enumerate(self.pressure_history):
            state[Config.SINGLE_STATE_NUM+Config.HISTORY_WINDOW+i] = (p - self.pressure)/0.2
        for i, a in enumerate(self.action_history):
            state[Config.SINGLE_STATE_NUM+2*Config.HISTORY_WINDOW+i] = a
        return state

class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.feature_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.actor_mean = nn.Linear(64, 1)
        self.actor_logstd = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.feature_net(x)
        mean = torch.tanh(self.actor_mean(features))
        std = torch.exp(self.actor_logstd).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.critic(features)
        return dist, value

def train_ppo():
    env = RopeLiftEnv()
    policy = ActorCritic(env.state_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    gamma = 0.99
    clip_param = 0.2
    entropy_coef = 0.01
    epochs = 4
    for episode in range(2):
        print('episode: ', episode)
        env.reset()
        state = env._get_state()
        states, actions, rewards, log_probs = [], [], [], []
        step_count = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                dist, value = policy(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            if env.current_force > Config.MAX_FORCE * 0.8 and action > 0:
                action = torch.clamp(action, -1, 0)
            next_state, reward, done, _ = env.step(action.numpy())
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            step_count += 1
            if done or step_count >= 200:  # 防止无限循环
                break
        states_tensor = torch.FloatTensor(np.array(states, dtype=np.float32))
        actions_tensor = torch.cat(actions).view(-1)  # 确保是1D张量
        old_log_probs = torch.cat(log_probs).detach()
        returns = np.zeros(len(rewards), dtype=np.float32)
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            returns[i] = R
        returns = torch.from_numpy(returns)  # 直接从numpy数组创建

        for _ in range(epochs):
            dist, values = policy(states_tensor)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            advantages = returns - values.squeeze().detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip_param, 1+clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(-1), returns)  # 确保形状匹配
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
    torch.save(policy, "ppo_rope_lift.pth")

class RealTimeController:
    def __init__(self, policy_path):
        self.policy = torch.load(policy_path)
        self.policy.eval()
        self.sensor_buffer = deque([{'force': 0, 'pressure': 0} for _ in range(Config.HISTORY_WINDOW)], maxlen=Config.HISTORY_WINDOW)
        self.action_history = deque([0.0]*Config.HISTORY_WINDOW, maxlen=Config.HISTORY_WINDOW)
        self.cur_torque = 0.0

    def update_sensors(self, force, pressure):
        self.sensor_buffer.append({'force': force, 'pressure': pressure})
        current = self.sensor_buffer[-1]
        state = np.zeros(Config.SINGLE_STATE_NUM + 3*Config.HISTORY_WINDOW, dtype=np.float32)
        state[0] = current['force'] / Config.MAX_FORCE
        state[1] = current['pressure']
        state[2] = 0.0
        state[3] = 0.0
        state[4] = 0.0
        state[5] = 0.0
        hist_forces = [f['force'] for f in self.sensor_buffer]
        hist_pressures = [p['pressure'] for p in self.sensor_buffer]
        for i in range(Config.HISTORY_WINDOW):
            state[Config.SINGLE_STATE_NUM+i] = (hist_forces[i] - current['force'])/Config.MAX_FORCE
        for i in range(Config.HISTORY_WINDOW):
            state[Config.SINGLE_STATE_NUM+Config.HISTORY_WINDOW+i] = (hist_pressures[i] - current['pressure'])/0.2
        for i in range(Config.HISTORY_WINDOW):
            state[Config.SINGLE_STATE_NUM+2*Config.HISTORY_WINDOW+i] = 0.0
        self.state = state

    def get_action(self):
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.policy(state_tensor)
            action = dist.mean.item()
        torque = action * Config.MAX_TORQUE
        self.cur_torque = 0.8 * torque + 0.2 * self.cur_torque
        self.action_history.append(action)
        for i in range(Config.HISTORY_WINDOW):
            self.state[Config.SINGLE_STATE_NUM+2*Config.HISTORY_WINDOW+i] = self.action_history[i]
        return self.cur_torque


# ==================== 主程序 ====================
if __name__ == "__main__":
    train_ppo()
    controller = RealTimeController("ppo_rope_lift.pth")
    sim_force = np.random.normal(15, 2)
    sim_pressure = np.random.normal(1.0, 0.1)
    controller.update_sensors(force=np.clip(sim_force, 5, Config.MAX_FORCE), pressure=np.clip(sim_pressure, 0.5, 1.5))
    torque = controller.get_action()

