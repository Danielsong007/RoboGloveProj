import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gym
import threading
import queue
import time

class Config:
    HISTORY_WINDOW = 5  # 历史数据窗口大小
    SINGLE_STATE_NUM = 6

class RopeLiftEnv(gym.Env):
    def __init__(self):
        super(RopeLiftEnv, self).__init__()
        self.state_dim = Config.SINGLE_STATE_NUM + 3 * Config.HISTORY_WINDOW  # 基础6维 + (力/压力/动作各HISTORY_WINDOW步)
        self.current_force = 0.0
        self.pressure = 0.0
        self.position = 0.0
        self.velocity = 0.0
        self.force_integral = 0.0
        self.pressure_integral = 0.0
        self.force_history = deque([self.current_force] * Config.HISTORY_WINDOW, maxlen=Config.HISTORY_WINDOW)
        self.pressure_history = deque([self.pressure] * Config.HISTORY_WINDOW, maxlen=Config.HISTORY_WINDOW)
        self.action_history = deque([0.0] * Config.HISTORY_WINDOW, maxlen=Config.HISTORY_WINDOW)

    def step(self, action):
        self.current_force = np.random.normal(1, 3)
        self.pressure = np.random.normal(0, 0.02)
        self.position = np.random.normal(-1, 1)
        self.velocity = np.random.normal(-1, 1)
        self.force_integral = 0
        self.pressure_integral = 0
        self.force_history.append(self.current_force)
        self.pressure_history.append(self.pressure)
        self.action_history.append(action[0])
        reward = -0.1 * abs(action[0] - self.action_history[-1])
        done = 0
        return self._get_state(), reward, done, {}

    def _get_state(self):
        state = np.zeros(Config.SINGLE_STATE_NUM + 3*Config.HISTORY_WINDOW, dtype=np.float32)
        state[0] = self.current_force
        state[1] = self.pressure
        state[2] = self.position
        state[3] = self.velocity
        state[4] = self.force_integral
        state[5] = self.pressure_integral
        for i, f in enumerate(self.force_history):
            state[Config.SINGLE_STATE_NUM+i] = f
        for i, p in enumerate(self.pressure_history):
            state[Config.SINGLE_STATE_NUM+Config.HISTORY_WINDOW+i] = p
        for i, a in enumerate(self.action_history):
            state[Config.SINGLE_STATE_NUM+2*Config.HISTORY_WINDOW+i] = a
        return state

class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()
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

class PPOTrainer:
    def __init__(self, policy, gamma=0.99, clip_param=0.2, entropy_coef=0.01, lr=3e-4):
        self.policy = policy
        self.latest_policy = policy  # 最新策略
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.epochs = 4
        self.data_queue = queue.Queue(maxsize=10)  # 限制队列大小防止内存溢出
        self.stop_event = threading.Event()
        self.train_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.train_thread.start()
    
    def _training_loop(self):
        while not self.stop_event.is_set():
            try:
                states, actions, rewards, old_log_probs = self.data_queue.get(timeout=1.0)
                returns = np.zeros(len(rewards), dtype=np.float32)
                R = 0
                for i in reversed(range(len(rewards))):
                    R = rewards[i] + self.gamma * R
                    returns[i] = R
                returns = torch.from_numpy(returns)
                states_tensor = torch.FloatTensor(np.array(states, dtype=np.float32))
                actions_tensor = torch.cat(actions).view(-1)
                old_log_probs_tensor = torch.cat(old_log_probs).detach()
                for _ in range(self.epochs):
                    dist, values = self.policy(states_tensor)
                    new_log_probs = dist.log_prob(actions_tensor)
                    entropy = dist.entropy().mean()
                    advantages = returns - values.squeeze().detach()
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(values.squeeze(-1), returns)
                    loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()
                self.latest_policy.load_state_dict(self.policy.state_dict())
            except queue.Empty:
                continue


if __name__ == "__main__":
    env = RopeLiftEnv()
    policy = ActorCritic(env.state_dim)
    trainer = PPOTrainer(policy)
    try:
        step_count = 0
        state = env._get_state()
        states, actions, rewards, log_probs = [], [], [], []
        current_policy = trainer.latest_policy
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                dist, value = current_policy(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            if env.current_force > 8 and action > 0:
                action = torch.clamp(action, -1, 0)
            next_state, reward, done, _ = env.step(action.numpy())
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            step_count += 1
            if step_count % 100 == 0:
                print(step_count/100)
                trainer.data_queue.put((states, actions, rewards, log_probs), block=False) # 非阻塞方式添加数据，如果队列满则跳过
                states, actions, rewards, log_probs = [], [], [], []
                current_policy = trainer.latest_policy
    
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        trainer.stop_event.set()
        trainer.train_thread.join()
        torch.save(trainer.latest_policy.state_dict(), "/home/song/myws/RoboGloveProj/RL M/ppo_rope_lift_final.pth")


