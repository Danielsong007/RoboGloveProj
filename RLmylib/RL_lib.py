import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gym
import threading
import queue
import time


class RopeLiftEnv(gym.Env):
    def __init__(self):
        super(RopeLiftEnv, self).__init__()
        HISTORY_WINDOW = 5  # 历史数据窗口大小
        self.cur_force = 0.0
        self.pressure = 0.0
        self.position = 0.0
        self.his_force = deque([self.cur_force] * HISTORY_WINDOW, maxlen=HISTORY_WINDOW)
        self.his_press = deque([self.pressure] * HISTORY_WINDOW, maxlen=HISTORY_WINDOW)
        self.his_pos = deque([self.position] * HISTORY_WINDOW, maxlen=HISTORY_WINDOW)
        self.his_action = deque([0.0] * HISTORY_WINDOW, maxlen=HISTORY_WINDOW)
        self.state = np.concatenate([np.array(self.his_force, dtype=np.float32), np.array(self.his_press, dtype=np.float32), np.array(self.his_pos, dtype=np.float32), np.array(self.his_action, dtype=np.float32)])
        self.state_dim = len(self.state)

    def step(self, action):
        self.his_force.append(self.cur_force)
        self.his_press.append(self.pressure)
        self.his_pos.append(self.position)
        self.his_action.append(action[0][0])
        self.state = np.concatenate([np.array(self.his_force, dtype=np.float32), np.array(self.his_press, dtype=np.float32), np.array(self.his_pos, dtype=np.float32), np.array(self.his_action, dtype=np.float32)])
        reward = -0.1 * abs(action[0] - self.his_action[-1])
        done = 0
        return self.state, reward, done

class RealEnv(gym.Env):
    def __init__(self, buffer_weight_Srope,buffer_weight_Stouch,buffer_weight_CurPos,buffer_weight_CurVel,buffer_weight_CurAcc):
        super(RealEnv, self).__init__()
        self.upd_state(buffer_weight_Srope,buffer_weight_Stouch,buffer_weight_CurPos,buffer_weight_CurVel,buffer_weight_CurAcc)
    
    def upd_state(self, buffer_weight_Srope,buffer_weight_Stouch,buffer_weight_CurPos,buffer_weight_CurVel,buffer_weight_CurAcc):
        buffer_weight_Srope = deque([x/1000 for x in buffer_weight_Srope], maxlen=len(buffer_weight_Srope))
        self.his_force = buffer_weight_Srope
        buffer_weight_Stouch = deque([x/500 for x in buffer_weight_Stouch], maxlen=len(buffer_weight_Stouch))
        self.his_press = buffer_weight_Stouch
        buffer_weight_CurPos = deque([x/100000000 for x in buffer_weight_CurPos], maxlen=len(buffer_weight_CurPos))
        self.his_pos = buffer_weight_CurPos
        buffer_weight_CurVel = deque([x/10000 for x in buffer_weight_CurVel], maxlen=len(buffer_weight_CurVel))
        self.his_vel = buffer_weight_CurVel
        buffer_weight_CurAcc = deque([x/100 for x in buffer_weight_CurAcc], maxlen=len(buffer_weight_CurAcc))
        self.his_acc = buffer_weight_CurAcc
        self.state = np.concatenate([np.array(self.his_force, dtype=np.float32), np.array(self.his_press, dtype=np.float32), np.array(self.his_pos, dtype=np.float32), np.array(self.his_vel, dtype=np.float32), np.array(self.his_acc, dtype=np.float32)])
        self.state_dim = len(self.state)
        # print(self.his_force, self.his_press, self.his_pos, self.his_vel, self.his_acc)

    def step(self, buffer_weight_Srope,buffer_weight_Stouch,buffer_weight_CurPos,buffer_weight_CurVel,buffer_weight_CurAcc):
        self.upd_state(buffer_weight_Srope,buffer_weight_Stouch,buffer_weight_CurPos,buffer_weight_CurVel,buffer_weight_CurAcc)
        reward = -0.1*(np.mean(self.his_force) + np.mean(self.his_press))
        done = 0
        return self.state, reward, done

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
        self.epochs = 10
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
                    print('total loss: ', loss.item(), 'policy_loss: ', policy_loss.item(), 'value_loss: ', value_loss.item(), 'entropy: ', entropy.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()
                self.latest_policy.load_state_dict(self.policy.state_dict())
            except queue.Empty:
                continue


# if __name__ == "__main__":
#     env = RopeLiftEnv()
#     policy = ActorCritic(env.state_dim)
#     trainer = PPOTrainer(policy)
#     try:
#         count = 0
#         state = env.state
#         states, actions, rewards, log_probs = [], [], [], []
#         current_policy = trainer.latest_policy
#         while True:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             with torch.no_grad():
#                 dist, value = current_policy(state_tensor)
#                 action = dist.sample()
#                 log_prob = dist.log_prob(action)
#             if env.cur_force > 8 and action > 0:
#                 action = torch.clamp(action, -1, 0)
#             next_state, reward, done = env.step(action.numpy())
#             states.append(state)
#             actions.append(action)
#             rewards.append(reward)
#             log_probs.append(log_prob)
#             state = next_state
#             count += 1
#             if count % 100 == 0:
#                 print(count/100)
#                 trainer.data_queue.put((states, actions, rewards, log_probs), block=False) # 非阻塞方式添加数据，如果队列满则跳过
#                 states, actions, rewards, log_probs = [], [], [], []
#                 current_policy = trainer.latest_policy
    
#     except KeyboardInterrupt:
#         print("Training interrupted")
#     finally:
#         trainer.stop_event.set()
#         trainer.train_thread.join()
#         torch.save(trainer.latest_policy.state_dict(), "/home/mo/RoboGloveWS/RoboGloveProj/RLmylib/ppo_rope_lift_final.pth")


