import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gym
from gym import spaces
import threading
import time
import copy
import random
import queue

class Config:
    MAX_FORCE = 100.0  # 最大安全拉力(N)
    MAX_TORQUE = 5.0  # 电机最大扭矩(Nm)
    HISTORY_WINDOW = 5  # 历史数据窗口大小
    SINGLE_STATE_NUM = 6
    CONTROL_FREQ = 100  # 控制频率(Hz)
    DT = 1.0 / CONTROL_FREQ
    TARGET_FORCE = 10.0  # 目标拉力

class RopeLiftEnv(gym.Env):
    def __init__(self):
        super(RopeLiftEnv, self).__init__()
        self.state_dim = Config.SINGLE_STATE_NUM + 3 * Config.HISTORY_WINDOW
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
        return self._get_state()

    def step(self, action):
        self.pressure += np.random.normal(0, 0.02)
        self.pressure = np.clip(self.pressure, 0, 2)
        self.velocity = np.random.normal(-1, 1)
        self.position = np.random.normal(-1, 1)
        self.current_force = np.random.normal(1, 3)
        self.force_integral += (self.current_force - Config.TARGET_FORCE) * Config.DT
        self.pressure_integral += (self.pressure - 1.0) * Config.DT
        self.force_history.append(self.current_force)
        self.pressure_history.append(self.pressure)
        self.action_history.append(action[0])
        
        # 改进的奖励函数
        force_error = abs(self.current_force - Config.TARGET_FORCE) / Config.MAX_FORCE
        smoothness = -0.1 * abs(action[0] - (list(self.action_history)[-2] if len(self.action_history) > 1 else 0))
        reward = -force_error + smoothness
        
        done = self.current_force > Config.MAX_FORCE * 0.9
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
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU()
        )
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

class OnlineTrainer:
    def __init__(self, policy, buffer_size=10000, batch_size=64):
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)  # 目标网络，用于稳定训练
        self.optimizer = optim.Adam(policy.parameters(), lr=3e-4)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.train_lock = threading.Lock()
        self.experience_queue = queue.Queue()
        self.stop_signal = False
        self.update_counter = 0
        self.target_update_freq = 100  # 每100步同步一次目标网络
        
    def add_experience(self, state, action, reward, next_state, done):
        """添加经验到队列"""
        self.experience_queue.put((state, action, reward, next_state, done))
        
    def _process_experience_queue(self):
        """处理经验队列，转移到回放缓冲区"""
        while not self.experience_queue.empty():
            try:
                experience = self.experience_queue.get_nowait()
                self.replay_buffer.append(experience)
            except queue.Empty:
                break
                
    def train_step(self):
        """执行一次训练步骤"""
        self._process_experience_queue()
        
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # 随机采样批次
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions)).unsqueeze(-1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.BoolTensor(dones)
        
        # PPO训练逻辑
        with torch.no_grad():
            old_dist, old_values = self.target_policy(states_t)
            old_log_probs = old_dist.log_prob(actions_t)
            
        # 计算returns
        returns = self._compute_returns(rewards_t, dones_t)
        
        # 多轮更新
        for _ in range(4):
            dist, values = self.policy(states_t)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            
            # 计算优势函数
            advantages = returns - old_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        self.update_counter += 1
        # 定期同步目标网络
        if self.update_counter % self.target_update_freq == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            
    def _compute_returns(self, rewards, dones, gamma=0.99):
        """计算折扣回报"""
        returns = torch.zeros_like(rewards)
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R * (~dones[i]).float()
            returns[i] = R
        return returns
        
    def training_loop(self):
        """训练线程的主循环"""
        while not self.stop_signal:
            time.sleep(0.1)  # 每0.1秒训练一次
            with self.train_lock:
                self.train_step()
                
    def start_training(self):
        """启动训练线程"""
        self.train_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.train_thread.start()
        
    def stop_training(self):
        """停止训练"""
        self.stop_signal = True
        if hasattr(self, 'train_thread'):
            self.train_thread.join()
            
    def update_controller_policy(self, controller):
        """更新控制器的策略网络（线程安全）"""
        with self.train_lock:
            controller.policy.load_state_dict(self.target_policy.state_dict())

class RealTimeController:
    def __init__(self, initial_policy=None):
        # 初始化策略网络
        if initial_policy:
            self.policy = copy.deepcopy(initial_policy)
        else:
            env = RopeLiftEnv()
            self.policy = ActorCritic(env.state_dim)
            
        self.policy.eval()
        
        # 初始化训练器
        self.trainer = OnlineTrainer(copy.deepcopy(self.policy))
        self.trainer.start_training()
        
        # 状态跟踪
        self.sensor_buffer = deque([{'force': 0, 'pressure': 0} for _ in range(Config.HISTORY_WINDOW)], 
                                 maxlen=Config.HISTORY_WINDOW)
        self.action_history = deque([0.0] * Config.HISTORY_WINDOW, maxlen=Config.HISTORY_WINDOW)
        self.cur_torque = 0.0
        self.last_state = None
        self.last_action = 0.0
        self.step_count = 0
        self.policy_update_freq = 50  # 每50步更新一次策略
        
    def update_and_act(self, force, pressure):
        """更新传感器数据并生成动作，同时收集训练数据"""
        # 1. 更新当前状态
        current_state = self._update_state(force, pressure)
        
        # 2. 如果是第一步，只记录状态不收集经验
        if self.last_state is not None:
            # 计算奖励和完成标志
            reward = self._compute_reward(force, pressure)
            done = force > Config.MAX_FORCE * 0.9
            
            # 添加经验到训练器
            self.trainer.add_experience(
                self.last_state, 
                [self.last_action],  # 保持与训练格式一致
                reward, 
                current_state, 
                done
            )
            
            # 定期更新控制器策略
            self.step_count += 1
            if self.step_count % self.policy_update_freq == 0:
                self.trainer.update_controller_policy(self)
                print(f"Step {self.step_count}: Policy updated")
        
        # 3. 生成新动作
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.policy(state_tensor)
            action = dist.mean.item()
            
        # 安全限制
        if force > Config.MAX_FORCE * 0.8 and action > 0:
            action = max(action, -1.0)  # 限制为负值或零
            
        torque = action * Config.MAX_TORQUE
        self.cur_torque = 0.8 * torque + 0.2 * self.cur_torque
        
        # 4. 保存当前状态和动作
        self.last_state = current_state
        self.last_action = action
        self.action_history.append(action)
        
        return self.cur_torque
    
    def _compute_reward(self, force, pressure):
        """计算即时奖励"""
        force_error = abs(force - Config.TARGET_FORCE) / Config.MAX_FORCE
        smoothness = -0.1 * abs(self.last_action - (list(self.action_history)[-2] if len(self.action_history) > 1 else 0))
        return -force_error + smoothness
    
    def _update_state(self, force, pressure):
        """更新状态表示"""
        self.sensor_buffer.append({'force': force, 'pressure': pressure})
        current = self.sensor_buffer[-1]
        
        state = np.zeros(Config.SINGLE_STATE_NUM + 3 * Config.HISTORY_WINDOW, dtype=np.float32)
        state[0] = current['force'] / Config.MAX_FORCE
        state[1] = current['pressure']
        state[2] = 0.0  # 简化处理
        state[3] = 0.0
        state[4] = 0.0
        state[5] = 0.0
        
        hist_forces = [f['force'] for f in self.sensor_buffer]
        hist_pressures = [p['pressure'] for p in self.sensor_buffer]
        
        for i in range(Config.HISTORY_WINDOW):
            state[Config.SINGLE_STATE_NUM + i] = (hist_forces[i] - current['force']) / Config.MAX_FORCE
        for i in range(Config.HISTORY_WINDOW):
            state[Config.SINGLE_STATE_NUM + Config.HISTORY_WINDOW + i] = (hist_pressures[i] - current['pressure']) / 0.2
        for i in range(Config.HISTORY_WINDOW):
            state[Config.SINGLE_STATE_NUM + 2 * Config.HISTORY_WINDOW + i] = list(self.action_history)[i]
            
        return state
    
    def shutdown(self):
        """关闭控制器"""
        self.trainer.stop_training()

def main():
    """主程序：在线学习演示"""
    print("Starting online learning controller...")
    
    # 创建在线学习控制器
    controller = RealTimeController()
    
    try:
        # 模拟实时控制循环
        for episode in range(10):  # 运行10个episode
            print(f"\n=== Episode {episode + 1} ===")
            
            # 重置环境状态
            force = np.random.uniform(5, 15)
            pressure = np.random.uniform(0.5, 1.5)
            step = 0
            
            while step < 100:  # 每个episode最多100步
                # 模拟传感器数据变化
                force += np.random.normal(0, 2)
                pressure += np.random.normal(0, 0.1)
                
                force = np.clip(force, 5, Config.MAX_FORCE)
                pressure = np.clip(pressure, 0.5, 1.5)
                
                # 获取控制动作
                torque = controller.update_and_act(force, pressure)
                
                # 检查终止条件
                if force > Config.MAX_FORCE * 0.9:
                    print(f"Episode {episode + 1} terminated at step {step} (force limit)")
                    break
                
                if step % 20 == 0:
                    print(f"Step {step}: Force={force:.1f}, Pressure={pressure:.2f}, Torque={torque:.2f}")
                
                step += 1
                time.sleep(0.01)  # 模拟实时控制频率
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        controller.shutdown()

if __name__ == "__main__":
    main()