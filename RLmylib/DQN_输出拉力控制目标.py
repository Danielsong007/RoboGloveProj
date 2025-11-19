import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

class BoxLiftingEnv:
    """箱子提升环境（优化版）"""
    def __init__(self, history_length=10):
        # 系统参数 - 调整质量
        self.mass = 20.0  # 箱子重量增加到20kg
        self.g = 9.81
        self.weight = self.mass * self.g  # 196.2N
        
        # 力范围限制 - 调整压力范围
        self.max_hand_force = 100.0
        self.max_rope_force = 300.0
        self.max_pressure = 80.0  # 减小最大压力
        
        # 理想压力范围调整
        self.ideal_pressure_min = 35.0
        self.ideal_pressure_max = 45.0
        
        # 物理参数
        self.dt = 0.01
        self.max_steps = 500
        self.hand_force_change_interval = 20
        self.history_length = history_length
        
        # 状态和动作空间
        self.state_dim = 7 * history_length
        self.action_dim = 3
        self.actions = [-5.0, 0.0, 5.0]
        
        # 控制参数
        self.rope_control_noise = 2.0  # 减小噪声
        
        # 安全位置范围
        self.safe_position_min = 0.0
        self.safe_position_max = 2.0
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.step_count = 0
        self.hand_force_change_counter = 0
        
        # 初始状态
        self.hand_force = 0.0
        self.rope_force_target = 150.0  # 增加初始拉力（因为质量增加了）
        self.rope_force = 150.0
        self.pressure = 30.0
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        
        # 历史数据
        self.pressure_history = deque([self.pressure] * self.history_length, maxlen=self.history_length)
        self.rope_force_history = deque([self.rope_force] * self.history_length, maxlen=self.history_length)
        self.position_history = deque([self.position] * self.history_length, maxlen=self.history_length)
        self.velocity_history = deque([self.velocity] * self.history_length, maxlen=self.history_length)
        self.acceleration_history = deque([self.acceleration] * self.history_length, maxlen=self.history_length)
        self.action_history = deque([0] * self.history_length, maxlen=self.history_length)
        self.reward_history = deque([0.0] * self.history_length, maxlen=self.history_length)
        
        self._update_hand_force()
        return self._get_state()
    
    def _update_hand_force(self):
        """随机更新人手力"""
        if self.hand_force_change_counter <= 0:
            self.hand_force = np.random.uniform(-self.max_hand_force, self.max_hand_force)
            self.hand_force_change_counter = self.hand_force_change_interval
        else:
            self.hand_force_change_counter -= 1
    
    def _update_pressure(self):
        """更新压力传感器读数"""
        if self.hand_force >= 0:
            base_pressure = 25.0  # 降低基础压力
            pressure_sensitivity = 0.5  # 降低敏感度
            target_pressure = base_pressure + pressure_sensitivity * self.hand_force
        else:
            fluctuation = 10.0 * np.sin(self.step_count * 0.1)  # 减小波动
            target_pressure = 35.0 + fluctuation  # 调整基准值
        
        # 更平滑的压力变化
        pressure_change = target_pressure - self.pressure
        self.pressure += pressure_change * 0.05  # 减小平滑系数
        
        self.pressure = np.clip(self.pressure, 0, self.max_pressure)
    
    def _update_physics(self, rope_force_change):
        """更新物理状态"""
        # 更新拉力目标
        self.rope_force_target += rope_force_change
        self.rope_force_target = np.clip(self.rope_force_target, 0, self.max_rope_force)
        
        # 更平缓的拉力控制
        force_error = self.rope_force_target - self.rope_force
        control_gain = 0.3  # 减小控制增益
        noise = np.random.normal(0, self.rope_control_noise)
        
        self.rope_force += control_gain * force_error + noise
        self.rope_force = np.clip(self.rope_force, 0, self.max_rope_force)
        
        # 更新压力
        self._update_pressure()
        
        # 物理计算
        total_force = self.rope_force + self.hand_force - self.weight
        self.acceleration = total_force / self.mass
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt
        
        if self.position < 0:
            self.position = 0
            self.velocity = max(0, self.velocity)
    
    def _get_state(self):
        """获取归一化状态"""
        norm_pressure = [p / self.max_pressure for p in self.pressure_history]
        norm_rope_force = [f / self.max_rope_force for f in self.rope_force_history]
        norm_position = [p / 5.0 for p in self.position_history]
        norm_velocity = [v / 3.0 for v in self.velocity_history]
        norm_acceleration = [a / 5.0 for a in self.acceleration_history]
        norm_action = [a / (self.action_dim - 1) for a in self.action_history]
        norm_reward = [r / 10.0 for r in self.reward_history]  # 调整奖励归一化
        
        state = (norm_pressure + norm_rope_force + norm_position + 
                norm_velocity + norm_acceleration + norm_action + norm_reward)
        
        return np.array(state, dtype=np.float32)
    
    def _update_history(self, action_idx, reward):
        """更新历史数据"""
        self.pressure_history.append(self.pressure)
        self.rope_force_history.append(self.rope_force)
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        self.acceleration_history.append(self.acceleration)
        self.action_history.append(action_idx)
        self.reward_history.append(reward)
    
    def step(self, action):
        """执行动作"""
        self.step_count += 1
        
        rope_force_change = self.actions[action]
        self._update_hand_force()
        self._update_physics(rope_force_change)
        
        reward = self._calculate_reward(rope_force_change)
        self._update_history(action, reward)
        
        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done, {}
    
    def _calculate_reward(self, rope_force_change):
        """简化的奖励函数"""
        reward = 0.0
        
        # 1. 主要目标：压力控制（简化，线性奖励）
        if self.ideal_pressure_min <= self.pressure <= self.ideal_pressure_max:
            # 理想范围内有大奖励
            reward += 2.0
        else:
            # 范围外有线性惩罚，避免非线性导致的梯度问题
            if self.pressure < self.ideal_pressure_min:
                distance = self.ideal_pressure_min - self.pressure
            else:
                distance = self.pressure - self.ideal_pressure_max
            reward -= distance * 0.1  # 线性惩罚
        
        # 2. 次要目标：拉力效率（权重降低）
        if self.rope_force < 200:  # 提高阈值
            reward += 0.1  # 小奖励
        else:
            reward -= (self.rope_force - 200) * 0.01  # 轻微惩罚
        
        # 3. 位置安全（重要但权重适中）
        if self.safe_position_min <= self.position <= self.safe_position_max:
            reward += 0.5
        else:
            if self.position < self.safe_position_min:
                distance = self.safe_position_min - self.position
            else:
                distance = self.position - self.safe_position_max
            reward -= distance * 0.2
        
        # 4. 控制平滑性（保留，但权重降低）
        if len(self.rope_force_history) > 1:
            force_change = abs(self.rope_force - list(self.rope_force_history)[-2])
            if force_change > 5.0:  # 提高容忍度
                reward -= 0.05
        
        # 5. 移除稳定性奖励（速度和加速度），专注于主要目标
        
        # 6. 避免极端情况的重大惩罚
        if self.pressure > 70:  # 只有在极端情况下才有大惩罚
            reward -= 1.0
        if self.position > 2.5:
            reward -= 1.0
        
        return reward

class DQN(nn.Module):
    """DQN网络（优化结构）"""
    def __init__(self, state_dim, action_dim, hidden_size=128):  # 减小网络规模
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN智能体（优化训练参数）"""
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):  # 提高学习率
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 训练参数优化
        self.batch_size = 64  # 减小batch size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # 提高最小探索率
        self.epsilon_decay = 0.998
        self.tau = 0.01
        self.target_update_freq = 100  # 目标网络更新频率
        
        self.memory = deque(maxlen=10000)
        self.update_count = 0
        
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        # 修复张量创建警告
        if isinstance(state, list):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 修复张量创建警告
        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # 确保是numpy数组
        states = np.array(states)
        next_states = np.array(next_states)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions).squeeze()
        
        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (~dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 控制探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()

def train_dqn():
    """训练函数"""
    history_length = 5  # 减小历史长度
    env = BoxLiftingEnv(history_length=history_length)
    agent = DQNAgent(env.state_dim, env.action_dim)
    
    episodes = 1000
    rewards_history = []
    avg_rewards = []
    
    print("开始训练优化版DQN智能体...")
    print(f"质量: {env.mass}kg, 理想压力: {env.ideal_pressure_min}-{env.ideal_pressure_max}N")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_pressures = []
        
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            episode_pressures.append(env.pressure)
            agent.remember(state, action, next_state, reward, done)
            
            loss = agent.replay()
            state = next_state
        
        rewards_history.append(total_reward)
        avg_50 = np.mean(rewards_history[-50:]) if episode >= 50 else total_reward
        avg_rewards.append(avg_50)
        
        pressure_in_range = np.mean([env.ideal_pressure_min <= p <= env.ideal_pressure_max 
                                   for p in episode_pressures])
        
        if episode % 50 == 0 or episode < 10:
            print(f"Episode {episode:4d}, Reward: {total_reward:7.1f}, "
                  f"Avg50: {avg_50:7.1f}, ε: {agent.epsilon:.3f}, "
                  f"Press%: {pressure_in_range:.2f}")
        
        # 早期停止检查
        if episode > 100 and avg_50 > 0:  # 如果平均奖励为正，说明学习有效
            print("训练出现积极迹象，继续训练...")
    
    # 绘制简单结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, alpha=0.3, label='Raw')
    plt.plot(avg_rewards, label='Moving Avg (50)')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(rewards_history, bins=50, alpha=0.7)
    plt.title('Reward Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return agent, env

if __name__ == "__main__":
    trained_agent, env = train_dqn()