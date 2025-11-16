import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

class BoxLiftingEnv:
    """箱子提升环境"""
    def __init__(self):
        # 系统参数
        self.mass = 10.0  # 箱子重量 10kg
        self.g = 9.81
        self.weight = self.mass * self.g  # 98.1N
        
        # 力范围限制
        self.max_hand_force = 100.0  # 人手力范围 ±100N
        self.max_rope_force = 300.0  # 拉力范围 0-300N
        self.max_pressure = 200.0    # 压力范围 0-200N
        
        # 物理参数
        self.dt = 0.01  # 时间步长
        self.max_steps = 500
        self.hand_force_change_interval = 20  # 人手力变化间隔
        
        # 状态空间: [压力, 拉力, 位置, 速度, 加速度, 人手力]
        self.state_dim = 6
        # 动作空间: 拉力变化量 (-10, -5, -2, -1, 0, 1, 2, 5, 10)
        self.action_dim = 9
        self.actions = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.step_count = 0
        self.hand_force_change_counter = 0
        
        # 初始状态
        self.hand_force = 0.0  # 初始人手力
        self.rope_force = 50.0  # 初始拉力
        self.pressure = 30.0  # 初始压力
        self.position = 0.0  # 初始位置
        self.velocity = 0.0  # 初始速度
        self.acceleration = 0.0  # 初始加速度
        
        # 更新人手力
        self._update_hand_force()
        
        return self._get_state()
    
    def _update_hand_force(self):
        """随机更新人手力"""
        if self.hand_force_change_counter <= 0:
            # 人手力在-100N到100N之间随机变化
            self.hand_force = np.random.uniform(-self.max_hand_force, self.max_hand_force)
            self.hand_force_change_counter = self.hand_force_change_interval
        else:
            self.hand_force_change_counter -= 1
    
    def _update_physics(self, rope_force_change):
        """更新物理状态"""
        # 更新拉力（带限制）
        new_rope_force = self.rope_force + rope_force_change
        self.rope_force = np.clip(new_rope_force, 0, self.max_rope_force)
        
        # 计算压力（根据人手力状态）
        if self.hand_force >= 0:
            # 人手力为正时，压力随人手力增大而增大
            base_pressure = 30.0  # 基础压力
            pressure_sensitivity = 0.8  # 压力对人手力的敏感度
            target_pressure = base_pressure + pressure_sensitivity * self.hand_force
        else:
            # 人手力为负时，压力在30-60N范围内波动
            fluctuation = 15.0 * np.sin(self.step_count * 0.1)  # 正弦波动
            target_pressure = 45.0 + fluctuation  # 45±15N
        
        # 压力平滑变化（模拟实际系统的惯性）
        pressure_change = target_pressure - self.pressure
        self.pressure += pressure_change * 0.1  # 平滑系数
        
        # 限制压力范围
        self.pressure = np.clip(self.pressure, 0, self.max_pressure)
        
        # 根据牛顿第二定律计算加速度
        total_force = self.rope_force + self.pressure - self.weight
        self.acceleration = total_force / self.mass
        
        # 更新速度和位置
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # 位置不能为负（地面支撑）
        if self.position < 0:
            self.position = 0
            self.velocity = max(0, self.velocity)  # 撞击地面后速度归零
    
    def _get_state(self):
        """获取归一化状态"""
        return np.array([
            self.pressure / self.max_pressure,  # 归一化压力
            self.rope_force / self.max_rope_force,  # 归一化拉力
            self.position / 10.0,  # 假设最大高度10m
            self.velocity / 5.0,  # 假设最大速度5m/s
            self.acceleration / 10.0,  # 假设最大加速度10m/s²
            (self.hand_force + self.max_hand_force) / (2 * self.max_hand_force)  # 归一化人手力
        ], dtype=np.float32)
    
    def step(self, action):
        """执行动作"""
        self.step_count += 1
        
        # 获取拉力变化量
        rope_force_change = self.actions[action]
        
        # 更新人手力
        self._update_hand_force()
        
        # 更新物理状态
        self._update_physics(rope_force_change)
        
        # 计算奖励
        reward = self._calculate_reward(rope_force_change)
        
        # 检查是否结束
        done = self.step_count >= self.max_steps
        
        return self._get_state(), reward, done, {}
    
    def _calculate_reward(self, rope_force_change):
        """计算奖励函数"""
        reward = 0.0
        
        # 1. 压力优化奖励（目标：保持适度压力，避免过大或过小）
        pressure_target = 50.0  # 目标压力值
        pressure_diff = abs(self.pressure - pressure_target)
        reward += -pressure_diff * 0.1  # 压力接近目标值有奖励
        
        # 2. 拉力平滑奖励（避免拉力剧烈变化）
        reward += -abs(rope_force_change) * 0.01
        
        # 3. 能量效率奖励（使用较小的拉力）
        reward += -self.rope_force * 0.001
        
        # 4. 稳定性奖励（速度和加速度较小）
        reward += -abs(self.velocity) * 0.05
        reward += -abs(self.acceleration) * 0.1
        
        # 5. 特殊情况的惩罚
        if self.pressure > 150:  # 压力过大惩罚
            reward -= 1.0
        if self.rope_force > 250:  # 拉力过大惩罚
            reward -= 0.5
        if abs(self.position) > 8:  # 位置过高惩罚
            reward -= 0.2
        
        return reward

class DQN(nn.Module):
    """DQN网络"""
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_dim, action_dim, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 训练参数
        self.batch_size = 64
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.005  # 目标网络更新系数
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
        # 更新目标网络
        self.update_target_network()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, next_state, reward, done):
        """存储经验"""
        self.memory.append((state, action, next_state, reward, done))
    
    def act(self, state):
        """选择动作"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 软更新目标网络
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return loss.item()

class PController:
    """P控制器（用于底层控制）"""
    def __init__(self, kp=1.0):
        self.kp = kp
        self.target_rope_force = 0.0
    
    def set_target(self, target):
        """设置目标拉力"""
        self.target_rope_force = target
    
    def compute_control(self, current_rope_force):
        """计算控制输出"""
        error = self.target_rope_force - current_rope_force
        return error * self.kp

def train_dqn():
    """训练DQN智能体"""
    env = BoxLiftingEnv()
    agent = DQNAgent(env.state_dim, env.action_dim)
    p_controller = PController(kp=2.0)
    
    episodes = 1000
    rewards_history = []
    losses_history = []
    
    print("开始训练DQN智能体...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        
        # 设置初始目标拉力
        p_controller.set_target(env.rope_force)
        
        done = False
        while not done:
            # DQN选择动作（拉力变化）
            action = agent.act(state)
            rope_force_change = env.actions[action]
            
            # 使用P控制器实现拉力控制
            target_force = env.rope_force + rope_force_change
            p_controller.set_target(target_force)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # 存储经验
            agent.remember(state, action, next_state, reward, done)
            
            # 经验回放
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
        
        # 记录统计信息
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        rewards_history.append(total_reward)
        losses_history.append(avg_loss)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
    
    # 绘制训练结果
    plot_training_results(rewards_history, losses_history)
    
    return agent, env

def plot_training_results(rewards, losses):
    """绘制训练结果"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 奖励曲线
    ax1.plot(rewards)
    ax1.set_title('Training Rewards')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # 损失曲线
    ax2.plot(losses)
    ax2.set_title('Training Losses')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Episode')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_agent(agent, env, episodes=5):
    """测试训练好的智能体"""
    p_controller = PController(kp=2.0)
    
    for episode in range(episodes):
        state = env.reset()
        p_controller.set_target(env.rope_force)
        
        states_history = []
        done = False
        step = 0
        
        while not done and step < 200:
            action = agent.act(state)  # 测试时使用贪婪策略
            rope_force_change = env.actions[action]
            
            target_force = env.rope_force + rope_force_change
            p_controller.set_target(target_force)
            
            next_state, reward, done, _ = env.step(action)
            
            # 记录状态
            states_history.append({
                'pressure': env.pressure,
                'rope_force': env.rope_force,
                'hand_force': env.hand_force,
                'position': env.position,
                'velocity': env.velocity,
                'reward': reward
            })
            
            state = next_state
            step += 1
        
        # 绘制测试结果
        plot_episode_results(states_history, episode)

def plot_episode_results(states_history, episode):
    """绘制单次测试的结果"""
    time_steps = range(len(states_history))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 力曲线
    pressures = [s['pressure'] for s in states_history]
    rope_forces = [s['rope_force'] for s in states_history]
    hand_forces = [s['hand_force'] for s in states_history]
    
    ax1.plot(time_steps, pressures, label='Pressure', linewidth=2)
    ax1.plot(time_steps, rope_forces, label='Rope Force', linewidth=2)
    ax1.plot(time_steps, hand_forces, label='Hand Force', linewidth=2)
    ax1.axhline(y=50, color='r', linestyle='--', label='Target Pressure')
    ax1.set_title(f'Episode {episode+1} - Forces')
    ax1.set_ylabel('Force (N)')
    ax1.legend()
    ax1.grid(True)
    
    # 运动状态
    positions = [s['position'] for s in states_history]
    velocities = [s['velocity'] for s in states_history]
    
    ax2.plot(time_steps, positions, label='Position', linewidth=2)
    ax2.plot(time_steps, velocities, label='Velocity', linewidth=2)
    ax2.set_title('Motion States')
    ax2.set_ylabel('Position (m) / Velocity (m/s)')
    ax2.legend()
    ax2.grid(True)
    
    # 奖励
    rewards = [s['reward'] for s in states_history]
    cumulative_rewards = np.cumsum(rewards)
    
    ax3.plot(time_steps, rewards, label='Step Reward', alpha=0.7)
    ax3.plot(time_steps, cumulative_rewards, label='Cumulative Reward', linewidth=2)
    ax3.set_title('Rewards')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True)
    
    # 力分布散点图
    ax4.scatter(hand_forces, pressures, c=time_steps, cmap='viridis', alpha=0.6)
    ax4.set_xlabel('Hand Force (N)')
    ax4.set_ylabel('Pressure (N)')
    ax4.set_title('Force Distribution')
    ax4.grid(True)
    ax4.axhline(y=50, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 训练智能体
    trained_agent, env = train_dqn()
    
    # 测试智能体
    print("\n开始测试训练好的智能体...")
    test_agent(trained_agent, env)