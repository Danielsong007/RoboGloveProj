import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt


HIS_LENGTH = 10  # 历史数据长度
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, history_length=10):
        self.state_size = state_size
        self.action_size = action_size
        self.history_length = history_length
        
        # 网络参数
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # 经验回放
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        
        # 训练参数
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 100  # 目标网络更新频率
        
        self.steps_done = 0
        
        # 初始化目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def get_state(self, sensor_data):
        """构建状态向量：历史拉力值 + 历史压力值 + 历史位置速度加速度"""
        # sensor_data格式: [拉力, 压力, 位置, 速度, 加速度]
        state = []
        
        # 添加历史数据（最近history_length个时间步）
        for i in range(self.history_length):
            idx = max(0, len(sensor_data) - self.history_length + i)
            if idx < len(sensor_data):
                state.extend(sensor_data[idx])
            else:
                # 如果数据不足，用0填充
                state.extend([0, 0, 0, 0, 0])
        
        return np.array(state, dtype=np.float32)
    
    def select_action(self, state):
        """选择动作"""
        self.steps_done += 1
        
        # epsilon-greedy策略
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def compute_reward(self, prev_sensor_data, current_sensor_data):
        """计算奖励：拉力值增大率与压力值增大率的求和"""
        if len(prev_sensor_data) < 2 or len(current_sensor_data) < 2:
            return 0
            
        # 获取最新的传感器数据
        prev_force = prev_sensor_data[-1][0]  # 拉力值
        prev_pressure = prev_sensor_data[-1][1]  # 压力值
        current_force = current_sensor_data[-1][0]
        current_pressure = current_sensor_data[-1][1]
        
        # 计算增大率（避免除零）
        force_increase_rate = 0
        pressure_increase_rate = 0
        
        if prev_force != 0:
            force_increase_rate = (current_force - prev_force) / abs(prev_force)
        
        if prev_pressure != 0:
            pressure_increase_rate = (current_pressure - prev_pressure) / abs(prev_pressure)
        
        # 奖励是增大率的求和
        reward = force_increase_rate + pressure_increase_rate
        
        # 添加平滑性约束（可选）
        # 惩罚过大的扭矩变化
        if len(prev_sensor_data) > 1:
            # 这里可以添加对动作变化的惩罚
            
            # 添加安全性约束
            if current_force > 100:  # 假设最大拉力限制
                reward -= 10
            if abs(current_pressure) > 50:  # 假设最大压力限制
                reward -= 10
        
        return reward
    
    def update_model(self):
        """更新DQN模型"""
        if len(self.memory) < self.batch_size:
            return
        
        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 定期更新目标网络
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

class MotorControlEnvironment:
    def __init__(self):
        # 动作空间：扭矩控制目标的增大与减小（离散值）
        # 0: 大幅减小扭矩, 1: 小幅减小扭矩, 2: 保持扭矩, 3: 小幅增大扭矩, 4: 大幅增大扭矩
        self.action_space = 5
        
        # 状态空间：每个时间步包含[拉力, 压力, 位置, 速度, 加速度]
        self.state_dim = 5
        
        # 模拟参数
        self.max_episode_steps = 200
        self.current_step = 0
        
        # 模拟传感器数据历史
        self.sensor_history = []

        self.box_weight = 15.0  # 15 kg

        
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.sensor_history = []
        
        # 初始传感器读数
        initial_reading = [10.0, 5.0, 0.0, 0.0, 0.0]  # [拉力, 压力, 位置, 速度, 加速度]
        self.sensor_history.append(initial_reading)
        
        return self.sensor_history
    
    # def step(self, action, human_intent_force=0):
    #     """执行动作并返回新状态和奖励"""
    #     self.current_step += 1
        
    #     # 获取当前状态
    #     current_reading = self.sensor_history[-1].copy()
        
    #     # 根据动作调整扭矩（这里简化处理）
    #     torque_change = action - 2  # 将动作映射到[-2, -1, 0, 1, 2]
    #     torque_adjustment = torque_change * 0.5  # 缩放因子
        
    #     # 模拟物理响应（简化模型）
    #     new_force = max(0, current_reading[0] + torque_adjustment + human_intent_force * 0.1)
    #     new_pressure = max(0, current_reading[1] - human_intent_force * 0.05 + random.uniform(-0.5, 0.5))
        
    #     # 更新位置、速度、加速度（简化运动学）
    #     acceleration = (new_force - current_reading[0]) * 0.1
    #     new_velocity = current_reading[3] + acceleration * 0.1
    #     new_position = current_reading[2] + new_velocity * 0.1
        
    #     new_reading = [new_force, new_pressure, new_position, new_velocity, acceleration]
    #     self.sensor_history.append(new_reading)
        
    #     # 计算奖励
    #     reward = self.compute_reward_difference(self.sensor_history)
        
    #     # 检查是否结束
    #     done = self.current_step >= self.max_episode_steps
        
    #     return self.sensor_history, reward, done


    def step(self, action, human_intent_force=0):
        self.current_step += 1
        current_reading = self.sensor_history[-1]
        current_force, current_pressure, current_position, current_velocity, current_acceleration = current_reading
        torque_change = action - 2  # 将动作映射到[-2, -1, 0, 1, 2]
        torque_adjustment = torque_change * 100
        force_change = torque_adjustment - human_intent_force * 2
        pressure_change = human_intent_force * 0.8 - torque_adjustment * 0.5
        new_force = max(1.0, current_force + force_change + random.uniform(-50, 50))
        new_pressure = max(1.0, current_pressure + pressure_change + random.uniform(-50, 50))
        new_acceleration = (new_force - self.box_weight*9.8) / self.box_weight
        new_velocity = current_velocity + new_acceleration * 0.01
        new_position = max(0.0, current_position + new_velocity * 0.01)
        new_reading = [new_force, new_pressure, new_position, new_velocity, new_acceleration]
        self.sensor_history.append(new_reading)
        if len(self.sensor_history) > HIS_LENGTH + 10:
            self.sensor_history = self.sensor_history[-(HIS_LENGTH + 10):]
        reward = self.compute_reward_difference(self.sensor_history)
        done = self.current_step >= self.max_episode_steps
        return self.sensor_history, reward, done

    
    def compute_reward_difference(self, sensor_data):
        """计算基于增量的奖励函数"""
        if len(sensor_data) < 2:
            return 0
            
        prev_data = sensor_data[-2]
        current_data = sensor_data[-1]
        
        # 拉力增大率
        force_increase = (current_data[0] - prev_data[0]) / max(1, abs(prev_data[0]))
        
        # 压力增大率（注意：我们希望压力减小，所以取负号）
        pressure_increase = -(current_data[1] - prev_data[1]) / max(1, abs(prev_data[1]))
        
        # 总奖励
        reward = force_increase + pressure_increase
        
        return reward

# 训练函数
def train_dqn_agent():
    # 初始化环境和智能体
    env = MotorControlEnvironment()
    state_size = env.state_dim * 10  # 10个历史时间步
    agent = DQNAgent(state_size, env.action_space, history_length=10)
    
    # 训练参数
    episodes = 1000
    rewards_history = []
    loss_history = []
    
    for episode in range(episodes):
        sensor_data = env.reset()
        state = agent.get_state(sensor_data)
        total_reward = 0
        episode_losses = []
        
        while True:
            # 模拟人类意图（随机变化）
            human_force = random.uniform(-5, 5)
            
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_sensor_data, reward, done = env.step(action, human_force)
            next_state = agent.get_state(next_sensor_data)
            
            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)
            
            # 更新模型
            loss = agent.update_model()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 记录训练进度
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        rewards_history.append(total_reward)
        loss_history.append(avg_loss)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
    
    return agent

# 测试函数
def test_agent(agent, test_episodes=5):
    env = MotorControlEnvironment()
    
    for episode in range(test_episodes):
        sensor_data = env.reset()
        state = agent.get_state(sensor_data)
        total_reward = 0
        
        print(f"\nTest Episode {episode + 1}")
        print("Step | Action | Force | Pressure | Reward")
        print("-" * 40)
        
        step = 0
        while True:
            human_force = random.uniform(-3, 3)  # 测试时使用较小的变化
            
            action = agent.select_action(state)
            next_sensor_data, reward, done = env.step(action, human_force)
            next_state = agent.get_state(next_sensor_data)
            
            current_data = sensor_data[-1]
            print(f"{step:4d} | {action:6d} | {current_data[0]:5.1f} | {current_data[1]:8.1f} | {reward:6.2f}")
            
            state = next_state
            sensor_data = next_sensor_data
            total_reward += reward
            step += 1
            
            if done:
                break
        
        print(f"Total Reward: {total_reward:.2f}")

# 主程序
if __name__ == "__main__":
    # 训练智能体
    print("开始训练DQN智能体...")
    trained_agent = train_dqn_agent()
    
    # 测试智能体
    print("\n开始测试智能体...")
    test_agent(trained_agent)
    
    # 保存模型
    torch.save(trained_agent.policy_net.state_dict(), 'dqn_motor_control.pth')
    print("模型已保存为 'dqn_motor_control.pth'")