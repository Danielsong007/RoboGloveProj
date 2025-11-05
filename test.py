import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# 常量定义
HIS_LENGTH = 10  # 历史数据长度

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, action_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        return self.fc5(x)

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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005, weight_decay=1e-5)
        self.memory = ReplayBuffer(capacity=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.target_update = 200
        self.steps_done = 0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 存储历史action和reward
        self.action_history = []
        self.reward_history = []
        
    def get_state(self, sensor_history, action_history, reward_history):
        """构建包含传感器数据、历史action和reward的状态向量"""
        state = []
        
        # 处理传感器历史数据
        while len(sensor_history) < HIS_LENGTH:
            sensor_history.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])
        recent_sensor_data = sensor_history[-HIS_LENGTH:]
        for data_point in recent_sensor_data:
            state.extend(data_point)
        
        # 处理历史action数据
        while len(action_history) < HIS_LENGTH:
            action_history.insert(0, 0)
        recent_actions = action_history[-HIS_LENGTH:]
        # 对action进行one-hot编码
        for action in recent_actions:
            action_one_hot = [0] * self.action_size
            if 0 <= action < self.action_size:
                action_one_hot[action] = 1
            state.extend(action_one_hot)
        
        # 处理历史reward数据
        while len(reward_history) < HIS_LENGTH:
            reward_history.insert(0, 0.0)
        recent_rewards = reward_history[-HIS_LENGTH:]
        state.extend(recent_rewards)
        
        return np.array(state, dtype=np.float32)
    
    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        
        # 记录action到历史
        self.action_history.append(action)
        if len(self.action_history) > HIS_LENGTH + 10:
            self.action_history = self.action_history[-(HIS_LENGTH + 10):]
            
        return action
    
    def record_reward(self, reward):
        """记录reward到历史"""
        self.reward_history.append(reward)
        if len(self.reward_history) > HIS_LENGTH + 10:
            self.reward_history = self.reward_history[-(HIS_LENGTH + 10):]
    
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

class SimEnv:
    def __init__(self):
        self.action_space = 5
        self.sensor_history = []
        self.max_episode_steps = 200
        self.current_step = 0
        self.box_weight = 15.0  # 15 kg
        
    def reset(self):
        self.current_step = 0
        self.sensor_history = []
        initial_reading = [15.0, 8.0, 0.0, 0.0, 0.0]
        self.sensor_history.append(initial_reading)
        return self.sensor_history
    
    def step(self, action, human_intent_force=0):
        self.current_step += 1
        current_reading = self.sensor_history[-1]
        current_force, current_pressure, current_position, current_velocity, current_acceleration = current_reading
        
        torque_change = action - 2
        torque_adjustment = torque_change * 100
        
        # 修正物理模型
        force_change = torque_adjustment - human_intent_force * 2
        pressure_change = human_intent_force * 0.8 - torque_adjustment * 0.5
        
        # 添加更真实的噪声模型
        new_force = max(1.0, current_force + force_change + random.uniform(-50, 50))
        new_pressure = max(1.0, current_pressure + pressure_change + random.uniform(-50, 50))
        
        # 更精确的物理计算
        net_force = new_force - self.box_weight * 9.8
        new_acceleration = net_force / self.box_weight if self.box_weight > 0 else 0
        new_velocity = current_velocity + new_acceleration * 0.01
        new_position = max(0.0, current_position + new_velocity * 0.01)
        
        new_reading = [new_force, new_pressure, new_position, new_velocity, new_acceleration]
        self.sensor_history.append(new_reading)
        
        if len(self.sensor_history) > HIS_LENGTH + 10:
            self.sensor_history = self.sensor_history[-(HIS_LENGTH + 10):]
        
        reward = self.compute_decrease_reward(current_force, current_pressure, new_force, new_pressure)
        done = self.current_step >= self.max_episode_steps
        
        return self.sensor_history, reward, done
    
    def compute_decrease_reward(self, prev_force, prev_pressure, current_force, current_pressure):
        # 改进奖励函数，避免除零错误
        if abs(prev_force) < 1e-6:
            force_decrease = 0
        else:
            force_decrease = (prev_force - current_force) / abs(prev_force)
            
        if abs(prev_pressure) < 1e-6:
            pressure_decrease = 0
        else:
            pressure_decrease = (prev_pressure - current_pressure) / abs(prev_pressure)
        
        main_reward = force_decrease + pressure_decrease
        
        # 改进稳定性奖励
        stability_bonus = 0
        if 100 <= current_force <= 2000 and 50 <= current_pressure <= 1000:
            stability_bonus = 0.5
        
        # 改进平滑性奖励
        smoothness_bonus = 0
        if abs(force_decrease) < 0.1 and abs(pressure_decrease) < 0.1:
            smoothness_bonus = 0.3
        
        # 安全性约束
        safety_penalty = 0
        if current_force > 4000:
            safety_penalty -= 2
        if current_pressure > 1500:
            safety_penalty -= 2
        if current_force < 100:
            safety_penalty -= 1
        
        total_reward = main_reward + stability_bonus + smoothness_bonus + safety_penalty
        
        # 限制奖励范围
        return np.clip(total_reward, -5, 5)

def train_dqn_agent():
    env = SimEnv()
    # 新的状态维度：传感器数据(5 * 10) + action one-hot(5 * 10) + reward(10)
    state_size = 5 * HIS_LENGTH + env.action_space * HIS_LENGTH + HIS_LENGTH
    agent = DQNAgent(state_size, env.action_space)
    episodes = 500
    
    rewards_history = []
    losses_history = []
    
    for episode in range(episodes):
        # 重置环境和历史记录
        sensor_hist = env.reset()
        agent.action_history = []
        agent.reward_history = []
        
        # 获取初始状态
        state = agent.get_state(sensor_hist, agent.action_history, agent.reward_history)
        total_reward = 0
        episode_losses = []
        
        while True:
            human_force = random.uniform(-100, 100)
            action = agent.select_action(state)
            next_sensor_hist, reward, done = env.step(action, human_force)
            
            # 记录reward
            agent.record_reward(reward)
            
            next_state = agent.get_state(next_sensor_hist, agent.action_history, agent.reward_history)
            agent.memory.push(state, action, reward, next_state, done)
            
            loss = agent.update_model()
            if loss > 0:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards_history.append(total_reward)
        if episode_losses:
            losses_history.append(np.mean(episode_losses))
        
        if episode % 20 == 0:
            avg_loss = np.mean(episode_losses) if episode_losses else 0
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
    plt.plot(losses_history)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()
    
    return agent

def test_agent(agent, test_episodes=3):
    env = SimEnv()
    
    for episode in range(test_episodes):
        sensor_hist = env.reset()
        agent.action_history = []
        agent.reward_history = []
        
        state = agent.get_state(sensor_hist, agent.action_history, agent.reward_history)
        total_reward = 0
        
        human_forces = []
        actions = []
        rewards = []
        forces = []
        pressures = []
        steps = []
        
        step = 0
        while True:
            human_force = random.uniform(-100, 100)
            action = agent.select_action(state)
            next_sensor_hist, reward, done = env.step(action, human_force)
            agent.record_reward(reward)
            next_state = agent.get_state(next_sensor_hist, agent.action_history, agent.reward_history)
            
            # 记录数据
            human_forces.append(human_force)
            actions.append(action)
            rewards.append(reward)
            forces.append(next_sensor_hist[-1][0])
            pressures.append(next_sensor_hist[-1][1])
            steps.append(step)
            
            state = next_state
            total_reward += reward
            step += 1
            
            if done:
                break
        
        # 绘制测试结果
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 2, 1)
        plt.plot(steps, human_forces, 'b-', label='Human Force', linewidth=2)
        plt.plot(steps, actions, 'r--', label='Action', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'Human Force vs Action (Episode {episode + 1})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 2, 2)
        plt.scatter(steps, human_forces, c=actions, cmap='viridis', s=50)
        plt.colorbar(label='Action')
        plt.xlabel('Step')
        plt.ylabel('Human Force')
        plt.title('Human Force Distribution')
        plt.grid(True)
        
        plt.subplot(3, 2, 3)
        plt.plot(steps, forces, 'g-', label='Force', linewidth=2)
        plt.plot(steps, pressures, 'm-', label='Pressure', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Sensor Value')
        plt.title('Force and Pressure')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 2, 4)
        plt.plot(steps, rewards, 'c-', label='Reward', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward over Time')
        plt.grid(True)
        
        plt.subplot(3, 2, 5)
        action_counts = [actions.count(i) for i in range(5)]
        action_labels = ['大幅减小', '小幅减小', '保持', '小幅增大', '大幅增大']
        plt.bar(action_labels, action_counts, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
        plt.title('Action Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, 6)
        plt.hist(rewards, bins=20, alpha=0.7, color='blue')
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Test Episode {episode + 1}: Total Reward = {total_reward:.2f}")

if __name__ == "__main__":
    print("开始训练增强版DQN智能体...")
    print(f"状态维度: 传感器({5*HIS_LENGTH}) + action({5*HIS_LENGTH}) + reward({HIS_LENGTH}) = {5*HIS_LENGTH + 5*HIS_LENGTH + HIS_LENGTH}")
    trained_agent = train_dqn_agent()
    
    print("\n开始测试智能体...")
    test_agent(trained_agent)
    
    torch.save(trained_agent.policy_net.state_dict(), 'enhanced_dqn_motor_control.pth')
    print("模型已保存为 'enhanced_dqn_motor_control.pth'")