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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 100  # 目标网络更新频率
        self.steps_done = 0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def get_state(self, sensor_history):
        state = []
        while len(sensor_history) < HIS_LENGTH:
            sensor_history.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])
        recent_data = sensor_history[-HIS_LENGTH:]
        for data_point in recent_data:
            state.extend(data_point)
        return np.array(state, dtype=np.float32)
    
    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
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
        initial_reading = [15.0, 8.0, 0.0, 0.0, 0.0]  # [拉力, 压力, 位置, 速度, 加速度]
        self.sensor_history.append(initial_reading)
        return self.sensor_history
    
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
        reward = self.compute_decrease_reward(current_force, current_pressure, new_force, new_pressure)
        done = self.current_step >= self.max_episode_steps
        return self.sensor_history, reward, done
    
    def compute_decrease_reward(self, prev_force, prev_pressure, current_force, current_pressure):
        force_decrease = (prev_force - current_force) / max(1.0, prev_force)
        pressure_decrease = (prev_pressure - current_pressure) / max(1.0, prev_pressure)
        main_reward = force_decrease + pressure_decrease
        stability_bonus = 0
        if 1 <= current_force <= 2500 and 1 <= current_pressure <= 1500:
            stability_bonus = 0.3
        smoothness_bonus = 0
        if abs(force_decrease) < 500 and abs(pressure_decrease) < 300:
            smoothness_bonus = 0.2
        safety_penalty = 0
        if current_force > 4000:
            safety_penalty -= 1
        if current_pressure > 1500:
            safety_penalty -= 1
        if current_force < 300:
            safety_penalty -= 2
        total_reward = main_reward + stability_bonus + smoothness_bonus + safety_penalty
        return total_reward

def train_dqn_agent():
    env = SimEnv()
    state_size = 5 * HIS_LENGTH
    agent = DQNAgent(state_size, env.action_space)
    episodes = 100
    
    for episode in range(episodes):
        sensor_hist = env.reset()
        state = agent.get_state(sensor_hist)
        while True:
            human_force = random.uniform(-100, 100)
            action = agent.select_action(state)
            next_sensor_hist, reward, done = env.step(action, human_force)
            next_state = agent.get_state(next_sensor_hist)
            agent.memory.push(state, action, reward, next_state, done)
            agent.update_model()
            state = next_state
            if done:
                break
        if episode % 10 == 0:
            print(episode)
    return agent

def test_agent(agent, test_episodes=3):
    env = SimEnv()
    for episode in range(test_episodes):
        sensor_hist = env.reset()
        state = agent.get_state(sensor_hist)
        step = 0
        while True:
            human_force = random.uniform(-100, 100)
            action = agent.select_action(state)
            next_sensor_hist, reward, done = env.step(action, human_force)
            next_state = agent.get_state(next_sensor_hist)
            state = next_state
            step += 1
            if done:
                break

if __name__ == "__main__":
    trained_agent = train_dqn_agent()
    test_agent(trained_agent)
    torch.save(trained_agent.policy_net.state_dict(), 'dqn_motor_control_final.pth')
