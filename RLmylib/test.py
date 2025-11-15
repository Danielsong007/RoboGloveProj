import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

HIS_LENGTH = 3

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)  # 降低学习率
        
        self.memory = ReplayBuffer(10000)
        self.batch_size = 32
        
        self.gamma = 0.95  # 降低折扣因子
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # 提高最小探索率
        self.epsilon_decay = 0.998
        
        self.steps_done = 0
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_state(self, sensor_data):
        recent_data = sensor_data[-HIS_LENGTH:] if len(sensor_data) >= HIS_LENGTH else sensor_data
        while len(recent_data) < HIS_LENGTH:
            recent_data.insert(0, [0.0, 0.0])
        
        state = []
        for data in recent_data:
            state.extend(data)
        return np.array(state, dtype=np.float32)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.steps_done += 1
        return loss.item()

class SimpleEnvironment:
    def __init__(self):
        self.action_space = 5
        self.max_steps = 50  # 进一步缩短episode
        self.current_step = 0
        self.sensor_history = []
        
    def reset(self):
        self.current_step = 0
        self.sensor_history = []
        # 固定初始状态
        initial_force = 30.0
        initial_pressure = 15.0
        self.sensor_history.append([initial_force, initial_pressure])
        return self.sensor_history
    
    def step(self, action):
        self.current_step += 1
        current_data = self.sensor_history[-1]
        
        # 更简单的物理模型
        torque_change = action - 2  # [-2, -1, 0, 1, 2]
        
        # 力变化
        force_change = torque_change * 2  # 进一步减小变化幅度
        new_force = max(0.1, current_data[0] + force_change)
        
        # 压力变化
        pressure_change = -force_change * 0.8
        new_pressure = max(0.1, current_data[1] + pressure_change)
        
        new_data = [new_force, new_pressure]
        self.sensor_history.append(new_data)
        
        if len(self.sensor_history) > HIS_LENGTH + 5:
            self.sensor_history = self.sensor_history[-(HIS_LENGTH + 5):]
        
        reward = self.compute_reward()
        done = self.current_step >= self.max_steps
        
        return self.sensor_history, reward, done

    def compute_reward(self):
        """修复奖励函数：使用增量奖励而不是绝对值惩罚"""
        if len(self.sensor_history) < 2:
            return 0
            
        current_data = self.sensor_history[-1]
        prev_data = self.sensor_history[-2]
        
        current_force, current_pressure = current_data
        prev_force, prev_pressure = prev_data
        
        # 目标：力和压力都趋近于0
        target_force = 0.0
        target_pressure = 0.0
        
        # 使用距离目标的改进作为奖励（增量奖励）
        prev_force_distance = abs(prev_force - target_force)
        current_force_distance = abs(current_force - target_force)
        force_improvement = prev_force_distance - current_force_distance
        
        prev_pressure_distance = abs(prev_pressure - target_pressure)
        current_pressure_distance = abs(current_pressure - target_pressure)
        pressure_improvement = prev_pressure_distance - current_pressure_distance
        
        # 主要奖励：向目标靠近
        reward = force_improvement + pressure_improvement
            
        return reward

def train_dqn_agent():
    env = SimpleEnvironment()
    state_size = HIS_LENGTH * 2
    agent = DQNAgent(state_size, env.action_space)
    
    episodes = 2000
    rewards_history = []
    losses_history = []
    force_history = []
    pressure_history = []
    
    for episode in range(episodes):
        sensor_data = env.reset()
        state = agent.get_state(sensor_data)
        total_reward = 0
        episode_losses = []
        episode_forces = []
        episode_pressures = []
        
        while True:
            action = agent.select_action(state)
            next_sensor_data, reward, done = env.step(action)
            next_state = agent.get_state(next_sensor_data)
            
            current_force = sensor_data[-1][0]
            current_pressure = sensor_data[-1][1]
            episode_forces.append(current_force)
            episode_pressures.append(current_pressure)
            
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.update_model()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            sensor_data = next_sensor_data
            total_reward += reward
            
            if done:
                break
        
        avg_force = np.mean(episode_forces)
        avg_pressure = np.mean(episode_pressures)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        rewards_history.append(total_reward)
        losses_history.append(avg_loss)
        force_history.append(avg_force)
        pressure_history.append(avg_pressure)
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else total_reward
            avg_loss_50 = np.mean(losses_history[-50:]) if losses_history else avg_loss
            avg_force_50 = np.mean(force_history[-50:]) if force_history else avg_force
            avg_pressure_50 = np.mean(pressure_history[-50:]) if pressure_history else avg_pressure
            
            print(f"Ep {episode:4d}: Reward={total_reward:7.2f}, "
                  f"AvgR={avg_reward:7.2f}, Loss={avg_loss:7.4f}, "
                  f"Force={avg_force:5.1f}, Pressure={avg_pressure:5.1f}, "
                  f"Eps={agent.epsilon:.3f}")
            
            # 如果损失过大，调整学习率
            if avg_loss > 1000:
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  Learning rate reduced to {agent.optimizer.param_groups[0]['lr']}")

    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(losses_history)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(force_history)
    plt.title('Average Force')
    plt.xlabel('Episode')
    plt.ylabel('Force')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(pressure_history)
    plt.title('Average Pressure')
    plt.xlabel('Episode')
    plt.ylabel('Pressure')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return agent, force_history, pressure_history, losses_history

def test_agent(agent, test_episodes=3):
    env = SimpleEnvironment()
    
    for episode in range(test_episodes):
        sensor_data = env.reset()
        state = agent.get_state(sensor_data)
        total_reward = 0
        forces = []
        pressures = []
        
        print(f"\nTest Episode {episode + 1}:")
        print("Step | Action | Force | Pressure | Reward")
        print("-" * 40)
        
        step = 0
        while True:
            action = agent.select_action(state)
            next_sensor_data, reward, done = env.step(action)
            next_state = agent.get_state(next_sensor_data)
            
            current_data = sensor_data[-1]
            forces.append(current_data[0])
            pressures.append(current_data[1])
            
            if step % 5 == 0:
                print(f"{step:4d} | {action:6d} | {current_data[0]:5.1f} | {current_data[1]:8.1f} | {reward:6.2f}")
            
            state = next_state
            sensor_data = next_sensor_data
            total_reward += reward
            step += 1
            
            if done:
                break
        
        avg_force = np.mean(forces)
        avg_pressure = np.mean(pressures)
        print(f"Final - Avg Force: {avg_force:.1f}, Avg Pressure: {avg_pressure:.1f}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    print("开始训练修复版DQN")
    print("目标：最小化拉力和压力，使用增量奖励")
    trained_agent, force_hist, pressure_hist, loss_hist = train_dqn_agent()
    
    print("\n测试训练结果：")
    test_agent(trained_agent)
    
    torch.save(trained_agent.policy_net.state_dict(), 'fixed_dqn.pth')
    print("模型已保存")