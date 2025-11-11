import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        self.memory = ReplayBuffer(2000)
        
        # 超参数
        self.gamma = 0.99    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.tau = 0.01      # 目标网络软更新参数
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络初始化
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 同步目标网络权重
        self.update_target_network(tau=1.0)
    
    def update_target_network(self, tau=None):
        """更新目标网络权重（软更新）"""
        if tau is None:
            tau = self.tau
            
        for target_param, q_param in zip(self.target_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)
    
    def act(self, state):
        """根据当前状态选择动作"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到记忆回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """从记忆回放中学习"""
        if len(self.memory) < self.batch_size:
            return
        
        # 从记忆回放中采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 转换为PyTorch张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_dqn():
    """训练DQN代理"""
    # 初始化环境
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建DQN代理
    agent = DQNAgent(state_size, action_size)
    
    # 训练参数
    episodes = 250
    scores = []  # 存储每个episode的得分
    average_scores = []  # 存储平均得分
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 经验回放
            agent.replay()
            
            # 软更新目标网络
            agent.update_target_network()
            
            if done:
                scores.append(steps)
                
                # 计算最近100个episode的平均得分
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                average_scores.append(avg_score)
                
                if e % 1 == 0:
                    print(f"Episode: {e+1}/{episodes}, Score: {steps}, "
                          f"Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
                
                # 如果平均得分达到195，认为问题已解决
                if avg_score >= 195:
                    print(f"问题在 {e+1} 个episode后解决!")
                    agent.save('cartpole_dqn_solved.pth')
                
                break
    
    # 关闭环境
    env.close()
    
    # 绘制训练结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('DQN Performance on CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(average_scores)
    plt.title('Average Scores (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return agent

def test_agent(agent, episodes=10, render=True):
    """测试训练好的代理"""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 如果agent为None，创建新agent并加载权重
    if agent is None:
        agent = DQNAgent(state_size, action_size)
        agent.load('cartpole_dqn_solved.pth')
    
    scores = []
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            if render:
                env.render()
            
            # 使用训练好的模型选择动作（不探索）
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            state = next_state
            total_reward += reward
            
            if done:
                scores.append(total_reward)
                print(f"Test Episode: {e+1}, Score: {total_reward}")
                break
    
    env.close()
    print(f"平均测试得分: {np.mean(scores):.2f}")
    return scores

if __name__ == "__main__":
    # 训练代理
    print("开始训练DQN代理...")
    trained_agent = train_dqn()
    
    # 保存最终模型
    trained_agent.save('cartpole_dqn_final.pth')
    
    # 测试代理
    print("\n开始测试训练好的代理...")
    test_scores = test_agent(trained_agent, episodes=5, render=True)