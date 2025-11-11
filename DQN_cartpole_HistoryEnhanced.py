import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HistoryEnhancedDQN(nn.Module):
    def __init__(self, state_size, action_size, history_length=2):
        super(HistoryEnhancedDQN, self).__init__()
        self.history_length = history_length
        # 状态 + 历史动作 + 历史奖励
        total_input_size = state_size + history_length * 2
        
        self.fc1 = nn.Linear(total_input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, history_state):
        self.buffer.append((state, action, reward, next_state, done, history_state))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, history_states = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(history_states, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class HistoryEnhancedAgent:
    def __init__(self, state_size, action_size, history_length=2):
        self.state_size = state_size
        self.action_size = action_size
        self.history_length = history_length
        self.memory = ReplayBuffer(10000)
        
        # 超参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.batch_size = 64
        self.tau = 0.005
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 网络
        self.q_network = HistoryEnhancedDQN(state_size, action_size, history_length).to(self.device)
        self.target_network = HistoryEnhancedDQN(state_size, action_size, history_length).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 历史记录
        self.action_history = deque(maxlen=history_length)
        self.reward_history = deque(maxlen=history_length)
        
        # 初始化历史
        for _ in range(history_length):
            self.action_history.append(0)
            self.reward_history.append(0.0)
        
        self.update_target_network(tau=1.0)
    
    def get_history_state(self, current_state):
        """构建包含历史信息的增强状态"""
        action_history = np.array(list(self.action_history), dtype=np.float32)
        reward_history = np.array(list(self.reward_history), dtype=np.float32)
        
        # 归一化奖励历史
        if len(reward_history) > 0:
            reward_history = reward_history / 10.0
        
        enhanced_state = np.concatenate([
            current_state,
            action_history,
            reward_history
        ])
        
        return enhanced_state
    
    def update_history(self, action, reward):
        """更新历史记录"""
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def reset_history(self):
        """重置历史记录"""
        self.action_history = deque([0] * self.history_length, maxlen=self.history_length)
        self.reward_history = deque([0.0] * self.history_length, maxlen=self.history_length)
    
    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, q_param in zip(self.target_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)
    
    def act(self, state):
        """根据增强状态选择动作"""
        enhanced_state = self.get_history_state(state)
        
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        history_state = self.get_history_state(state)
        self.memory.push(state, action, reward, next_state, done, history_state)
    
    def replay(self):
        """经验回放学习"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones, history_states = batch
        
        # 修复：直接使用当前历史状态，不创建新agent实例
        next_history_states = []
        for i in range(len(next_states)):
            # 模拟历史更新：将当前动作和奖励添加到历史中
            temp_action_history = list(self.action_history)
            temp_reward_history = list(self.reward_history)
            
            # 添加新的动作和奖励
            temp_action_history.append(actions[i])
            temp_reward_history.append(rewards[i])
            
            # 保持历史长度
            if len(temp_action_history) > self.history_length:
                temp_action_history = temp_action_history[-self.history_length:]
                temp_reward_history = temp_reward_history[-self.history_length:]
            
            # 构建下一个历史状态
            action_hist = np.array(temp_action_history, dtype=np.float32)
            reward_hist = np.array(temp_reward_history, dtype=np.float32) / 10.0
            
            next_hist_state = np.concatenate([
                next_states[i],
                action_hist,
                reward_hist
            ])
            next_history_states.append(next_hist_state)
        
        # 转换为张量
        history_states = torch.FloatTensor(np.array(history_states)).to(self.device)
        next_history_states = torch.FloatTensor(np.array(next_history_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q = self.q_network(history_states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_history_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
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
            'epsilon': self.epsilon,
        }, filename)
        print(f"Model saved: {filename}")
    
    def load(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded: {filename}")

def train_enhanced_dqn():
    """训练增强版DQN代理"""
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    agent = HistoryEnhancedAgent(state_size, action_size, history_length=2)
    
    episodes = 400
    scores = []
    average_scores = []
    best_avg_score = 0
    
    for e in range(episodes):
        state, _ = env.reset()
        agent.reset_history()
        
        total_reward = 0
        steps = 0
        episode_actions = []
        
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            agent.update_history(action, reward)
            
            state = next_state
            total_reward += reward
            steps += 1
            episode_actions.append(action)
            
            # 经验回放
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            # 定期更新目标网络
            if steps % 4 == 0:
                agent.update_target_network()
            
            if done or steps >= 500:
                scores.append(steps)
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                average_scores.append(avg_score)
                
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                
                if e % 10 == 0:
                    action_balance = np.mean(episode_actions) if episode_actions else 0.5
                    print(f"Episode: {e+1:3d}/{episodes}, Score: {steps:3d}, "
                          f"Avg: {avg_score:6.1f}, Epsilon: {agent.epsilon:.3f}")
                
                if avg_score >= 195 and e >= 100:
                    print(f"Solved at episode {e+1}!")
                    agent.save('cartpole_enhanced_solved.pth')
                    env.close()
                    return agent, scores, average_scores
                
                break
    
    env.close()
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, label='Scores')
    if len(scores) >= 100:
        moving_avg = [np.mean(scores[max(0, i-99):i+1]) for i in range(len(scores))]
        plt.plot(moving_avg, 'r-', linewidth=2, label='Moving Avg (100)')
    plt.title('Enhanced DQN Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(average_scores)
    plt.title('Average Scores')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_dqn_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return agent, scores, average_scores

def test_agent(agent=None, episodes=5, render=False):
    """测试训练好的代理"""
    env = gym.make("CartPole-v1", render_mode="human" if render else "rgb_array")
    
    if agent is None:
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = HistoryEnhancedAgent(state_size, action_size)
        try:
            agent.load('cartpole_enhanced_solved.pth')
        except:
            print("No trained model found")
            return
    
    scores = []
    
    for e in range(episodes):
        state, _ = env.reset()
        agent.reset_history()
        total_reward = 0
        
        while True:
            if render:
                env.render()
            
            enhanced_state = agent.get_history_state(state)
            state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update_history(action, reward)
            state = next_state
            total_reward += reward
            
            if done:
                scores.append(total_reward)
                print(f"Test Episode {e+1}: Score = {total_reward}")
                break
    
    env.close()
    
    if scores:
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"Average test score: {avg_score:.1f} ± {std_score:.1f}")
    
    return scores

if __name__ == "__main__":
    print("Training Enhanced DQN Agent...")
    
    try:
        trained_agent, scores, avg_scores = train_enhanced_dqn()
        
        if trained_agent is not None:
            trained_agent.save('cartpole_enhanced_final.pth')
            print("Training completed successfully!")
            
            # 测试代理
            print("\nTesting trained agent...")
            test_scores = test_agent(trained_agent, episodes=3, render=True)
        else:
            print("Training did not reach the solved threshold.")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()