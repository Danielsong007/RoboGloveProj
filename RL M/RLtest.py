import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gym
from gym import spaces
import threading
import time

# ==================== 系统配置 ====================
class Config:
    MAX_FORCE = 100.0  # 最大安全拉力(N)
    TARGET_FORCE = 10.0  # 目标辅助拉力(N)
    MAX_TORQUE = 5.0  # 电机最大扭矩(Nm)
    CONTROL_FREQ = 100  # 控制频率(Hz)
    SAFE_LIMIT = 0.9 * MAX_FORCE  # 安全阈值
    PRESSURE_THRESH = 0.05  # 意图识别阈值(kPa)
    HISTORY_WINDOW = 5  # 历史数据窗口大小
    DT = 1.0 / CONTROL_FREQ  # 控制周期(s)

# ==================== 增强型环境 ====================
class RopeLiftEnv(gym.Env):
    def __init__(self):
        super(RopeLiftEnv, self).__init__()
        
        # 状态空间维度计算
        self.state_dim = 6 + 3 * Config.HISTORY_WINDOW  # 基础6维 + (力/压力/动作各HISTORY_WINDOW步)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        
        # 物理参数
        self.mass = 1.0  # 负载质量(kg)
        self.damping = 0.2  # 阻尼系数(N·s/m)
        
        self.reset()

    def reset(self):
        """重置环境状态"""
        self.current_force = np.random.uniform(5, 15)
        self.pressure = np.random.uniform(0.5, 1.5)
        self.position = 0.0
        self.velocity = 0.0
        self.force_integral = 0.0
        self.pressure_integral = 0.0
        
        # 初始化历史数据缓冲区
        self.force_history = deque([self.current_force] * Config.HISTORY_WINDOW, 
                                 maxlen=Config.HISTORY_WINDOW)
        self.pressure_history = deque([self.pressure] * Config.HISTORY_WINDOW,
                                    maxlen=Config.HISTORY_WINDOW)
        self.action_history = deque([0.0] * Config.HISTORY_WINDOW,
                                  maxlen=Config.HISTORY_WINDOW)
        
        return self._get_state()

    def step(self, action):
        """执行动作并返回新状态"""
        # 1. 转换动作到实际扭矩
        torque = np.clip(action, -1, 1)[0] * Config.MAX_TORQUE
        
        # 2. 更新系统动力学
        self._update_dynamics(torque)
        
        # 3. 更新历史数据
        self.force_history.append(self.current_force)
        self.pressure_history.append(self.pressure)
        self.action_history.append(action[0])
        
        # 4. 计算奖励
        reward = self._calculate_reward(action)
        
        # 5. 检查终止条件
        done = self.current_force > Config.SAFE_LIMIT
        
        return self._get_state(), reward, done, {}

    def _update_dynamics(self, torque):
        """更新物理模型"""
        # 模拟传感器噪声
        self.pressure += np.random.normal(0, 0.02)
        self.pressure = np.clip(self.pressure, 0, 2)
        
        # 动力学方程: F = ma + bv
        net_force = torque - self.damping * self.velocity
        acceleration = net_force / self.mass
        
        # 更新状态 (欧拉积分)
        self.velocity += acceleration * Config.DT
        self.position += self.velocity * Config.DT
        self.current_force = torque  # 简化假设
        
        # 边界检查
        self.current_force = np.clip(self.current_force, 0, Config.MAX_FORCE)
        self.velocity = np.clip(self.velocity, -1, 1)
        
        # 更新积分项
        self.force_integral += (self.current_force - Config.TARGET_FORCE) * Config.DT
        self.pressure_integral += (self.pressure - 1.0) * Config.DT

    # 在RopeLiftEnv类中替换_get_state方法
    def _get_state(self):
        """安全的状态构建方法"""
        # 1. 基础特征
        state = np.zeros(6 + 3*Config.HISTORY_WINDOW, dtype=np.float32)
        
        # 当前状态
        state[0] = self.current_force / Config.MAX_FORCE
        state[1] = (self.pressure - self.pressure_history[-2])/0.2 if len(self.pressure_history)>=2 else 0.0
        state[2] = self.position / 2.0
        state[3] = self.velocity / 1.0
        state[4] = np.clip(self.force_integral / 10.0, -1, 1)
        state[5] = np.clip(self.pressure_integral / 5.0, -1, 1)
        
        # 2. 历史特征
        hist_start = 6
        for i, f in enumerate(self.force_history):
            state[hist_start+i] = (f - self.current_force)/Config.MAX_FORCE
        for i, p in enumerate(self.pressure_history):
            state[hist_start+Config.HISTORY_WINDOW+i] = (p - self.pressure)/0.2
        for i, a in enumerate(self.action_history):
            state[hist_start+2*Config.HISTORY_WINDOW+i] = a
            
        return state

    def _calculate_reward(self, action):
        """复合奖励函数"""
        # 力跟踪误差
        force_error = -0.5 * abs(self.current_force - Config.TARGET_FORCE)
        
        # 意图匹配
        intent_match = 1.0 if self._check_intent(action) else -0.2
        
        # 动作平滑
        smoothness = -0.1 * abs(action[0] - self.action_history[-1])
        
        # 安全惩罚
        safety_penalty = -10 if self.current_force > Config.SAFE_LIMIT else 0
        
        return force_error + intent_match + smoothness + safety_penalty

    def _check_intent(self, action):
        """改进的意图检测"""
        if len(self.pressure_history) < 2:
            return False
            
        # 计算压力梯度
        grad_p = self.pressure - self.pressure_history[-2]
        
        # 动态阈值
        dynamic_thresh = Config.PRESSURE_THRESH * (1 + 0.5 * abs(action[0]))
        
        # 复合判断条件
        if (grad_p > dynamic_thresh and action[0] > 0) or \
           (grad_p < -dynamic_thresh and action[0] < 0):
            return True
        return False

# ==================== PPO网络 ====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        
        # 共享特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 策略网络
        self.actor_mean = nn.Linear(64, 1)
        self.actor_logstd = nn.Parameter(torch.zeros(1))
        
        # 价值网络
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        # 维度检查
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"Input shape mismatch. Expected {self.state_dim}, got {x.shape[-1]}")
            
        features = self.feature_net(x)
        mean = torch.tanh(self.actor_mean(features))
        std = torch.exp(self.actor_logstd).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.critic(features)
        return dist, value

# ==================== 训练工具函数 ====================
def compute_returns(rewards, gamma=0.99):
    """改进的折扣回报计算"""
    returns = np.zeros(len(rewards), dtype=np.float32)
    R = 0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R
        returns[i] = R
    return torch.from_numpy(returns)  # 直接从numpy数组创建


def train_ppo():
    env = RopeLiftEnv()
    policy = ActorCritic(env.state_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    # 训练参数
    batch_size = 128
    gamma = 0.99
    clip_param = 0.2
    entropy_coef = 0.01
    epochs = 4
    
    # 训练统计
    reward_history = []
    force_history = []
    episode_lengths = []
    
    print("\n===== Training Started =====")
    print(f"State Dimension: {env.state_dim}")
    print(f"Network Architecture: {policy}")
    print("="*40)
    
    for episode in range(100):
        state = env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        episode_reward = 0
        step_count = 0
        
        # 收集轨迹
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                dist, value = policy(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # 安全覆盖
            if env.current_force > Config.SAFE_LIMIT * 0.8 and action > 0:
                action = torch.clamp(action, -1, 0)
            
            # 环境交互
            next_state, reward, done, _ = env.step(action.numpy())
            
            # 存储轨迹
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if done or step_count >= 200:  # 防止无限循环
                break
        
        # 记录统计信息
        reward_history.append(episode_reward)
        force_history.append(np.mean(env.force_history))
        episode_lengths.append(step_count)
        
        # PPO更新
        states_tensor = torch.FloatTensor(np.array(states, dtype=np.float32))
        actions_tensor = torch.cat(actions).view(-1)  # 确保是1D张量

        old_log_probs = torch.cat(log_probs).detach()
        returns = compute_returns(rewards, gamma)
        
        # 训练统计
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(epochs):
            dist, values = policy(states_tensor)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            advantages = returns - values.squeeze().detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip_param, 1+clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(values.squeeze(-1), returns)  # 确保形状匹配

            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            # 记录训练指标
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

        # 打印训练进度
        if episode % 10 == 0:
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            avg_entropy = np.mean(entropies)
            # 确保所有格式化值都是标量
            avg_reward = float(np.mean(reward_history[-10:])) if reward_history else 0.0
            avg_force = float(np.mean(env.force_history)) if env.force_history else 0.0
            
            print(f"Ep {episode:4d} | "
                f"Reward: {float(episode_reward):7.1f} (Avg: {avg_reward:5.1f}) | "
                f"Length: {step_count:3d} | "
                f"Force: {avg_force:5.1f}N | "
                f"Loss: P {float(avg_policy_loss):6.3f}, V {float(avg_value_loss):6.3f} | "
                f"Entropy: {float(avg_entropy):5.3f}")

    
        
        # 每100episode打印详细报告
        if episode % 100 == 99:
            print("\n" + "="*60)
            print(f"Training Report @ Episode {episode+1}")
            print(f"- Last 100 Episodes Avg Reward: {np.mean(reward_history[-100:]):.1f}")
            print(f"- Avg Episode Length: {np.mean(episode_lengths[-100:]):.1f} steps")
            print(f"- Avg Force: {np.mean(force_history[-100:]):.1f}N")
            print(f"- Recent Losses: Policy {np.mean(policy_losses[-100:]):.3f}, "
                  f"Value {np.mean(value_losses[-100:]):.3f}")
            print("="*60 + "\n")
    
    # 训练结束保存模型
    torch.save(policy, "ppo_rope_lift.pth")
    print("\n===== Training Completed =====")
    print(f"Final Model Saved to: ppo_rope_lift.pth")

# ==================== 实时控制器 ====================
class RealTimeController:
    def __init__(self, policy_path):
        self.policy = torch.load(policy_path)
        self.policy.eval()
        
        # 数据缓冲区
        self.sensor_buffer = deque(maxlen=Config.HISTORY_WINDOW)
        self.action_history = deque([0.0]*Config.HISTORY_WINDOW, 
                                  maxlen=Config.HISTORY_WINDOW)
        self.state = None
        self.last_torque = 0.0
        
        # 安全监控
        self._safety_flag = False
        self._init_safety_monitor()

    def _init_safety_monitor(self):
        """启动安全监控线程"""
        self.safety_thread = threading.Thread(target=self._safety_check)
        self.safety_thread.daemon = True
        self.safety_thread.start()

    def update_sensors(self, force, pressure):
        """更新传感器数据并构建完整状态向量"""
        self.sensor_buffer.append({
            'force': force,
            'pressure': pressure,
            'timestamp': time.time()
        })
        
        # 至少需要2个数据点计算梯度
        if len(self.sensor_buffer) >= 2:
            current = self.sensor_buffer[-1]
            prev = self.sensor_buffer[-2]
            dt = max(current['timestamp'] - prev['timestamp'], 1e-5)
            
            # 基础特征 (6维)
            state = np.zeros(6 + 3*Config.HISTORY_WINDOW, dtype=np.float32)
            state[0] = current['force'] / Config.MAX_FORCE
            state[1] = (current['pressure'] - prev['pressure']) / (0.2 * dt)
            state[2] = 0.0  # 位置 (需实际传感器)
            state[3] = 0.0  # 速度 (需实际传感器)
            state[4] = 0.0  # 拉力积分
            state[5] = 0.0  # 压力积分
            
            # 历史特征 (15维)
            if len(self.sensor_buffer) >= Config.HISTORY_WINDOW:
                hist_forces = [f['force'] for f in self.sensor_buffer]
                hist_pressures = [p['pressure'] for p in self.sensor_buffer]
                
                # 力历史 (5维)
                for i in range(Config.HISTORY_WINDOW):
                    state[6+i] = (hist_forces[i] - current['force'])/Config.MAX_FORCE
                
                # 压力历史 (5维)
                for i in range(Config.HISTORY_WINDOW):
                    state[6+Config.HISTORY_WINDOW+i] = (hist_pressures[i] - current['pressure'])/0.2
                
                # 动作历史 (5维) - 初始化为0
                for i in range(Config.HISTORY_WINDOW):
                    state[6+2*Config.HISTORY_WINDOW+i] = 0.0
            
            self.state = state


    def get_action(self):
        """生成控制指令"""
        if self.state is None or self._safety_flag:
            return 0.0
        
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
        
        with torch.no_grad():
            dist, _ = self.policy(state_tensor)
            action = dist.mean.item()
            
        # 安全限制
        if self.sensor_buffer and self.sensor_buffer[-1]['force'] > Config.SAFE_LIMIT * 0.8:
            action = min(action, 0.0)
        
        # 低通滤波
        torque = action * Config.MAX_TORQUE
        self.last_torque = 0.8 * torque + 0.2 * self.last_torque
        
        # 更新动作历史
        self.action_history.append(action)
        self._update_state_with_actions()  # 更新状态中的动作历史
        
        return self.last_torque

    
    def _update_state_with_actions(self):
        """更新状态中的动作历史部分"""
        if self.state is not None and len(self.action_history) >= Config.HISTORY_WINDOW:
            start_idx = 6 + 2*Config.HISTORY_WINDOW
            for i in range(Config.HISTORY_WINDOW):
                self.state[start_idx + i] = self.action_history[i]


    def _safety_check(self):
        """安全监控循环"""
        while True:
            if self.sensor_buffer and self.sensor_buffer[-1]['force'] > Config.SAFE_LIMIT:
                self._safety_flag = True
                self._emergency_stop()
            time.sleep(0.001)

    def _emergency_stop(self):
        """紧急停止协议"""
        print("[Safety] Emergency stop triggered!")
        self.last_torque = 0.0
        # 实际硬件中应调用: motor_controller.emergency_stop()

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=== Rope Lift Control System ===")
    
    # 训练配置
    train_config = {
        'lr': 1e-4,
        'batch_size': 64,
        'gamma': 0.95,
        'epochs': 3
    }
    
    # 训练模型
    print("1. Training PPO policy...")
    train_ppo()
    
    # 测试控制器
    print("\n2. Testing real-time control...")
    controller = RealTimeController("ppo_rope_lift.pth")
    
    # 模拟运行 (带随机扰动)
    for i in range(100):
        # 模拟传感器输入 (添加噪声)
        sim_force = np.random.normal(15, 2)
        sim_pressure = np.random.normal(1.0, 0.1)
        
        # 更新控制
        controller.update_sensors(
            force=np.clip(sim_force, 5, Config.MAX_FORCE),
            pressure=np.clip(sim_pressure, 0.5, 1.5)
        )
        torque = controller.get_action()
        
        print(f"Step {i:3d} | "
              f"Force: {sim_force:5.1f}N | "
              f"Pressure: {sim_pressure:4.1f}kPa | "
              f"Torque: {torque:5.2f}Nm")
        
        time.sleep(Config.DT)