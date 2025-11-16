import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class BoxWeightEstimator:
    def __init__(self, sequence_length=50, hidden_size=64):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_simulation_data(self, num_samples=5000, dt=0.01):
        """生成模拟的传感器数据"""
        print("生成模拟数据...")
        
        # 随机生成不同的箱子重量 (5kg 到 30kg)
        masses = np.random.uniform(5, 30, num_samples)
        
        # 存储特征和目标
        features = []
        targets = []
        
        for i, mass in enumerate(masses):
            if i % 500 == 0:
                print(f"生成进度: {i}/{num_samples}")
                
            # 模拟单次搬运过程
            time_steps = self.sequence_length
            t = np.arange(time_steps) * dt
            
            # 生成真实的物理过程
            F_hand, T_rope, y, v, a = self.simulate_lifting_process(mass, time_steps, dt)
            
            # 添加传感器噪声 (5%的噪声)
            noise_level = 0.05
            F_hand_noisy = F_hand * (1 + noise_level * np.random.randn(time_steps))
            T_rope_noisy = T_rope * (1 + noise_level * np.random.randn(time_steps))
            y_noisy = y * (1 + noise_level * 0.1 * np.random.randn(time_steps))  # 位置噪声较小
            v_noisy = v * (1 + noise_level * 0.5 * np.random.randn(time_steps))
            a_noisy = a * (1 + noise_level * 0.5 * np.random.randn(time_steps))
            
            # 组合特征序列 [F_hand, T_rope, y, v, a]
            sequence = np.column_stack([F_hand_noisy, T_rope_noisy, y_noisy, v_noisy, a_noisy])
            features.append(sequence)
            targets.append(mass)
        
        return np.array(features), np.array(targets)
    
    def simulate_lifting_process(self, mass, time_steps, dt):
        """模拟单次箱子搬运的物理过程"""
        g = 9.81
        weight = mass * g
        
        # 初始化状态
        y = np.zeros(time_steps)  # 位置
        v = np.zeros(time_steps)  # 速度
        a = np.zeros(time_steps)  # 加速度
        F_hand = np.zeros(time_steps)  # 手部压力
        T_rope = np.zeros(time_steps)  # 绳子拉力
        
        # 模拟搬运过程的不同阶段
        for t in range(1, time_steps):
            if t < time_steps // 5:  # 阶段1: 准备阶段，轻微用力
                F_hand_target = weight * 0.1
                T_rope_target = weight * 0.2
            elif t < 2 * time_steps // 5:  # 阶段2: 开始提升
                F_hand_target = weight * 0.15
                T_rope_target = weight * 0.7
            elif t < 3 * time_steps // 5:  # 阶段3: 稳定提升
                F_hand_target = weight * 0.1
                T_rope_target = weight * 0.8
            elif t < 4 * time_steps // 5:  # 阶段4: 减速准备停止
                F_hand_target = weight * 0.08
                T_rope_target = weight * 0.6
            else:  # 阶段5: 保持静止
                F_hand_target = weight * 0.05
                T_rope_target = weight * 0.5
            
            # 添加一些动态变化
            F_hand[t] = F_hand_target * (1 + 0.1 * np.sin(0.1 * t))
            T_rope[t] = T_rope_target * (1 + 0.05 * np.sin(0.08 * t))
            
            # 根据牛顿定律计算加速度
            total_force = F_hand[t] + T_rope[t] - weight
            a[t] = total_force / mass
            
            # 更新速度和位置
            v[t] = v[t-1] + a[t] * dt
            y[t] = y[t-1] + v[t] * dt
            
            # 添加物理约束
            if y[t] < 0:
                y[t] = 0
                v[t] = max(0, v[t])
        
        return F_hand, T_rope, y, v, a
    
    def create_lstm_model(self, input_size=5):
        """创建LSTM神经网络模型"""
        class WeightEstimationModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers=2):
                super(WeightEstimationModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=0.2)
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1),
                    nn.Softmax(dim=1)
                )
                self.regressor = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 4, 1)
                )
                
            def forward(self, x):
                # LSTM层
                lstm_out, (h_n, c_n) = self.lstm(x)
                
                # 注意力机制
                attention_weights = self.attention(lstm_out)
                context_vector = torch.sum(attention_weights * lstm_out, dim=1)
                
                # 回归预测
                output = self.regressor(context_vector)
                return output.squeeze()
        
        return WeightEstimationModel(input_size, self.hidden_size)
    
    def train_model(self, features, targets, test_size=0.2, batch_size=32, epochs=100):
        """训练神经网络模型"""
        print("准备训练数据...")
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=42
        )
        
        # 标准化特征
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_2d)
        X_train_scaled = self.scaler.transform(X_train_2d).reshape(X_train.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        self.model = self.create_lstm_model(input_size=features.shape[-1])
        self.model.to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print("开始训练模型...")
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                test_predictions = self.model(X_test_tensor)
                test_loss = criterion(test_predictions, y_test_tensor)
                test_losses.append(test_loss.item())
            
            train_losses.append(epoch_train_loss / len(train_loader))
            scheduler.step(test_loss)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}/{epochs}, Train Loss: {train_losses[-1]:.4f}, '
                      f'Test Loss: {test_losses[-1]:.4f}')
        
        # 绘制训练损失
        self.plot_training_curve(train_losses, test_losses)
        
        return train_losses, test_losses
    
    def plot_training_curve(self, train_losses, test_losses):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Model Training Progress')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.show()
    
    def evaluate_model(self, features, targets):
        """评估模型性能"""
        if not hasattr(self, 'model'):
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 预处理数据
        features_2d = features.reshape(-1, features.shape[-1])
        features_scaled = self.scaler.transform(features_2d).reshape(features.shape)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features_tensor).cpu().numpy()
        
        # 计算评估指标
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        
        print(f"模型评估结果:")
        print(f"MAE: {mae:.3f} kg")
        print(f"RMSE: {rmse:.3f} kg")
        print(f"R² Score: {r2:.3f}")
        
        # 绘制预测vs真实值散点图
        self.plot_predictions_vs_actual(targets, predictions)
        
        return predictions, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def plot_predictions_vs_actual(self, actual, predicted):
        """绘制预测值与真实值的对比图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.6)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Weight (kg)')
        plt.ylabel('Predicted Weight (kg)')
        plt.title('Predicted vs Actual Box Weights')
        plt.grid(True)
        
        # 添加误差统计
        error = predicted - actual
        plt.text(0.05, 0.95, f'MAE: {np.mean(np.abs(error)):.2f} kg\n'
                            f'RMSE: {np.sqrt(np.mean(error**2)):.2f} kg', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        plt.show()
    
    def predict_single_sequence(self, sensor_data):
        """预测单个传感器序列的重量"""
        if not hasattr(self, 'model'):
            raise ValueError("模型尚未训练")
        
        # 预处理输入数据
        sensor_data_scaled = self.scaler.transform(sensor_data.reshape(-1, sensor_data.shape[-1]))
        sensor_data_scaled = sensor_data_scaled.reshape(1, *sensor_data.shape)
        sensor_tensor = torch.FloatTensor(sensor_data_scaled).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sensor_tensor).cpu().item()
        
        return prediction

# 使用示例
def main():
    # 初始化重量估计器
    estimator = BoxWeightEstimator(sequence_length=50, hidden_size=64)
    
    # 生成模拟数据
    features, targets = estimator.generate_simulation_data(num_samples=2000)
    print(f"数据形状: 特征 {features.shape}, 目标 {targets.shape}")
    
    # 训练模型
    train_losses, test_losses = estimator.train_model(features, targets, epochs=100)
    
    # 评估模型
    predictions, metrics = estimator.evaluate_model(features, targets)
    
    # 测试单个序列预测
    test_sequence = features[0]  # 取第一个样本测试
    predicted_weight = estimator.predict_single_sequence(test_sequence)
    actual_weight = targets[0]
    print(f"\n单样本测试:")
    print(f"真实重量: {actual_weight:.2f} kg")
    print(f"预测重量: {predicted_weight:.2f} kg")
    print(f"误差: {abs(predicted_weight - actual_weight):.2f} kg")

if __name__ == "__main__":
    main()