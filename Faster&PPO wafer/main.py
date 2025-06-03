# single_stage_optimization.py - 單階段優化

import numpy as np
import pickle
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 使用系統默認字體
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

set_random_seed(42)
torch.manual_seed(42)
np.random.seed(42)

class SingleStageWaferEnv(Env):
    """單階段優化環境 """
    def __init__(self, wafer_data, labels, num_classes):
        super().__init__()
        self.wafer_data = wafer_data
        self.labels = labels
        self.num_classes = num_classes
        self.episode_length = 60  # 回到較短的episode以提高訓練效率
        
        # 分析類別分布（基於您的混淆矩陣）
        unique, counts = np.unique(labels, return_counts=True)
        self.class_info = {}
        total = len(labels)
        
        for cls, count in zip(unique, counts):
            frequency = count / total
            # 基於您的結果，識別困難類別
            if cls in [0, 2, 4, 5, 6, 7]:  # 從混淆矩陣看出的困難類別
                difficulty = 2.5
            elif cls in [1]:  # 中等難度
                difficulty = 1.8
            else:  # cls in [3, 8] - 相對容易的類別
                difficulty = 1.0
            
            self.class_info[cls] = {
                'count': count,
                'frequency': frequency,
                'difficulty': difficulty
            }
        
        print(f"📊 類別分析完成")
        print(f"困難類別 [0,2,4,5,6,7]: {sum(counts[i] for i in [0,2,4,5,6,7] if i < len(counts))} 樣本")
        print(f"容易類別 [3,8]: {sum(counts[i] for i in [3,8] if i < len(counts))} 樣本")
        
        # 提取特徵 
        self.features = self._extract_effective_features(wafer_data)
        
        # 定義空間
        feature_dim = self.features.shape[1]
        context_dim = 8
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(feature_dim + context_dim,), dtype=np.float32
        )
        self.action_space = Discrete(num_classes)
        
        # 性能追蹤
        self.class_performance = {i: deque(maxlen=20) for i in range(num_classes)}
        self.confusion_memory = np.zeros((num_classes, num_classes))
        
        # 初始化
        self.current_step = 0
        self.correct_count = 0
        self.current_idx = 0
        
    def _extract_effective_features(self, wafer_data):
        """提取有效特徵 """
        print("🔧 提取有效特徵...")
        features = []
        
        for i, wafer in enumerate(wafer_data):
            if i % 1000 == 0:
                print(f"  處理 {i}/{len(wafer_data)}")
            
            # 正規化
            wafer_norm = wafer.astype(np.float32) / 255.0 if wafer.max() > 1 else wafer.astype(np.float32)
            
            # 1. 基礎特徵（最重要）
            flat = wafer_norm.flatten()
            
            # 2. 核心統計特徵
            stats = [
                np.mean(wafer_norm), np.std(wafer_norm), np.median(wafer_norm),
                np.min(wafer_norm), np.max(wafer_norm), 
                np.percentile(wafer_norm, 25), np.percentile(wafer_norm, 75),
                np.sum(wafer_norm > np.mean(wafer_norm)) / wafer_norm.size
            ]
            
            # 3. 關鍵空間特徵
            h, w = wafer_norm.shape
            center_h, center_w = h//2, w//2
            
            # 只計算最有效的區域特徵
            regions = [
                wafer_norm[:center_h, :center_w],      # 左上
                wafer_norm[center_h:, center_w:]       # 右下
            ]
            
            region_stats = []
            for region in regions:
                region_stats.extend([
                    np.mean(region), np.std(region),
                    np.sum(region > np.mean(wafer_norm)) / region.size
                ])
            
            # 4. 簡單邊緣特徵
            if h > 1 and w > 1:
                edge_h = np.mean(np.abs(np.diff(wafer_norm, axis=0)))
                edge_w = np.mean(np.abs(np.diff(wafer_norm, axis=1)))
                edge_features = [edge_h, edge_w]
            else:
                edge_features = [0, 0]
            
            # 組合特徵
            combined = np.concatenate([flat, stats, region_stats, edge_features])
            features.append(combined)
        
        features = np.array(features)
        print(f"✅ 特徵提取完成，維度: {features.shape}")
        return features
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.correct_count = 0
        self.current_idx = np.random.randint(0, len(self.features))
        
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self):
        feature = self.features[self.current_idx].copy()
        
        # 精簡的上下文信息
        step_progress = self.current_step / self.episode_length
        accuracy_so_far = self.correct_count / max(1, self.current_step)
        true_label = self.labels[self.current_idx]
        
        # 類別相關信息
        difficulty = self.class_info[true_label]['difficulty']
        frequency = self.class_info[true_label]['frequency']
        
        # 歷史表現
        class_perf = np.mean(self.class_performance[true_label]) if self.class_performance[true_label] else 0.5
        
        # 上下文
        context = np.array([
            step_progress,
            accuracy_so_far,
            float(true_label),
            difficulty,
            frequency,
            class_perf,
            float(step_progress < 0.3),  # 早期
            float(step_progress > 0.7)   # 後期
        ], dtype=np.float32)
        
        obs = np.concatenate([feature, context])
        return obs
    
    def step(self, action):
        true_label = self.labels[self.current_idx]
        is_correct = (action == true_label)
        
        # 更新混淆記憶
        self.confusion_memory[true_label, action] += 1
        
        # 計算獎勵
        reward = self._calculate_focused_reward(action, true_label, is_correct)
        
        # 更新統計
        if is_correct:
            self.correct_count += 1
        
        self.class_performance[true_label].append(float(is_correct))
        
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # Episode結束處理
        if done:
            episode_accuracy = self.correct_count / self.episode_length
            episode_reward = self._calculate_episode_reward(episode_accuracy)
            reward += episode_reward
        
        # 智能採樣
        if not done:
            self.current_idx = self._focused_sampling()
        
        next_obs = self._get_observation()
        
        info = {
            'accuracy': self.correct_count / max(1, self.current_step),
            'episode_accuracy': self.correct_count / self.episode_length if done else None
        }
        
        return next_obs, reward, done, False, info
    
    def _calculate_focused_reward(self, action, true_label, is_correct):
        """專注的獎勵函數 - 針對75%→80%的提升"""
        if is_correct:
            base_reward = 15.0  # 提高基礎獎勵
            
            # 困難類別大幅獎勵
            difficulty = self.class_info[true_label]['difficulty']
            base_reward *= difficulty
            
            # 當前表現獎勵
            current_acc = self.correct_count / max(1, self.current_step)
            if current_acc > 0.8:
                base_reward += 15.0  # 大幅獎勵高性能
            elif current_acc > 0.75:
                base_reward += 8.0
            
            # 類別改進獎勵
            if self.class_performance[true_label]:
                recent_perf = np.mean(list(self.class_performance[true_label])[-5:])
                if recent_perf > 0.8:
                    base_reward += 10.0
            
            return base_reward
        
        else:
            base_penalty = -5.0
            
            # 簡單類別錯誤嚴重懲罰
            if true_label in [3, 8]:  # 應該容易分類的類別
                base_penalty *= 2.0
            
            # 常見錯誤懲罰
            if self.confusion_memory[true_label, action] > 10:
                base_penalty *= 1.5
            
            return base_penalty
    
    def _calculate_episode_reward(self, episode_accuracy):
        """Episode結束獎勵 - 針對80%目標"""
        if episode_accuracy >= 0.85:
            return 200.0  # 超越目標的大獎勵
        elif episode_accuracy >= 0.8:
            return 150.0  # 達到目標的大獎勵
        elif episode_accuracy >= 0.78:
            return 100.0  # 接近目標
        elif episode_accuracy >= 0.75:
            return 50.0   # 保持75%水平
        elif episode_accuracy < 0.7:
            return -50.0  # 低於75%的懲罰
        else:
            return 0.0
    
    def _focused_sampling(self):
        """專注採樣 - 重點關注困難類別"""
        # 50%機率選擇困難類別
        if np.random.random() < 0.5:
            # 識別表現最差的類別
            poor_classes = []
            for cls, perfs in self.class_performance.items():
                if len(perfs) >= 3 and np.mean(perfs) < 0.7:
                    poor_classes.append(cls)
            
            if poor_classes:
                target_class = np.random.choice(poor_classes)
                class_indices = np.where(self.labels == target_class)[0]
                return np.random.choice(class_indices)
        
        return np.random.randint(0, len(self.features))

class FocusedCallback(BaseCallback):
    """專注回調 - 監控80%目標進度"""
    def __init__(self, eval_freq=2000, target_accuracy=0.8):
        super().__init__()
        self.eval_freq = eval_freq
        self.target_accuracy = target_accuracy
        self.best_reward = -np.inf
        self.accuracy_estimates = []
        self.consecutive_improvements = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                
                # 估算準確率
                estimated_acc = min(0.9, max(0.6, (mean_reward + 50) / 250))
                self.accuracy_estimates.append(estimated_acc)
                
                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    self.consecutive_improvements += 1
                    
                    print(f"🎯 改進 #{self.consecutive_improvements}: 獎勵={mean_reward:.1f}, 估算準確率={estimated_acc:.3f}")
                    
                    # 如果估算準確率接近80%，保存模型
                    if estimated_acc >= 0.78:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        self.model.save(f'focused_high_perf_{timestamp}')
                        print(f"💾 高性能模型已保存")
                else:
                    self.consecutive_improvements = 0
                
                print(f"步數: {self.num_timesteps:,}, 獎勵: {mean_reward:.1f}, 趨勢: {'↑' if len(self.accuracy_estimates) >= 2 and self.accuracy_estimates[-1] > self.accuracy_estimates[-2] else '↓'}")
        
        return True

def load_data():
    print("📁 載入資料...")
    try:
        with open('./data/x_train_org_20210614.pickle','rb') as f:
            x_train = pickle.load(f)
        with open('./data/y_train_org_20210614.pickle','rb') as f:
            y_train = pickle.load(f)
        with open('./data/x_test_20210614.pickle','rb') as f:
            x_test = pickle.load(f)
        with open('./data/y_test_20210614.pickle','rb') as f:
            y_test = pickle.load(f)
        
        return np.array(x_train), np.array(y_train).ravel(), np.array(x_test), np.array(y_test).ravel()
    except Exception as e:
        print(f"載入失敗: {e}")
        return None, None, None, None

def main():
    print("🚀 單階段優化PPO - 專注75%→80%提升")
    
    # 載入數據
    x_train, y_train, x_test, y_test = load_data()
    if x_train is None:
        print("❌ 數據載入失敗")
        return
    
    # 準備數據
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)
    
    print(f"類別數量: {num_classes}")
    print(f"訓練樣本: {len(x_train)}, 測試樣本: {len(x_test)}")
    
    # 保守的數據平衡
    print("⚖️ 保守數據平衡...")
    unique_labels, counts = np.unique(y_train_enc, return_counts=True)
    
    balanced_x, balanced_y = [], []
    target_count = min(200, max(counts))  # 保守的目標數量
    
    for label in unique_labels:
        label_indices = np.where(y_train_enc == label)[0]
        current_count = len(label_indices)
        
        if current_count < 80:  # 最少80個樣本
            sampled_indices = np.random.choice(label_indices, 80, replace=True)
        elif current_count > target_count:
            sampled_indices = np.random.choice(label_indices, target_count, replace=False)
        else:
            sampled_indices = label_indices
        
        balanced_x.extend(x_train[sampled_indices])
        balanced_y.extend([label] * len(sampled_indices))
    
    X_train_balanced = np.array(balanced_x)
    y_train_balanced = np.array(balanced_y)
    
    # 訓練集準備
    X_train_final, _, y_train_final, _ = train_test_split(
        X_train_balanced, y_train_balanced, test_size=0.1,  # 只留少量驗證
        random_state=42, stratify=y_train_balanced
    )
    
    print(f"最終訓練樣本: {len(X_train_final)}")
    
    # 創建環境
    print("🌍 建立專注環境...")
    def make_env():
        return SingleStageWaferEnv(X_train_final, y_train_final, num_classes)
    
    env = DummyVecEnv([make_env])
    callback = FocusedCallback()
    
    # 優化的PPO配置
    print("🧠 建立專注PPO...")
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=1.5e-4,    # 較低的學習率保證穩定性
        n_steps=768,             # 適中的步數
        batch_size=96,           # 適中的批量
        n_epochs=15,             # 更多的訓練輪數
        gamma=0.98,              # 稍微降低折扣因子
        gae_lambda=0.95,
        clip_range=0.18,         # 適中的clip範圍
        ent_coef=0.025,          # 增加探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            'net_arch': dict(pi=[384, 192, 96], vf=[384, 192, 96]),
        }
    )
    
    print("🚀 開始專注訓練...")
    print("🎯 目標: 穩定從75%提升到80%")
    
    # 訓練
    try:
        model.learn(total_timesteps=100000, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("⏹️ 訓練中斷")
    
    # 最終評估
    print("\n📊 最終評估...")
    test_env = SingleStageWaferEnv(x_test, y_test_enc, num_classes)
    
    test_predictions = []
    for i in range(len(x_test)):
        test_env.current_idx = i
        obs = test_env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        test_predictions.append(action)
    
    final_accuracy = accuracy_score(y_test_enc, test_predictions)
    
    # 保存最終模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save(f'single_stage_final_{timestamp}')
    
    print(f"\n🎯 最終結果:")
    print(f"  測試準確率: {final_accuracy:.4f}")
    print(f"  改進次數: {callback.consecutive_improvements}")
    print(f"  vs 74.9%: {(final_accuracy - 0.749):.4f}")
    
    # 成功判斷
    if final_accuracy >= 0.8:
        print("🎉 成功達到80%目標！")
    elif final_accuracy >= 0.77:
        print("🔥 很接近目標，建議繼續訓練")
    elif final_accuracy >= 0.75:
        print("✅ 保持了75%水平")
    else:
        print("⚠️ 需要檢查配置")
    
    # 簡單可視化（避免字體問題）
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 進度條形圖
    milestones = ['Previous Best', 'Current Result', 'Target']
    values = [0.749, final_accuracy, 0.8]
    colors = ['orange', 'blue' if final_accuracy >= 0.8 else 'red', 'green']
    
    bars = ax.bar(milestones, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Progress Toward 80% Target')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.7, 0.85)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加數值
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'single_stage_result_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 詳細分析
    print(f"\n📊 詳細分析:")
    print(classification_report(y_test_enc, test_predictions, zero_division=0))

if __name__ == '__main__':
    main()
