# HW4-3: Enhance DQN for Random Mode with Training Tips

## 1. 方法概述

**Deep Q-Network (DQN) 轉換至 Keras**  
- 使用 Keras 搭建兩層全連接網路近似 Q(s,a;θ)。  
- 一步一步實現原始 DQN：
  - **Experience Replay**：環形緩衝區儲存 (s,a,r,s',d)，隨機抽取 mini-batch 更新，打破時間相關性。  
  - **Target Network**：定期同步主網路權重，穩定 target 計算。

**訓練強化技巧**  
- **梯度裁剪 (Gradient Clipping)**：設定 `clipnorm=1.0`，避免梯度爆炸。  
- **學習率調度 (LR Scheduling)**：採用指數衰減 (ExponentialDecay)，自動降低學習率。  
- **Batch Normalization / Dropout**（如有需要，可自行擴充）：增強訓練穩定性與收斂速度。

---

## 2. 實驗設定

**環境 (random mode)**  
- Player、Goal、Pit、Wall 四種物件位置皆隨機生成。  
- Board 大小：4×4

**網路架構 (Keras)**  
- Input: 4×4×4 one-hot 展平為 64 維  
- 隱藏層：150 個 ReLU 全連接神經元  
- Output: 4 維 Q(s,a) 預測值

**超參數**  
- Episodes: 1000  
- Batch size: 64  
- Replay Buffer 容量: 5000  
- 初始學習率: 1e-3  
- 學習率調度: ExponentialDecay(decay_steps=1000, decay_rate=0.96)  
- 梯度裁剪: clipnorm=1.0  
- 折扣因子 γ: 0.99  
- ε-greedy: ε 從 1.0 指數衰減至 0.05（decay_rate 對應 Episodes）  
- 目標網路同步: 每 50 個訓練步驟

---

## 3. 結果觀察

**Loss 曲線**  
- 初期 loss 大且波動，隨著 buffer 積累和目標網路穩定化，loss 在數百步後快速下降並趨於穩定。

**Episode Reward 曲線**  
- 隨機模式下回報因隨機起點而更波動；
- 經過約 300–400 集後，平均總回報逐漸提升，最終穩定在約 +5 至 +7 範圍；
- 與未使用訓練技巧比較，梯度裁剪與 LR 調度顯著降低了訓練抖動幅度。

![image](https://github.com/user-attachments/assets/77a716fd-a8b0-4ab0-bb22-db79b71a75c2)


---

## 4. 心得與討論

- **梯度裁剪**：有效避免了梯度爆炸，令訓練更穩定，特別是在 early episodes 中。  
- **學習率調度**：使學習率隨訓練逐步降低，提升後期微調能力，加快收斂。  
- **Random mode 挑戰**：隨機起點增加了學習難度，agent 需要學會通用策略而非記憶固定路徑。

---

## 5. 小結

本實驗將原始 PyTorch DQN 成功轉換為 Keras 實現，並在 random mode 中：  
- 集成梯度裁剪與學習率調度等訓練技巧，顯著提高了訓練穩定性與收斂速度；  
- 實現了 static/player/random 三種模式的完整擴展。  
後續可引入 Double/Dueling 架構或優化 replay 策略，進一步提升隨機環境中的表現。
