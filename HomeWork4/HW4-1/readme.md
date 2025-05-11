# HW4-1: Naive DQN for Static Mode

## 1. 方法概述

**Deep Q-Network (DQN)**  
- 用一個兩層全連接神經網路 \(Q(s,a;\theta)\) 來近似狀態—動作價值函數。  
- 每次更新時，計算 TD target：  
  \[
    y = r + \gamma \max_{a'} Q(s',a';\theta)
  \]  
- 最小化均方誤差  
  \[
    L = \bigl(Q(s,a;\theta) - y\bigr)^2
  \]

**Experience Replay**  
- 把每個 transition \((s,a,r,s',d)\) 存入環形緩衝區。  
- 訓練時從中隨機抽 mini-batch，打破時間相關性，提高樣本效率和訓練穩定性。

---

## 2. 實驗設定

**環境 (static mode)**  
- 固定 Player、Goal、Pit、Wall 的位置，不隨機化。  
- Board 大小：4×4

**網路架構**  
- Input: 4×4×4 one-hot 狀態展平為 64 維向量  
- 隱藏層：150 個 ReLU 神經元  
- Output: 4 個動作對應的 Q 值

**超參數**  
- Episodes: 1000  
- Batch size: 64  
- Replay 面板容量: 5000  
- 學習率: 1e-3 (Adam)  
- 折扣因子 \(\gamma\): 0.99  
- ε-greedy: ε 從 1.0 指數衰減至 0.05（衰減速率 500）

---

## 3. 結果觀察

**Loss 曲線**  
- 訓練初期 loss 高且波動，隨著抽樣和網路學習，loss 在數百步後快速下降並趨於平穩。

**Episode Reward 曲線**  
- 前期平均總回報偏低（常碰到 pit），約 −5 到 0 之間；  
- 約在 Episode 200–300 開始逐漸穩定上升，最後能持續以 +10（每集拿到 goal）作為最終回報。

> **圖表提示**：將 `results/hw4_1_training.png` 中的兩張子圖貼入並分別標示關鍵拐點。

---

## 4. 心得與討論

- **ε-decay 影響**：  
  - 較快衰減（小 decay 值）可更早利用已學到策略，但可能導致早期陷入次優解；  
  - 較慢衰減則探索更充分，但訓練時間更長且前期波動更大。

- **static mode 的易學性**：  
  - 因為環境不隨機，每一趟走法相對固定，agent 只要記住短路徑即可快速收斂；  
  - 可作為基礎測試，為後續 player/random 模式下的變體實驗打好基礎。

---

## 5. 小結

這份實現完整包含了 basic DQN 與 Experience Replay Buffer 的核心機制，並在簡單的 static gridworld 環境下成功收斂。後續可接著做 Double DQN、Dueling DQN 以及隨機化模式下的比較，以探究各種架構與隨機性的影響。

![image](https://github.com/user-attachments/assets/aebfc747-8091-4481-ba26-e349611fe313)

