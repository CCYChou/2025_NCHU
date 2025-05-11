# HW4-2: Double DQN & Dueling DQN for Player Mode

## 1. 方法概述

**Double DQN**  
- 使用 **online** 和 **target** 兩個 Q 網路。  
- Target 計算：  
  \[
    y = r + \gamma \; Q_{\text{target}}\bigl(s', \arg\max_a Q_{\text{online}}(s',a)\bigr)
  \]  
- 減少過度估計偏差（overestimation bias），並每隔固定步數同步一次權重。

**Dueling DQN**  
- 將 Q 網路拆成兩條支路：**Value** \(V(s)\) 與 **Advantage** \(A(s,a)\)。  
- 最終計算：  
  \[
    Q(s,a) = V(s) + \Bigl(A(s,a) - \tfrac{1}{|\mathcal A|}\sum_{a'}A(s,a')\Bigr)
  \]  
- 幫助網路專注於區分「狀態本身價值」與「動作優勢」，在狀態價值接近時更快收斂。

---

## 2. 實驗設定

**環境 (player mode)**  
- 只有 Player 的初始位置隨機，Goal、Pit、Wall 固定。  
- Board 大小：4×4

**網路架構**  
- Input: 4×4×4 one-hot 狀態展平為 64 維向量  
- Hidden: 150 個 ReLU 神經元  
- Output: 4 個動作對應的 Q 值

**共用超參數**  
- Episodes: 1000  
- Batch size: 64  
- Replay 面板容量: 5000  
- 學習率: 1e-3 (Adam)  
- 折扣因子 \(\gamma\): 0.99  
- ε-greedy: ε 從 1.0 指數衰減至 0.05（衰減速率 500）  
- Double DQN: 同步目標網路間隔 50 次更新

---

## 3. 結果觀察

**Double DQN Reward 曲線**  
- 收斂速度與穩定性：  
  - 相較 basic DQN，抖動幅度更小，較早到達穩定回報。
![image](https://github.com/user-attachments/assets/ff973fc7-c383-434d-bd75-3e1036e72376)


**Dueling DQN Reward 曲線**  
- 收斂行為：  
  - 在早期即獲得較高平均回報，後期收斂平穩。
![image](https://github.com/user-attachments/assets/a7541fc7-e3a0-4c4a-8fd5-54b7e7247cfb)




---

## 4. 心得與討論

- **Double DQN**  
  - 有效降低 Q 值過度估計，使學習更穩定；  
  - 同步頻率可影響性能，過高頻率則接近 single-network 行為。

- **Dueling DQN**  
  - 能更快區分狀態價值與動作優勢，在隨機起點情境中更具魯棒性；  
  - 適合在多動作但狀態差異較小的環境中使用。

---

## 5. 小結

本實驗實現並比較了 **Double DQN** 與 **Dueling DQN**，並在 `player` 模式下驗證：  
- Double DQN 提升了學習穩定性；  
- Dueling DQN 加快了策略收斂速度。  

後續可在 `random` 模式或其他超參數下進行更多測試，以深入探討各變體在不同隨機性條件下的表現差異。 
