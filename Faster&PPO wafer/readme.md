
# 📈 Single-Stage Wafer Classification Optimization with PPO

This project applies **Reinforcement Learning (PPO)** to optimize classification accuracy on wafer defect images, focusing on improving performance from 75% to 80%+. It follows the **CRISP-DM** methodology.

---

## 🔍 1. Business Understanding

In semiconductor manufacturing, accurate classification of wafer defects is crucial. However, standard classification models often plateau at around 75% accuracy. The objective of this project is to design an **RL-based single-stage optimization system** that pushes model performance beyond 80% through **reward shaping and focused sampling**.

---

## 📊 2. Data Understanding

### Dataset:
- Source: Preprocessed wafer defect image data
- Format: 2D grayscale numpy arrays
- Labels: 9-class defect types
- Input shape: `(32, 32)` images (converted to features)

### Class Distribution:
The dataset is imbalanced. Rare and difficult classes (e.g., 0, 2, 4, 5, 6, 7) are emphasized via sampling and reward strategy.

---

## 🧹 3. Data Preparation

### Steps:
- Loaded data using pickle: `x_train`, `y_train`, `x_test`, `y_test`
- Label encoding (`LabelEncoder`)
- Balanced training data with over/undersampling
- Feature extraction:
  - Flattened pixel values
  - Statistical summary (mean, std, percentiles)
  - Spatial region analysis
  - Edge strength

### Output:
- Final feature vectors for PPO agent
- Contextual information appended for decision-making

---

## 🧠 4. Modeling

### Reinforcement Learning with PPO:
- Framework: `stable-baselines3` PPO
- Environment: `SingleStageWaferEnv`
- Policy: MLP with custom architecture
- Reward Design:
  - Large reward for correct difficult class predictions
  - Bonus for improving accuracy from 75% → 80%
  - Penalties for wrong predictions on easy classes
- Focused sampling mechanism prioritizes underperforming or confusing classes

### Hyperparameters:
```python
learning_rate = 1.5e-4
n_steps = 768
batch_size = 96
n_epochs = 15
gamma = 0.98
clip_range = 0.18
ent_coef = 0.025
net_arch = dict(pi=[384, 192, 96], vf=[384, 192, 96])
```

---

## 📈 5. Evaluation

### During Training:
- Evaluated reward trends every 2000 steps
- Monitored estimated accuracy
- Saved checkpoints for high-performance models

### After Training:
- Evaluated on held-out test set (`x_test`)
- Accuracy reported
- Visualized progress against baseline (74.9%) and target (80%)

### Example Output:
```text
🎯 最終結果:
  測試準確率: 0.8012
  改進次數: 5
  vs 74.9%: +0.052
🎉 成功達到80%目標！
```

### Visual:
![Progress](./single_stage_result_YYYYMMDD_HHMMSS.png)

---

## 🛠 6. Deployment

- Model saved using Stable-Baselines3: `single_stage_final_TIMESTAMP.zip`
- Usable for future inference or transfer learning
- Environment can be reused in a multi-stage setup

---

## 🔁 7. Next Steps

- Integrate with a **multi-stage pipeline**
- Add **active learning** to prioritize ambiguous samples
- Incorporate real-time feedback into the reward function

---

## 📁 File Structure

```
├── single_stage_optimization.py     # Main training script
├── data/
│   ├── x_train_org_20210614.pickle
│   ├── y_train_org_20210614.pickle
│   ├── x_test_20210614.pickle
│   └── y_test_20210614.pickle
├── single_stage_result_TIMESTAMP.png  # Accuracy visualization
└── single_stage_final_TIMESTAMP.zip   # Trained PPO model
```

---

## 📦 Dependencies

- Python 3.8+
- NumPy, scikit-learn, matplotlib, seaborn
- torch, stable-baselines3, gymnasium

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

- Tom Chou
- Reinforcement Learning for Semiconductor Inspection
