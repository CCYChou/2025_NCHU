# README: HW3 – Multi-Armed Bandit Exploration vs Exploitation

This project implements four Bernoulli multi-armed bandit algorithms in a Colab-ready Jupyter Notebook. It follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework to structure the development, analysis, and deployment of the experiments.

## 1. Business Understanding
- **Objective:** Explore the trade-off between exploration and exploitation in sequential decision-making.
- **Algorithms:** Epsilon-Greedy, UCB1 (Upper Confidence Bound), Softmax (Boltzmann), Thompson Sampling.
- **Metrics:** Average cumulative reward over time.

## 2. Data Understanding
- **Simulated Data:** Synthetic Bernoulli arms with fixed success probabilities (`true_probs = [0.1, 0.5, 0.8, 0.3, 0.6]`).
- **Horizon:** Number of time steps (`HORIZON = 1000`).
- **Runs:** Number of independent simulation runs (`RUNS = 100`).

## 3. Data Preparation
- No external dataset required; probability parameters are defined in code.
- Results stored in arrays of shape `(HORIZON,)` for cumulative rewards.

## 4. Modeling
Each algorithm is implemented as a Python function:
1. **Epsilon-Greedy:** Random vs. greedy selection controlled by parameter ε.
![Epsilon](https://github.com/user-attachments/assets/4ada4940-fbe9-43fb-b096-18ed7e75d0c8)
2. **UCB1:** Confidence bound balancing using parameter c.
![UCB1](https://github.com/user-attachments/assets/27da68de-92c6-442f-a8d5-29b66815dc13)
3. **Softmax:** Probability-weighted selection controlled by temperature τ.
![Softmax](https://github.com/user-attachments/assets/bc4a5a3f-cf2a-4465-847a-c9473d83752e)
4. **Thompson Sampling:** Bayesian sampling with Beta priors (`α`, `β`).
![Thompson Sampling](https://github.com/user-attachments/assets/428162d5-0bcd-45f0-b412-339d41a2794a)

![ALL](https://github.com/user-attachments/assets/e18ef90d-58c2-45f3-bb18-066ffe2691e2)
## 5. Evaluation
- **Visualization:** Plots of average cumulative reward vs. time step for each algorithm.
- **Comparison:** Overlaid curves highlight convergence speed and long-term performance.
- **Interpretation:** Each cell includes a textual discussion of spatial (reward curves) and temporal (convergence) behavior.

## 6. Deployment
- **Colab Notebook:** `HW3_MAB_Bandit.ipynb` can be run directly in Google Colab.
- **Dependencies:** Python 3.x, `numpy`, `matplotlib`.
- **Usage:** Clone or upload notebook to Colab, ensure dependencies are installed, then execute all cells.

---
For questions or improvements, please refer to the notebook or contact the author.
