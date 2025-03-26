# Grid World Optimal Policy Application

This project utilizes Flask to demonstrate a visual interactive implementation of the Value Iteration algorithm in a Grid World environment. Users can intuitively set start, end, and obstacle states on a grid and visualize the optimal policy derived through Value Iteration.

## CRISP-DM Methodology

### 1. Business Understanding

#### Objective:
- Provide a visual, interactive educational tool to understand and demonstrate the Value Iteration algorithm for solving Markov Decision Processes (MDPs).

#### Requirements:
- Visual representation of states and transitions.
- Easy setting of start, end, and obstacles.
- Clear visualization of optimal policy and value functions.

### 2. Data Understanding

#### Data Description:
- Grid represented as a two-dimensional array.
- Each cell state includes attributes:
  - `start`: Boolean (Starting cell)
  - `end`: Boolean (Goal cell)
  - `obstacle`: Boolean (Non-accessible cell)
  - `value`: Float (State-value)
  - `policy`: String (Optimal action)
  - `highlight`: Boolean (Optimal path visibility)

### 3. Data Preparation

#### Processing:
- Grid initialization based on user inputs.
- JSON-formatted data for backend processing.
- Continuous updates based on frontend interaction.

### 4. Modeling

#### Algorithm:
- **Value Iteration Algorithm**:
  - Iteratively calculates optimal values for each state.
  - Determines the optimal policy for maximizing cumulative rewards.

#### Parameters:
- Discount factor (`GAMMA`): `0.9`
- Convergence threshold (`THRESHOLD`): `1e-3`
- Step reward (`REWARD_STEP`): `-0.01`
- Goal reward (`REWARD_GOAL`): `1.0`

### 5. Evaluation

#### Metrics:
- Convergence checked through threshold-based stopping criteria.
- Optimal policy visually validated through highlighted paths.

#### User Validation:
- Interactive frontend allows for intuitive assessment and real-time visualization of policy changes.

### 6. Deployment

#### Deployment Platform:
- Flask web application deployed locally or through a cloud service.

#### Execution:
1. Install Python dependencies:
   ```bash
   pip install flask
   python app.py

  ![螢幕擷取畫面 2025-03-26 195845](https://github.com/user-attachments/assets/1a215ca1-c8fe-4ecd-96f3-2da9fbb8bc5c)

