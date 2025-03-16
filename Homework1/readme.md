# Grid Map Path Planning with Value Iteration

This project is a Flask-based web application that allows users to create a grid map, designate a start point, an end point, and obstacles, then computes an optimal path using the Value Iteration algorithm. The following document outlines the project following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

## 1. Business Understanding

The purpose of this project is to simulate a simple grid-based environment where an agent can determine the optimal path from a starting position to a target endpoint while avoiding obstacles. The application demonstrates key concepts of reinforcement learning and dynamic programming through the Value Iteration algorithm. It is designed as an educational tool for visualizing decision-making processes in grid-based systems.

## 2. Data Understanding

### Input Data:
- **Grid Configuration:**  
  A user-defined square grid (size between 3x3 and 9x9).  
- **User Selections:**  
  - **Start Point:** Marked in green.  
  - **End Point:** Marked in red.  
  - **Obstacles:** Marked in gray (maximum of n-2 obstacles).

### Data Characteristics:
- The grid is represented as a 2D array where each cell holds a state that influences the reward and transitions in the Value Iteration process.
- Rewards are assigned as follows:
  - **End Point:** +20 reward.
  - **Obstacle:** 0 reward.
  - **Regular cell:** -1 (step cost).

## 3. Data Preparation

### Grid Initialization and User Interaction:
- The web interface allows users to input the grid dimensions and interactively set the start, end, and obstacles.
- Upon interaction, the grid is initialized with default values ("empty") and updated based on user clicks.
- The final grid configuration is serialized to JSON and sent to the backend for processing.

### Preprocessing for Modeling:
- The grid is converted into a suitable data structure for the Value Iteration algorithm.
- Each cell is prepared with an initial value (0 for most cells, except +20 for the end state) before iterative processing begins.

## 4. Modeling

### Value Iteration Algorithm:
- **Objective:** Compute the optimal state-value function and derive the best policy (optimal actions) for each cell.
- **Process:**
  1. **Initialization:**  
     Each cell's value is initialized based on its type (end cell gets +20, obstacles get 0, others start at 0).
  2. **Iteration:**  
     The algorithm iterates over each cell and updates its value using the formula:  
     `V(s) = R(s) + γ * max<sub>a</sub> V(s')`  
     where γ (discount factor) is set to 0.9 and the reward R(s) is defined by the cell type.
  3. **Convergence:**  
     The process repeats until the maximum change in value across all cells is less than a threshold (θ = 1e-3).

### Policy Extraction:
- Once the state values converge, the optimal policy is determined for each cell by selecting the action that maximizes the expected value.

## 5. Evaluation

### Output Visualization:
- The results are displayed on a web page as a grid, where:
  - Each cell shows the computed state value.
  - Optimal actions are represented by directional arrows (↑, ↓, ←, →) indicating the best move.
- Additionally, the agent's movement is simulated on the grid to demonstrate the application of the computed policy in real time.

### Performance Considerations:
- The iterative process ensures that values are accurately computed within the tolerance level, making the approach robust for grids of varying sizes (within the provided limits).

## 6. Deployment

### How to Run the Application:
1. **Install Dependencies:**
   ```bash
   pip install flask
