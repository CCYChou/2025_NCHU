from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --------------------------
# 全域參數設定
# --------------------------
GAMMA = 0.9         # 折扣因子
THRESHOLD = 1e-3    # 收斂判斷閾值
REWARD_STEP = -0.01 # 每一步的獎勵
REWARD_GOAL = 1.0   # 終點獎勵
DEFAULT_SIZE = 5    # 預設 grid 大小

# --------------------------
# 幫助函式：找出鄰近可移動的格子 (上、下、左、右)
# --------------------------
def get_neighbors(r, c, grid):
    moves = []
    rows = len(grid)
    cols = len(grid[0])
    directions = {
        '↑': (-1, 0),
        '↓': (1, 0),
        '←': (0, -1),
        '→': (0, 1)
    }
    for action, (dr, dc) in directions.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc]['obstacle']:
            moves.append((action, nr, nc))
    return moves

# --------------------------
# 價值迭代演算法
# --------------------------
def value_iteration(grid):
    rows = len(grid)
    cols = len(grid[0])
    # 初始化所有格子的 value
    for r in range(rows):
        for c in range(cols):
            grid[r][c]['value'] = 0.0

    while True:
        delta = 0.0
        new_values = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]['obstacle']:
                    new_values[r][c] = 0.0
                    continue
                if grid[r][c]['end']:
                    new_values[r][c] = REWARD_GOAL
                    continue

                neighbors = get_neighbors(r, c, grid)
                if not neighbors:
                    new_values[r][c] = 0.0
                    continue

                q_values = []
                for action, nr, nc in neighbors:
                    reward = REWARD_GOAL if grid[nr][nc]['end'] else REWARD_STEP
                    q = reward + GAMMA * grid[nr][nc]['value']
                    q_values.append(q)
                best_q = max(q_values)
                new_values[r][c] = best_q

        for r in range(rows):
            for c in range(cols):
                diff = abs(new_values[r][c] - grid[r][c]['value'])
                if diff > delta:
                    delta = diff
        for r in range(rows):
            for c in range(cols):
                grid[r][c]['value'] = new_values[r][c]
        if delta < THRESHOLD:
            break

    # 推導最佳政策
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]['obstacle']:
                grid[r][c]['policy'] = 'X'
            elif grid[r][c]['end']:
                grid[r][c]['policy'] = 'GOAL'
            else:
                neighbors = get_neighbors(r, c, grid)
                if not neighbors:
                    grid[r][c]['policy'] = 'N/A'
                    continue
                best_action = None
                best_q = float('-inf')
                for action, nr, nc in neighbors:
                    reward = REWARD_GOAL if grid[nr][nc]['end'] else REWARD_STEP
                    q = reward + GAMMA * grid[nr][nc]['value']
                    if q > best_q:
                        best_q = q
                        best_action = action
                grid[r][c]['policy'] = best_action
    return grid

# --------------------------
# 顯示最佳路徑 (Highlight) - 從 Start 沿最佳政策標記路徑
# --------------------------
def highlight_optimal_path(grid):
    rows = len(grid)
    cols = len(grid[0])
    start_rc = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]['start']:
                start_rc = (r, c)
                break
        if start_rc is not None:
            break

    if start_rc is None:
        return grid

    r, c = start_rc
    while True:
        grid[r][c]['highlight'] = True
        if grid[r][c]['end']:
            break
        action = grid[r][c]['policy']
        if action in ('GOAL', 'N/A', 'X', None):
            break
        nr, nc = r, c
        if action == '↑':
            nr -= 1
        elif action == '↓':
            nr += 1
        elif action == '←':
            nc -= 1
        elif action == '→':
            nc += 1
        if not (0 <= nr < rows and 0 <= nc < cols):
            break
        if grid[nr][nc]['obstacle']:
            break
        r, c = nr, nc

    return grid

# --------------------------
# HTML + JS (內嵌模板) - 美化版 Grid 外觀與人性化提示
# --------------------------
html_template = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <title>美化版 Grid World 價值迭代</title>
  <!-- 引入 Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary-color: #333;
      --secondary-color: #4CAF50;
      --danger-color: #F44336;
      --obstacle-color: #555;
      --bg-color: #f4f4f4;
      --cell-bg: #ffffff;
      --highlight-bg: #c8e6c9;
    }
    body {
      font-family: 'Roboto', sans-serif;
      margin: 0;
      padding: 0;
      background-color: var(--bg-color);
    }
    .header {
      background: linear-gradient(135deg, var(--primary-color), #222);
      color: #fff;
      padding: 20px;
      text-align: center;
    }
    .container {
      max-width: 900px;
      margin: 20px auto;
      background-color: #fff;
      padding: 20px 30px;
      border-radius: 10px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .btn {
      padding: 10px 16px;
      margin: 5px;
      background-color: var(--primary-color);
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }
    .btn:hover {
      background-color: #555;
    }
    .grid-container {
      margin: 20px auto;
      display: grid;
      grid-template-columns: repeat(5, 70px);
      gap: 8px;
      justify-content: center;
    }
    .cell {
      width: 70px;
      height: 70px;
      background-color: var(--cell-bg);
      border: 2px solid #ddd;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      font-weight: 500;
      position: relative;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .cell:hover {
      transform: translateY(-3px);
      box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .cell.obstacle {
      background-color: var(--obstacle-color);
      color: #fff;
    }
    .cell.start {
      background-color: var(--secondary-color);
      color: #fff;
    }
    .cell.end {
      background-color: var(--danger-color);
      color: #fff;
    }
    .cell.highlight {
      background-color: var(--highlight-bg) !important;
    }
    .value-label {
      font-size: 10px;
      position: absolute;
      bottom: 4px;
      right: 4px;
      color: var(--primary-color);
    }
    .footer {
      text-align: center;
      font-size: 12px;
      color: #666;
      margin-top: 20px;
    }
    .note {
      margin: 15px 0;
      padding: 10px 15px;
      background: #e8f5e9;
      border-left: 4px solid var(--secondary-color);
      color: #333;
      border-radius: 4px;
    }
  </style>
</head>
<body>

<div class="header">
  <h1>HW2使用價值迭代算法推導最佳政策</h1>
  <p>直覺化顯示最佳政策與價值函數</p>
</div>

<div class="container">
  <div class="note">
    <p>請點選格子設定「起點」、「終點」或「障礙物」，並可利用「清除」或「Reset」功能調整。完成設定後，按「計算最佳政策」查看最佳路徑與各狀態價值。</p>
  </div>
  <div>
    <button class="btn" onclick="setMode('start')">設定起點</button>
    <button class="btn" onclick="setMode('end')">設定終點</button>
    <button class="btn" onclick="setMode('obstacle')">設定障礙物</button>
    <button class="btn" onclick="setMode('clear')">清除格子</button>
    <button class="btn" onclick="computePolicy()">計算最佳政策</button>
    <button class="btn" onclick="resetGridConfirm()">Reset</button>
  </div>
  <div id="grid-container" class="grid-container"></div>
  <div class="footer">
    <p>© 2025 - NCHU 周俊言!</p>
  </div>
</div>

<script>
let gridSize = 5;
let mode = 'start';
let grid = [];

window.onload = function() {
  initGrid(gridSize);
  renderGrid();
};

function initGrid(size) {
  grid = [];
  for (let r = 0; r < size; r++) {
    let row = [];
    for (let c = 0; c < size; c++) {
      row.push({
        start: false,
        end: false,
        obstacle: false,
        value: 0.0,
        policy: '',
        highlight: false
      });
    }
    grid.push(row);
  }
}

function renderGrid() {
  const container = document.getElementById('grid-container');
  container.innerHTML = '';
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      const cellDiv = document.createElement('div');
      cellDiv.className = 'cell';

      if (grid[r][c].obstacle) {
        cellDiv.classList.add('obstacle');
      } else if (grid[r][c].start) {
        cellDiv.classList.add('start');
      } else if (grid[r][c].end) {
        cellDiv.classList.add('end');
      }
      if (grid[r][c].highlight) {
        cellDiv.classList.add('highlight');
      }

      if (grid[r][c].policy) {
        cellDiv.innerText = grid[r][c].policy;
      }

      let valueLabel = document.createElement('div');
      valueLabel.className = 'value-label';
      valueLabel.innerText = grid[r][c].value.toFixed(2);
      cellDiv.appendChild(valueLabel);

      cellDiv.onclick = () => { handleCellClick(r, c); };

      container.appendChild(cellDiv);
    }
  }
}

function resetGridConfirm() {
  const confirmReset = confirm("確定要重置整個網格嗎？這將清除所有設定。");
  if (confirmReset) {
    resetGrid();
  }
}

function resetGrid() {
  initGrid(gridSize);
  renderGrid();
}

function handleCellClick(r, c) {
  if (mode === 'start') {
    clearCell(r, c);
    grid[r][c].start = true;
  } else if (mode === 'end') {
    clearCell(r, c);
    grid[r][c].end = true;
  } else if (mode === 'obstacle') {
    clearCell(r, c);
    grid[r][c].obstacle = true;
  } else if (mode === 'clear') {
    clearCell(r, c);
  }
  renderGrid();
}

function clearCell(r, c) {
  grid[r][c].start = false;
  grid[r][c].end = false;
  grid[r][c].obstacle = false;
  grid[r][c].value = 0.0;
  grid[r][c].policy = '';
  grid[r][c].highlight = false;
}

function setMode(newMode) {
  mode = newMode;
}

function computePolicy() {
  fetch('/compute_policy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ size: gridSize, grid: grid })
  })
  .then(response => response.json())
  .then(data => {
    grid = data;
    renderGrid();
  })
  .catch(err => { console.error('Error:', err); });
}
</script>

</body>
</html>
"""

# --------------------------
# Flask 路由設定
# --------------------------
@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/compute_policy', methods=['POST'])
def compute_policy():
    data = request.json
    grid_size = data.get('size', DEFAULT_SIZE)
    grid_data = data.get('grid', [])
    grid = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            cell_info = grid_data[r][c]
            cell = {
                'start': cell_info.get('start', False),
                'end': cell_info.get('end', False),
                'obstacle': cell_info.get('obstacle', False),
                'value': 0.0,
                'policy': '',
                'highlight': False
            }
            row.append(cell)
        grid.append(row)
    value_iteration(grid)
    highlight_optimal_path(grid)
    return jsonify(grid)

if __name__ == '__main__':
    app.run()

