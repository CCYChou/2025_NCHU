from flask import Flask, request, render_template_string
import json, random, copy

app = Flask(__name__)

# -------------------------------
# HTML：HW1-1 網格地圖開發（建立與設定網格）
# -------------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>HW1-1: 網格地圖開發</title>
  <link href="https://fonts.googleapis.com/css?family=Noto+Sans+TC&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Noto Sans TC', sans-serif;
      background: #f7f7f7;
      margin: 0;
      padding: 20px;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: #fff;
      padding: 30px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
      border-radius: 8px;
    }
    h1 { text-align: center; color: #333; }
    p, li { color: #555; }
    .text-center { text-align: center; }
    form { text-align: center; }
    input[type="number"] {
      width: 60px;
      padding: 5px;
      font-size: 16px;
      margin-right: 10px;
    }
    button {
      background: #3498db;
      color: #fff;
      border: none;
      padding: 10px 15px;
      font-size: 16px;
      border-radius: 5px;
      cursor: pointer;
      margin: 5px;
    }
    button:hover {
      background: #2980b9;
    }
    table {
      border-collapse: collapse;
      margin: 20px auto;
    }
    td {
      width: 50px;
      height: 50px;
      border: 1px solid #ccc;
      text-align: center;
      vertical-align: middle;
      cursor: pointer;
      font-weight: bold;
      border-radius: 4px;
    }
    .empty { background-color: #fff; }
    .start { background-color: #27ae60; color: #fff; }
    .end { background-color: #e74c3c; color: #fff; }
    .obstacle { background-color: #7f8c8d; color: #fff; }
  </style>
</head>
<body>
  <div class="container">
    <h1>HW1-1: 網格地圖開發</h1>
    <p class="text-center">請輸入網格維度 n（介於 3 到 9）：</p>
    <form id="gridForm" method="POST" action="/evaluate">
      <input type="number" id="dimension" name="dimension" min="3" max="9" required>
      <button type="button" onclick="createGrid()">生成網格</button>
      <br><br>
      <div id="instructions" class="text-center">
        <p>操作步驟：</p>
        <ol style="text-align: left; display: inline-block;">
          <li>點擊一個單元格設定起點（綠色）。</li>
          <li>點擊另一個單元格設定終點（紅色）。</li>
          <li>點擊其他單元格設定障礙物（灰色），總數為 n-2 個。</li>
        </ol>
      </div>
      <div id="gridContainer"></div>
      <br>
      <!-- 將網格設定存入隱藏欄位 -->
      <input type="hidden" id="gridConfig" name="gridConfig">
      <button type="submit">策略評估</button>
    </form>
  </div>
  <script>
    let n;
    let grid = [];
    let startSet = false;
    let endSet = false;
    let obstaclesCount = 0;
    let maxObstacles = 0;
    
    function createGrid(){
      n = parseInt(document.getElementById("dimension").value);
      if(n < 3 || n > 9) {
        alert("n 必須介於 3 到 9 之間");
        return;
      }
      maxObstacles = n - 2;
      // 初始化所有單元格為 "empty"
      grid = [];
      for(let i = 0; i < n; i++){
        let row = [];
        for(let j = 0; j < n; j++){
          row.push("empty");
        }
        grid.push(row);
      }
      // 重置選取狀態
      startSet = false;
      endSet = false;
      obstaclesCount = 0;
      renderGrid();
      document.getElementById("gridConfig").value = JSON.stringify(grid);
    }
    
    function renderGrid(){
      let container = document.getElementById("gridContainer");
      container.innerHTML = "";
      let table = document.createElement("table");
      for(let i = 0; i < n; i++){
        let tr = document.createElement("tr");
        for(let j = 0; j < n; j++){
          let td = document.createElement("td");
          td.id = "cell-" + i + "-" + j;
          td.setAttribute("data-i", i);
          td.setAttribute("data-j", j);
          td.className = grid[i][j];
          td.onclick = cellClicked;
          tr.appendChild(td);
        }
        table.appendChild(tr);
      }
      container.appendChild(table);
    }
    
    function cellClicked(){
      let i = parseInt(this.getAttribute("data-i"));
      let j = parseInt(this.getAttribute("data-j"));
      
      if(grid[i][j] !== "empty"){
        return;
      }
      
      if(!startSet){
        grid[i][j] = "start";
        startSet = true;
      } else if(!endSet){
        grid[i][j] = "end";
        endSet = true;
      } else if(obstaclesCount < maxObstacles){
        grid[i][j] = "obstacle";
        obstaclesCount++;
      } else {
        alert("障礙物數量已達上限 (n-2)");
        return;
      }
      renderGrid();
      document.getElementById("gridConfig").value = JSON.stringify(grid);
    }
  </script>
</body>
</html>
"""

# -------------------------------
# HTML：HW1-2 策略顯示與價值評估結果（含自動移動功能，自動啟動不需按鈕）
# -------------------------------
RESULT_HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>策略評估結果</title>
  <link href="https://fonts.googleapis.com/css?family=Noto+Sans+TC&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Noto Sans TC', sans-serif;
      background: #f7f7f7;
      margin: 0;
      padding: 20px;
    }
    .container {
      max-width: 1000px;
      margin: 0 auto;
      background: #fff;
      padding: 30px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
      border-radius: 8px;
    }
    h1 { text-align: center; color: #333; }
    p, li { color: #555; }
    .text-center { text-align: center; }
    .annotation {
      margin: 20px auto;
      background: #ecf0f1;
      padding: 15px;
      border-radius: 6px;
      max-width: 800px;
    }
    table {
      border-collapse: collapse;
      margin: 20px auto;
    }
    td {
      width: 70px;
      height: 70px;
      border: 1px solid #ccc;
      text-align: center;
      vertical-align: middle;
      font-weight: bold;
      font-size: 14px;
      border-radius: 4px;
      position: relative;
    }
    .empty { background-color: #fff; }
    .start { background-color: #27ae60; color: #fff; }
    .end { background-color: #e74c3c; color: #fff; }
    .obstacle { background-color: #7f8c8d; color: #fff; }
    .agent {
      border: 3px solid yellow;
      box-sizing: border-box;
    }
    .back-btn {
      display: block;
      width: 150px;
      margin: 20px auto 0 auto;
      text-align: center;
      background: #3498db;
      color: #fff;
      padding: 10px;
      text-decoration: none;
      border-radius: 5px;
      cursor: pointer;
    }
    .back-btn:hover {
      background: #2980b9;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>HW1-2: 策略顯示與價值評估</h1>
    <p class="text-center">下表顯示每個單元格的最適策略（箭頭）與狀態價值 V(s)：</p>
    <div class="annotation">
      <p><strong>計算公式：</strong> V(s) = R(s) + γ * max<sub>a</sub> V(s')</p>
      <p>其中：</p>
      <ul>
        <li>R(s)：若狀態為終點 (end)，R(s) = +20；若狀態為障礙 (obstacle)，R(s) = 0；否則 R(s) = -1（每步成本）。</li>
        <li>γ：折扣因子，設定為 0.9。</li>
        <li>max<sub>a</sub> V(s')：在所有可能動作下，下一狀態的最大價值；若動作會導致出界或撞上障礙，則保持原位。</li>
      </ul>
      <p>本計算採用 <strong>值迭代 (Value Iteration)</strong> 方法，直到所有狀態價值變化小於 θ = 1e-3 為止。</p>
    </div>
    <table id="resultTable">
      {% for i in range(n) %}
        <tr>
          {% for j in range(n) %}
            {% set cell = grid[i][j] %}
            <td id="cell-{{ i }}-{{ j }}" class="{{ cell.type }}">
              {% if cell.type == 'obstacle' %}
                X
              {% elif cell.type == 'end' %}
                終點<br>V: {{ "%.2f"|format(cell.value) }}
              {% else %}
                {% if cell.policy_symbol %}
                  {{ cell.policy_symbol }}<br>
                {% endif %}
                V: {{ "%.2f"|format(cell.value) }}
              {% endif %}
            </td>
          {% endfor %}
        </tr>
      {% endfor %}
    </table>
    <div class="text-center">
      <a class="back-btn" href="/">返回 HW1-1</a>
    </div>
  </div>
  <!-- 自動移動腳本 -->
  <script>
    var policyGrid = {{ grid|tojson }};
    var n = {{ n }};
    // 複製一份用於模擬自動移動（原始 grid 不變）
    var moveGrid = JSON.parse(JSON.stringify(policyGrid));
    
    function nextState(i, j, action) {
      let new_i = i, new_j = j;
      if(action === "up") {
        new_i = i - 1;
      } else if(action === "down") {
        new_i = i + 1;
      } else if(action === "left") {
        new_j = j - 1;
      } else if(action === "right") {
        new_j = j + 1;
      }
      if(new_i < 0 || new_i >= n || new_j < 0 || new_j >= n) {
        return {i: i, j: j};
      }
      if(moveGrid[new_i][new_j]["type"] === "obstacle") {
        return {i: i, j: j};
      }
      return {i: new_i, j: new_j};
    }
    
    function autoMove() {
      // 移除所有 cell 的 agent 樣式
      for (let i = 0; i < n; i++){
        for (let j = 0; j < n; j++){
          let cellElem = document.getElementById("cell-" + i + "-" + j);
          if(cellElem) cellElem.classList.remove("agent");
        }
      }
      // 尋找起點位置
      let startPos = null;
      for (let i = 0; i < n; i++){
        for (let j = 0; j < n; j++){
          if(moveGrid[i][j]["type"] === "start"){
            startPos = {i: i, j: j};
            break;
          }
        }
        if(startPos) break;
      }
      if(!startPos) {
        alert("找不到起點！");
        return;
      }
      let current = startPos;
      document.getElementById("cell-" + current.i + "-" + current.j).classList.add("agent");
      
      let interval = setInterval(function(){
        if(moveGrid[current.i][current.j]["type"] === "end"){
          clearInterval(interval);
          setTimeout(autoMove, 1000);
          return;
        }
        let action = moveGrid[current.i][current.j]["policy"];
        if(!action){
          clearInterval(interval);
          return;
        }
        let next = nextState(current.i, current.j, action);
        document.getElementById("cell-" + current.i + "-" + current.j).classList.remove("agent");
        current = {i: next.i, j: next.j};
        document.getElementById("cell-" + current.i + "-" + current.j).classList.add("agent");
        if(moveGrid[current.i][current.j]["type"] === "end"){
          clearInterval(interval);
          setTimeout(autoMove, 1000);
        }
      }, 500);
    }
    
    window.onload = function() {
      autoMove();
    };
  </script>
</body>
</html>
"""

# -------------------------------
# 政策箭頭符號對應（最佳策略箭頭）
# -------------------------------
policy_symbols = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→"
}

# -------------------------------
# 值迭代計算最適價值與最佳策略
# -------------------------------
def value_iteration(grid_config, n, gamma=0.9, theta=1e-3):
    grid = []
    for i in range(n):
        row = []
        for j in range(n):
            cell_type = grid_config[i][j]
            value = 20.0 if cell_type == "end" else 0.0
            cell = {"type": cell_type, "value": value, "policy": None}
            row.append(cell)
        grid.append(row)
    
    def reward(i, j):
        if grid[i][j]["type"] == "end":
            return 20
        elif grid[i][j]["type"] == "obstacle":
            return 0
        else:
            return -1
    
    def next_state(i, j, action):
        if action == "up":
            new_i, new_j = i - 1, j
        elif action == "down":
            new_i, new_j = i + 1, j
        elif action == "left":
            new_i, new_j = i, j - 1
        elif action == "right":
            new_i, new_j = i, j + 1
        else:
            new_i, new_j = i, j
        if new_i < 0 or new_i >= n or new_j < 0 or new_j >= n or grid[new_i][new_j]["type"] == "obstacle":
            return i, j
        return new_i, new_j

    actions = ["up", "down", "left", "right"]
    
    while True:
        delta = 0
        new_grid = copy.deepcopy(grid)
        for i in range(n):
            for j in range(n):
                if grid[i][j]["type"] in ["end", "obstacle"]:
                    continue
                best_value = float("-inf")
                for a in actions:
                    ni, nj = next_state(i, j, a)
                    action_value = reward(i, j) + gamma * grid[ni][nj]["value"]
                    best_value = max(best_value, action_value)
                new_grid[i][j]["value"] = best_value
                delta = max(delta, abs(grid[i][j]["value"] - best_value))
        grid = new_grid
        if delta < theta:
            break

    for i in range(n):
        for j in range(n):
            if grid[i][j]["type"] in ["end", "obstacle"]:
                grid[i][j]["policy"] = None
            else:
                best_action = None
                best_action_value = float("-inf")
                for a in actions:
                    ni, nj = next_state(i, j, a)
                    action_value = reward(i, j) + gamma * grid[ni][nj]["value"]
                    if action_value > best_action_value:
                        best_action_value = action_value
                        best_action = a
                grid[i][j]["policy"] = best_action
                grid[i][j]["policy_symbol"] = policy_symbols[best_action] if best_action else ""
    return grid

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    n = int(request.form.get('dimension'))
    grid_config_json = request.form.get('gridConfig')
    if not grid_config_json:
        return "請先生成並設定網格", 400
    grid_config = json.loads(grid_config_json)
    evaluated_grid = value_iteration(grid_config, n)
    return render_template_string(RESULT_HTML, grid=evaluated_grid, n=n)

if __name__ == '__main__':
    app.run(debug=True)
