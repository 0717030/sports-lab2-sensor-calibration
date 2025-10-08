# CSDS30012 Sports Data Processing and Analysis Lab 2: Sensor Calibration
### 檔案結構
```
/
├─ src/
│  ├─ main.py        # CLI 入口
│  ├─ pipeline.py    # 流程：讀→校正→姿態→重力補償→積分→出圖
│  ├─ core.py        # 演算法：bias/零速偵測/互補濾波/積分/去漂
│  ├─ io.py          # 讀 CSV、輸出目錄、manifest、(可選) 匯出校正後 CSV
│  └─ viz.py         # 視覺化（含動作插件：pendulum/elevator/square）
├─ data/             # 資料 CSV 放這裡（e.g., pendulum.csv, elevator.csv）
└─ results/          # 圖檔與輸出會寫到這裡
```
### 使用方式
0. 先靜止幾秒再錄指定動作，輸出 CSV
1. 把 CSV 放在 data/，檔名要包含 pendulum 或 elevator 以自動判斷動作
2. **多動作一起處理** 
    `python -m src.main --data-dir data --out-root results`
   **指定動作** 
   `python -m src.main --csv data/pendulum.csv --action pendulum --out-root results`
3. 輸出會到：
    ```
    results/
      ├─ pendulum/
      │    ├─ acc_after_world.png, vel_after_world.png, pos_after_world.png
      │    ├─ gyro_VT_deg_per_s.png, gyro_XT_deg.png
      │    ├─ pendulum_yaw.png
      │    └─ manifest.json
      └─ elevator/
           ├─ acc_after_world.png, vel_after_world.png, pos_after_world.png
           ├─ gyro_VT_deg_per_s.png, gyro_XT_deg.png
           ├─ elevator_Az.png, elevator_Vz.png, elevator_Z.png
           └─ manifest.json

    ```
