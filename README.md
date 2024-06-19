# Advanced Sensor Data Fusion in Distributed Systems

## Program Assignment 
- Simulate a target with motion and process noise.
- 4D state space: $x = (x, y, x’, y’)^\intercal$
- Linear Measurements: $z = (x, y)^\intercal + v$
- $v \sim N (0, R), R = R I, R = 100$.
- Parameter $S =$ number of sensors (e.g. $S = 4$).
- For $k = 1,...$
  - Each sensor produces a measurement.
  - Each sensor uses Kalman Filter for local processing.
  - Its estimate is sent to fusion center (FC) instance.
    1) The FC uses the convex combination for Track-to-Track Fusion
    2) The FC uses Tracklet Fusion

## Usage
```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
