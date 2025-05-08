# Quarter-car Active Suspension Control System

In this project, an active suspension control system is implemented on a simplified quarter-car model and simulated under uneven road conditions for better driving comfort. Internal stability, controllability and observability of the quarter-car model are analyzed as prerequisites of controller design. A Luenberger observer is designed so that it can be assumed that system states are perfectly observable. Ultimately, H2- and H∞-optimal state feedback controllers are designed to enhance model performance in terms of driving comfort.

## Installation
An environment with Python 3.8 or higher version is required, as well as `pip` package.

```
git clone https://github.com/xchen2763/ActiveSuspensionControl

cd ActiveSuspensionControl

pip install -r requirements.txt
```

## Quickstart
Run [Stability.py](./Stability.py), [Controllability.py](./Controllability.py) and [Observability.py](./Observability.py) for internal stability, controllability and observability analysis.

Run [Passive_Suspension.py](./Passive_Suspension.py), [H2_optimal_control.py](./H2_optimal_control.py) and [H_inf_state_feedback_control.py](./H_inf_state_feedback_control.py) for open-loop, H2-optimal and H∞-optimal simulation.