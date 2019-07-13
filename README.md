# Robot_Learning

Sawyer robot learning with PPO algorithms with visual-based observations, with Unity for Sawyer robot simulation and game building.

## Getting Started

```
git clone https://github.com/quantumiracle/Robot_Learning.git

cd Robot_Learning

chmod +x sawyer_visual.x86

(./sawyer_visual.x86 could see the robot env running)

conda create -n robo python=3.5

source activate robo

pip install tensorflow-gpu==1.12.0

pip install matplotlib scipy gym gym_unity

python ppo_sawyer_visual.py --train
```

## Troubleshoot:
If you meet the problem(error) of array or list with gym_unity, try:

change in conda files:  `/envs/robo/lib/python3.5/site-packages/gym_unity/envs/unity_env.py` line 172 in `_single_step()`function add `info.visual_observations=np.array(info.visual_observations)`.
