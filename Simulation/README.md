# 3D Simulation Code
---

# Results
---
### Predefine Sequence
* IsaacGym simulation based on MPC-RL locomotion controller developed by Yulun Zhuang 


<img src="Images/short_sequence_init.gif" width=500>

<img src="Images/short_sequence_stop.gif" width=500>



## Folder structure
---
```
├── 0_isaacGym_test1
│   └── basic_anymal_spawn.py
├── 1_IsaacGym_mpc_base
│   ├── RL_MPC_Locomotion_custom.py
│   └── trajectory_lib
│       ├── basic_commands.py
│       ├── __init__.py
│       └── __pycache__
│           ├── basic_commands.cpython-37.pyc
│           └── __init__.cpython-37.pyc
├── Images
│   ├── sequence_3robot.png
│   ├── sequence_mpc_3robots_cleaned.mp4
│   ├── short_sequence_init.gif
│   └── short_sequence_stop.gif
├── notes.txt
└── README.md
```



## Reference
---
1) MPC-RL locomotion controller developed by Yulun Zhuang https://github.com/silvery107/rl-mpc-locomotion.git


## Extra

* /IsaacGym_repo/rl-mpc-locomotion/RL_Environment/sim_utils.py
