# 3D Simulation Code
---

### Results
---
#### Predefine Sequence
* IsaacGym simulation based on MPC-RL locomotion controller developed by Yulun Zhuang 


<img src="Images/short_sequence_init.gif" width=500>

<img src="Images/short_sequence_stop.gif" width=500>



### Folder structure
---
```
├── 0_isaacGym_test1
│   ├── basic_anymal_spawn.py
|   |
│   ├── Omni_IsaacGym_models_2022
│   │   ├── anymal.pth
│   │   ├── anymal_terrain.pth
│   │   ├── cartpole.pth
│   │   ├── nucleus_steps.txt
│   │   └── policy_1.pt
|   |
│   └── README.md
|
├── 1_IsaacGym_mpc_base
│   ├── RL_MPC_Locomotion_custom.py
│   ├── sim_utils.py
|   |
│   └── trajectory_lib
│       ├── basic_commands.py
|
├── notes.txt
└── README.md

```



### Reference
---
1) MPC-RL locomotion controller developed by Yulun Zhuang https://github.com/silvery107/rl-mpc-locomotion.git


### Extra

* /IsaacGym_repo/rl-mpc-locomotion/RL_Environment/sim_utils.py
