# IsaacGym 

Testing with the IsaacGym basic templates.


### Results 
----
1) Main IsaacGym 
```bash
 python train.py task=Anymal checkpoint=/home/camilo/repos/IsaacGym_repo/IsaacGymEnvs/isaacgymenvs/runs/Anymal_from_omni/anymal.pth test=True num_envs=6 rl_device=cpu sim_device=cpu pipeline=cpu
 ```

 <img src="Images/IsaacGym_baseRepo_load_OmniModel_anymal.png" width=700>

 2) Random location, and no locomation controller. (basic_ant_test1.py)

  <img src="Images/IsaacGym_basic_ant_random_location.png" width=700>

 3) **legged_gym project** some changes has been done to force the CPU execution. unfortunately, It wasn't able to upload the controller Omni-model (i.e. policy.pt / anymalTerrain.pth)   


 ### Notes
 ---
 1) Some testing and commands are listed in the "notes" file  