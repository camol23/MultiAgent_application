import sys
import time
import numpy as np

from Env import env_v1
from aux_libs import ploting


# Agents Settings
agents_settings = {
               # (x0, y0)
    'start_pos': (100, 550),  #(50, 550),
    'num_agents': 1,
    'formation_type': 2                         # 2: V formation
}

# Map Settings
map_settings = {
    'map_dimensions': (1200, 600),
    'num_obs': 3,
    'type_obs': None,                           # Random sqaure obstacles
    'seed_val_obs': 286,                        # Test obstacles location
    'mouse_flag': True,                          # Mouse pointer is turned in a sqaere obstacle
    'max_rect_obs_size': 200                    # maximun Obstacle size
}


# Initialize Environment
env = env_v1.Environment(map_settings, agents_settings)
env.initialize()

goal_pos = (700, 300) #(1000, 200)
path = np.transpose(np.array([agents_settings['start_pos'], goal_pos]))
env.load_path(path)
print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


# plotter = ploting.MultiRealtimePlot(num_plots=3, max_points=50)

while env.running_flag:

    env.env_step(normalize_states=True, training=True)
    # env.visuzalization()
    

    # Debuging States and Rewards functions
    # env.state_angl_between()
    # env.compute_angl_error_reward()

    # env.compute_distance_to_goal()
    # env.compute_distance_reward()


    # Real-Time plot  (It can be slow)
    # value1 = env.state_distance[-1][-1]
    # value2 = env.reward_distance_list[-1][-1]
    # value3 = env.reward_dist_guideline__semiDiscrete_list[-1][-1]

    # plotter.add_points([value1, value2, value3])

    
    

# Plot  
print()
plot_flag = input("Do you wanna Plot? y/n ... ")

if plot_flag == 'y' :

    list_1 = np.array(env.state_distance).squeeze()
    list_2 = np.array(env.reward_distance_list).squeeze()
    list_3 = np.array(env.reward_distance_semiDiscrete_list).squeeze()
    titles = ['Dist. State', 'Lin. Reward', 'Semi-Discrete Reward']

    ploting.plot_list(list_1, list_2, list_3, titles)


    list_1 = np.array(env.state_dist_to_guideline).squeeze()
    list_2 = np.array(env.reward_dist_guideline_list).squeeze()
    list_3 = np.array(env.reward_dist_guideline__semiDiscrete_list).squeeze()
    titles = ['Dist. guide-line State', 'Lin. Reward', 'Semi-Discrete Reward']

    ploting.plot_list(list_1, list_2, list_3, titles)




sys.exit()