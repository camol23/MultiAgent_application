import sys
import numpy as np

from Env import env_v1


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
    'mouse_flag': True                          # Mouse pointer is turned in a sqaere obstacle
}


# Initialize Environment
env = env_v1.Environment(map_settings, agents_settings)
env.initialize()

goal_pos = (700, 300) #(1000, 200)
path = np.transpose(np.array([agents_settings['start_pos'], goal_pos]))
env.load_path(path)
print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


while env.running_flag:

    env.env_step(normalize_states=True, training=True)
    env.visuzalization()
    

    # Debuging States and Rewards functions
    # env.state_angl_between()
    # env.compute_angl_error_reward()

    # env.compute_distance_to_goal()
    # env.compute_distance_reward()


sys.exit()