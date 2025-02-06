import sys
import numpy as np

from Env import env_v1
from DRL import a2c_standart_v1


# Agents Settings
agents_settings = {
               # (x0, y0)
    'start_pos': (100, 550),                    #(50, 550),
    'num_agents': 1,
    'formation_type': 2                         # 2: V formation
}

# Map Settings
map_settings = {
    'map_dimensions': (1200, 600),
    'num_obs': 0,
    'type_obs': 'random',                       # Random sqaure obstacles
    'seed_val_obs': 286,                        # Test obstacles location
    'mouse_flag': True,                         # Mouse pointer is turned in a sqaere obstacle
    'max_rect_obs_size': 200                    # maximun Obstacle size
}


# Initialize Environment
env = env_v1.Environment(map_settings, agents_settings, training_flag=True)
env.initialize()

goal_pos = (700, 300) #(1000, 200)
path = np.transpose(np.array([agents_settings['start_pos'], goal_pos]))
env.load_path(path)
print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


# DRL model
state_dim = 2
action_dim = 2
model = a2c_standart_v1.ActorCritic_Agent(state_dim=state_dim, action_dim=action_dim)

# Training Parameters
num_iterations = 2000
env.max_steps = 10



for i in range(0, num_iterations):

    # *It shoud be in a reset function*
    states = np.zeros((1, 2))
    next_state = np.zeros((1, 2))
    rewards = np.zeros((1, 1))
    rewards_steps = np.zeros((1, 1))
    # actions = np.zeros((1, 1))
    actions_steps = np.zeros((1, 1))

    env.reset_env()
    env.get_output_step(normalize_states = True)
    states[0, 0] = env.state_distance[-1][-1]
    states[0, 1] = env.state_dist_to_guideline[-1][-1]

    done = False
    env.global_iterations = i

    while not done :
        
        actions, _ = model.select_action(states, vis_flag=True, vis_vals=True)
        env.apply_one_action_left_right(actions)
        env.env_step(normalize_states=True, training=True)

        next_state[0, 0] = env.state_distance[-1][-1]
        next_state[0, 1] = env.state_dist_to_guideline[-1][-1]

        rewards[0, 0] = env.reward_total_list[-1]
        
        done = env.stop_steps
        env.visuzalization()

        model.train_step(states, actions, rewards, next_state, done, vis_flag=True)

        # Update
        states = next_state
    


    # Stop by Keyboard command "Q"
    if not(env.running_flag):
        print("Stopped by user")
        break
        


# model.plot_rewards()
model.plot_training()
sys.exit()