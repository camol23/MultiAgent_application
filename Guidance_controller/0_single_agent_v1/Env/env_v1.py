import numpy as np
import math

from Env import agents_v1
from Env import env_engine_v1


'''
    Class to define RL enviroment

        1) agents
        2) graphics
        3) RL functions
            3.1) rewards
            3.2) Stop conditions
        4) Step execution


    To Do:
        1) compute_total_reward() : it should be prepare to handle multiple agent rewards (now take just one)

'''

class Environment:
    # def __init__(self, map_dimensions, obstacles, num_agents=1, formation_type=0, mouse_flag=False, reference_path=np.array([])):
    def __init__(self, map_settings, agents_settings, reference_path=np.array([]), training_flag = False):
        
        self.training_flag = training_flag 
        
        # Init. agents
        self.start_pos_agent = agents_settings['start_pos']
        self.num_agents = agents_settings['num_agents']
        self.formation_type = agents_settings['formation_type']
        self.init_pos_agents = []
        self.agents_obj = []


        # Map
        self.env_map = None
        self.map_dimensions = map_settings['map_dimensions']
        self.img_path = '/home/camilo/Documents/SDU/master/Testing_code/pygame_approach/code_test1/Images' 
        self.map_bkg_path = self.img_path + '/blank_backgroun_0.png'
        self.larger_map_side = 0

        self.num_obstacles = map_settings['num_obs']
        self.obstacles_type = map_settings['type_obs']
        self.seed_rand_obs = map_settings['seed_val_obs']
        self.max_rect_obs_size = map_settings['max_rect_obs_size']
        print('seed val now = ', self.seed_rand_obs)
        self.mouse_flag = map_settings['mouse_flag']

        # Sensor
        self.sensor_range = None
        self.proximity_sensor = None

        # Execution variables
        self.dt = 0
        self.last_time = 0
        self.running_flag = True
        self.pause_sim_flag = False


        # Reference Path
        self.reference_path = reference_path


        # Rewards
        self.reward_ang_error_list = []
        self.reward_distance_list = []
        self.reward_total_list = []
        self.reward_dist_guideline_list = []

        # States
        self.state_theta = []                                    # angle between agent and guide line 
        self.state_distance = []                                 # current distance to the goal
        self.state_dist_to_guideline = []                        # distance to the guid line (90 degree angle)

        self.factor_norm_dist_guideline = 1                      # self.agent_init_distance_list/factor_norm_dist_guideline to normalize s_ditst_guideline

        # Agent stop
        self.stop_steps = False                                   # Stop iterations if agent is no alive  

        # Record
        self.steps = 0                                            # Iterations counter (env_septs)                                      
        self.max_steps = 100                                      # limit to stop steps
        self.global_iterations = 0
        self.agent_init_distance_list = []                        # Distance to the goal

        


        

    def initialize(self):

        # Init. Agents
        self.create_agents()

        # Init Map
        self.init_map()
        self.init_sensors()

        # Axiliar vals.
        self.larger_map_side = self.get_max_map_size()


    def init_map(self):
        self.env_map = env_engine_v1.Env_map(self.map_dimensions, self.agents_obj, map_img_path=self.map_bkg_path, mouse_obs_flag=self.mouse_flag)

        self.env_map.max_rect_obs_size = self.max_rect_obs_size
        if self.obstacles_type == 'random':
            self.env_map.random_obstacles(number=self.num_obstacles, seed_val=self.seed_rand_obs) #  (seed 21 / 185 / 285 / 286 ) 88
        elif self.obstacles_type == 'warehouse_0' :
            self.env_map.warehouse_grid(grid_number=0)
        elif self.obstacles_type == 'warehouse_1' :
            self.env_map.warehouse_grid(grid_number=1)
        
        else: 
            self.env_map.random_obstacles(number=self.num_obstacles, seed_val=self.seed_rand_obs) 

        # Reference path to be Draw (indicative)
        self.env_map.path_agent = np.copy(self.reference_path)

    def init_sensors(self):
        self.sensor_range = 250, math.radians(40)
        self.proximity_sensor = env_engine_v1.proximity_sensor(self.sensor_range, self.env_map.map)

    def create_agents(self):

        self.agents_init_pos()

        for id, agent_pos in enumerate(self.init_pos_agents):
            self.agents_obj.append( agents_v1.particle(id, agent_pos, self.training_flag) )

    def agents_init_pos(self):
        '''
            Formation type
                0) random
                1) line 
                2) V formation 
        '''

        if self.formation_type == 0:
            self.init_pos_random()

        if self.formation_type == 1:
            self.init_pos_line()

        if self.formation_type == 2:
            self.init_pos_Vformation()

        
    def init_pos_random(self):
        pass
        

    def init_pos_line(self):
        pass

    def init_pos_Vformation(self):
        x0 = self.start_pos_agent[0]  
        y0 = self.start_pos_agent[1]
        dx = 20

        agent_init_pos = (x0, y0)
        self.init_pos_agents.append(agent_init_pos)
        
        for i in range(1, self.num_agents):
            if i%2 : # Odd 
                agent_init_pos = (x0-(i*dx), y0+(i*dx))
            else:
                agent_init_pos = (x0-(i*dx), y0-(i*dx))

            self.init_pos_agents.append(agent_init_pos)

    def reset_env(self):
        self.steps = 0

        # rewards
        self.reward_ang_error_list.clear()
        self.reward_distance_list.clear()
        self.reward_dist_guideline_list.clear()
        self.reward_total_list.clear()
        

        # States
        self.state_theta.clear()                                    # angle between agent and guide line 
        self.state_distance.clear()                                 # current distance to the goal
        self.state_dist_to_guideline.clear()

        # Agent stop
        self.stop_steps = False  
        ### It could be uncomment (last time)
        # self.env_map.last_time = 0                                # it makes reset dt in env_map.compute_dt() to avoid big space jumps caused by long radients computation



    def env_step(self, normalize_states = True, training=False):        
        self.env_map.read_externals(self.agents_obj)                # Read Keyboard commands
        self.running_flag = self.env_map.running
        self.pause_sim_flag = self.env_map.pause_sim_flag
        
        if not(self.env_map.pause_sim_flag):
            self.env_map.compute_dt()                               # Take sim. tame
            self.env_map.map.blit(self.env_map.map_img, (0, 0))            
            
            for agent in self.agents_obj:
                agent.kinematics(self.env_map.dt)
                
                self.env_map.draw_scene(agent)
                agent.collition_flag = self.env_map.collition_flag

                point_cloud = self.proximity_sensor.sense_obstacles(agent.x, agent.y, agent.heading)
                self.env_map.draw_sensor_data(point_cloud)

                # Enviroment states
                self.is_alive(agent)

            if training:
                self.get_output_step(normalize_states)  #*Should be move inside of the for loop 
            self.env_map.display_update()
            self.steps = self.steps + 1



    def load_path(self, path_wp):
        self.reference_path = np.copy(path_wp)        
        self.env_map.path_agent = self.reference_path

        for agent in self.agents_obj:
            distance_to_goal = agents_v1.distance((agent.x, agent.y), (self.reference_path[0, -1], self.reference_path[1, -1]) )
            self.agent_init_distance_list.append(distance_to_goal)
            
            print("Init. distance to the goal ", agent.id, ' ', distance_to_goal)
            

    def get_output_step(self, normalize_states = False):
        '''
            Compute States and Rewards
        '''
        
        # Compute angent distance to the goal (wp) 
        # self.state_angl_between(normalize=normalize_states)
        # self.compute_angl_error_reward()

        # Compute angle between agent and guide line to the goal 
        self.compute_state_distance_to_goal(normalize_states=normalize_states)
        self.compute_distance_reward(normalize_states=normalize_states)

        # State agent distance to the guide line
        self.compute_state_dist_guideline(normalize_states=normalize_states)
        self.compute_dist_guideline_reward(normalize_states=normalize_states)

        # Compute total reward (Sum all)
        self.compute_total_reward()



    def is_alive(self, agent):
        '''
            The iterations should stop (stop_steps = True)

                1) If the agent collided 
        '''

        # Stopped by Collition
        if agent.collition_flag :
            self.stop_steps = True

        # Stopped by reach the goal
        _, agent.wp_current, goal_reached = agents_v1.follow_path_wp(agent, self.reference_path, get_angl_flag=False)
        if goal_reached :
            self.stop_steps = True

        # print("Is alive ", self.stop_steps)

        if self.steps >= self.max_steps :
            self.stop_steps = True


    def compute_total_reward(self):
        '''
            Sum. all the reward (max val 1.)
            step reward
        '''
        # self.compute_angl_error_reward()
        # self.compute_distance_reward()

        # w_ang_error = 0.5
        w_dist_goal = 0.5
        w_dist_guideline = 0.5


        self.reward_total_list.append( w_dist_guideline* self.reward_dist_guideline_list[-1][-1] +
                                       w_dist_goal* self.reward_distance_list[-1][-1]             )
        
        # w_ang_error* self.reward_ang_error_list[-1]


    def compute_dist_guideline_reward(self, normalize_states=False):

        reward_list = []

        for i in range(0, len(self.agents_obj)):
            max_distance = self.agent_init_distance_list[i]/self.factor_norm_dist_guideline

            if normalize_states:
                current_dist = max_distance*self.state_dist_to_guideline[-1][i]
            else:
                current_dist = self.state_dist_to_guideline[-1][i]

            reward = (max_distance-abs(current_dist))/(max_distance)
            reward_list.append(reward)
            # self.reward_dist_guideline_list.append(reward)
            
            print('Max. Distance guide line ', max_distance)
            # print('dist. guide line reward = ', reward)
        
        # self.reward_dist_guideline_list.append(reward)
        self.reward_dist_guideline_list.append(reward_list)
        print('dist. guide line reward = ', self.reward_dist_guideline_list[-1][-1])


    def compute_angl_error_reward(self, normilize_states = False):
        '''
            (Discarted)
            Angle between the agent to goal line and guide line

                reward [-1, 1] where (1)  := theta = 0   (degrees)
                               where (-1) := theta = 180 (degrees)

                Note:
                    1) Angle turns to degrees
        '''        
        # reward_val = 1
        ang_zero_y = 0.5                                    # theta > zero_ang_y then negative reward 
        f_zero_y = math.log(ang_zero_y)                     # log_x = b
        max_neg_reward = 6.4                                # -(-math.log(180) + math.log(0.3))
        max_post_reward = 4                                 # maximum value when the reward is positive
        # theta_list = self.state_angl_between()
        # theta_deg = math.degrees(abs(theta_list[0])) 
        
        if normilize_states:
            theta_in = (math.pi/2)*self.state_theta[-1]
        else:
            theta_in = self.state_theta[-1]

        theta_deg = math.degrees(abs(theta_in)) 
        # print('state = ', theta_deg)
        if theta_deg == 0 : 
            log_x = -max_post_reward + f_zero_y
        else:
            log_x = math.log(theta_deg)
        
        # reward = ( -log_x + math.log(ang_zero_y) )
        reward = ( -log_x + f_zero_y )

        # Scale values [1, -1]
        if theta_deg >= ang_zero_y :
            reward = reward/max_neg_reward
        else:
            reward = (reward/max_post_reward)
        
        self.reward_ang_error_list.append( reward )
        # print("reward ang error = ", reward, log_x )



    def compute_distance_reward(self, normalize_states=False):
        reward_list = []

        for i in range(0, len(self.agents_obj)):
            max_distance = self.agent_init_distance_list[i]

            if normalize_states:
                # current_dist = max_distance*self.state_distance[i]
                current_dist = max_distance*self.state_distance[-1][i]
            else:
                current_dist = self.state_distance[-1][i]

            reward = (max_distance-current_dist)/(max_distance)
            reward_list.append(reward)
            # self.reward_distance_list.append(reward)
            
            print('Max. Distance ', max_distance)
            # print('distance reward = ', reward)

        self.reward_distance_list.append(reward_list)
        print('distance reward = ', self.reward_distance_list[-1][-1])


    def compute_state_dist_guideline(self, normalize_states=False):
        '''
            
        '''
        
        distances_goal = self.compute_distance_to_goal(normilize = False)
        theta_list = self.state_angl_between()
        dist_to_guideline_list = []
        sign = 1

        self.factor_norm_dist_guideline = 4

        # larger_map_side = self.get_max_map_size()

        for i, _ in enumerate(self.agents_obj):
            theta = theta_list[i]

            if theta < 0 :
                sign = -1
            else:
                sign = 1

            distance = sign* (distances_goal[i])*math.sin(abs(theta))

            if normalize_states :
                # distance = distance/self.larger_map_side
                max_dist = self.agent_init_distance_list[i]/self.factor_norm_dist_guideline
                distance = distance/max_dist

            dist_to_guideline_list.append( distance )

        self.state_dist_to_guideline.append(dist_to_guideline_list)

        print("distance to the guide line ", self.state_dist_to_guideline[-1][-1])


    
    def state_angl_between(self, normalize = False):
        '''
            Theta form 0 to pi
                
                Note: 
                    1) Depends on the side gets (+) or (-) sign
        '''

        goal_point = (self.reference_path[0, -1], self.reference_path[1, -1])
        init_point = (self.reference_path[0, -2], self.reference_path[1, -2])
        theta_list = []
        # self.state_theta.clear()

        for agent in self.agents_obj:
            # Cosine Law
            a = agents_v1.distance( goal_point, (agent.x, agent.y) )
            b = agents_v1.distance( goal_point, (agent.start_pos[0], agent.start_pos[1]) )
            c = agents_v1.distance( (agent.x, agent.y), (agent.start_pos[0], agent.start_pos[1]) )

            # print("denominator angle ", (a**2 + b**2 - c**2), " / ", 2*a*b)
            relation = (a**2 + b**2 - c**2)/(2*a*b)
            if (abs(relation) < 1.1 ) & (abs(relation) > 0.99 ):
                relation = 0.99*(relation/abs(relation))
            theta = math.acos( relation )

            # define side (- := up side in the window)
            m = (goal_point[1] - init_point[1])/(goal_point[0] - init_point[0])
            b = goal_point[1] - goal_point[0]*m

            y_line = (agent.x)*m + b
            if y_line > agent.y :
                theta = (-1)*theta

            if normalize :
                theta = theta/(math.pi/2)


            theta_list.append(theta)
            self.state_theta.append(theta)
            # print("Theta = ", math.degrees(theta) )

            
        return theta_list
    
    def compute_state_distance_to_goal(self, normalize_states = False):

        distances = self.compute_distance_to_goal(normilize = normalize_states)

        self.state_distance.append(distances)
        print('distance to the goal state = ', self.state_distance[-1][-1])



    def compute_distance_to_goal(self, normilize = False):
        goal_point = (self.reference_path[0, -1], self.reference_path[1, -1])
        # self.state_distance.clear()
        distances = []

        for agent in self.agents_obj:
            distance_to_goal = agents_v1.distance((agent.x, agent.y), goal_point)

            if normilize :
                distance_to_goal = distance_to_goal/(self.agent_init_distance_list[agent.id])

            # self.state_distance.append(distance_to_goal)
            distances.append(distance_to_goal)

            # print('distance state = ', distance_to_goal, (agent.x, agent.y))
        
        # self.state_distance.append(distances)
        # print('distance state = ', self.state_distance[-1][-1], (agent.x, agent.y))

        return distances

    
    def apply_actions_left_right(self, action):
        '''
            actions := np.shape(n, 1)
        '''
        for i, agent in enumerate(self.agents_obj):    
            if action[i, 0] :
                agent.move_right()                    
            else:
                agent.move_left()



    def get_diagonal_size(self):
        pass

    def get_max_map_size(self):
        x, y = self.map_dimensions

        if x > y : 
            larger = x
        else:
            larger = y


        return larger 

    def visuzalization(self):
        
        print("Epoch ", self.global_iterations)
        print("Inner iteration ", self.steps)
        # print("rewards ",self.reward_ang_error_list[-1]," ", self.reward_distance_list[-1] )
        print("rewards ", self.reward_distance_list[-1][-1], " ", self.reward_dist_guideline_list[-1][-1] )
        print("reward ", self.reward_total_list[-1])
        # print("States ", self.state_theta[-1], " ", self.state_distance[-1][-1])
        print("States ", self.state_distance[-1][-1], " ", self.state_dist_to_guideline[-1][-1])

        print("--------------------------------------------------------------------")


