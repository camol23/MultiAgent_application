'''
    PSO for Path planning (customized)

        Note: 
            1) It depends of pygame fuction to detect line object collition 

'''

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class PSO:
    def __init__(self, map, init_pos, target_pos, pso_params, obs_list):

        self.infinity = 10**6        

        # map properties
        # self.window_w = map.width
        # self.window_h = map.height
        # self.rect_obs_list = map.random_rect_obs_list       # Rectangle shape (x_upper, y_upper, width, height)
        self.window_w, self.window_h = map 

        self.x_init, self.y_init = init_pos
        self.x_target, self.y_target = target_pos

        print(self.x_init, self.y_init)
        print(self.x_target, self.y_target)

        self.obs_rect_list = copy.deepcopy(obs_list)

        # PSO Parametera
        self.iter = pso_params['iterations']                # Max. number of iterations
        self.w = pso_params['w']                            # Inertia weight (exploration & explotation)    
        self.Cp = pso_params['Cp']                          # Personal factor        
        self.Cg = pso_params['Cg']                          # Global factor
        self.rp = 0                                         # Personal random factor [0, 1]
        self.rg = 0                                         # Global random factor [0, 1]
        # self.Vmin = pso_params['Vmin']                    # Minimum particle Velocity  (Max. range of movement)
        # self.Vmax = pso_params['Vmax']                    # Maximum particle Velocity
        self.Vmin = 0                      
        self.Vmax = self.window_h                           # The maximum for y_i coordinate 
        self.num_particles = pso_params['num_particles']    # Number of Paths (first Method)
        self.resolution = pso_params['resolution']          # Numbre of points in the Path

        self.output_path = None
        
        # PSO Variables
        # self.V = np.random.uniform(self.Vmin, self.Vmax, (self.num_particles, self.resolution) )          # Considering the range for the coordinate y_i
        self.V = np.zeros((self.num_particles, self.resolution))
        self.X = np.zeros( (self.num_particles, self.resolution) )                                          # Considering the range for the coordinate y_i        
        self.P = np.zeros((self.num_particles, self.resolution))                                            # The best in the Particle
        self.G = np.zeros((self.num_particles, self.resolution))                                            # The best in the Population (Global)
        self.cost_val = np.zeros(self.num_particles)                                                        # current Cost value for each particle (1. each path)
        self.p_cost = np.zeros(self.num_particles)                                                          # Particle cost val    
        self.p_cost[:] = self.infinity 
        self.g_cost = self.infinity                                                                                # GLobal cost value

        self.cost_penalty = np.zeros(self.num_particles)    # penalty for collition
        self.distance = np.zeros(self.num_particles)        # f1 in the fitness function

        # Fixed points (First method)
        # self.grid_dist = 50                                 # Distance between vertical lines where are placed the y_i points 
        self.x_fixed = np.linspace(self.x_init, self.x_target, self.resolution)
        self.x_fixed = np.float64( np.int32(self.x_fixed) )
        # self.x_fixed = self.x_fixed.reshape( (1, self.resolution) )
        self.diff_xi = np.zeros( (self.num_particles, self.resolution-1) )


    def reset_vals(self):
        # self.V = np.random.uniform(self.Vmin, self.Vmax, (self.num_particles, self.resolution) )          # Considering the range for the coordinate y_i
        self.X = np.random.uniform(self.Vmin, self.Vmax, (self.num_particles, self.resolution) )            # Considering the range for the coordinate y_i        
        self.V[:,:] = 0        
        self.P[:,:] = 0                                                                                     # The best in the Particle
        self.G[:,:] = 0                                                                                     # The best in the Population (Global)
        self.cost_val[:] = self.infinity                                                                           # current Cost value for each particle (1. each path)
        self.p_cost[:] = 0                                                                                  # Particle cost val    
        self.p_cost[:] = self.infinity 
        self.g_cost = self.infinity                                                                                # GLobal cost value

        self.X[:, 0] = self.y_init                                                                          # Agent init. position (x_init, y_init)
        width_lines = np.abs(self.x_fixed[0] - self.x_fixed[1])
        self.diff_xi[:,:] = width_lines

        self.cost_penalty[:] = 0 
        self.distance[:] = 0

        self.convert_rect_coord()



    def convert_rect_coord(self):
        '''
            list = (x_down_left, y_down_left, x_rigth, y_up)
        '''
        # In pygame the axis increase going down in the screen 
        
        for i in range(0, len(self.obs_rect_list)):
            rect_w = self.obs_rect_list[i][0] + self.obs_rect_list[i][2] 
            rect_h = self.obs_rect_list[i][1] + self.obs_rect_list[i][3] 
            self.obs_rect_list[i] = (self.obs_rect_list[i][0], self.obs_rect_list[i][1], rect_w, rect_h)
        

    def particle_collition(self):
        pass

    def validate_points(self):

        for i in range(0, len(self.obs_rect_list)):
            #  y_botton < X > y_up  (X.shape(num_particles, num_points))
            mask_collision = (self.obs_rect_list[i][1] < self.X) & (self.X < self.obs_rect_list[i][3])
            
            # x_bottom < x_fix < x_up
            mask_columns = (self.obs_rect_list[i][0] < self.x_fixed) & (self.x_fixed < self.obs_rect_list[i][2])
            mask_collision = mask_collision & mask_columns

            # Then replaced the Invalid points
            # self.X =  np.logical_not(mask_collision)*self.X + mask_collision*np.random.uniform(0, self.Vmax, (self.num_particles, self.resolution))
            rand_desition = np.random.rand()
            rand_desition = np.round(rand_desition)
            if rand_desition:
                self.X =  np.logical_not(mask_collision)*self.X + mask_collision*np.random.uniform(self.obs_rect_list[i][3], self.window_h, (self.num_particles, self.resolution))
            else:
                self.X =  np.logical_not(mask_collision)*self.X + mask_collision*np.random.uniform(0, self.obs_rect_list[i][1], (self.num_particles, self.resolution))
            

    def collision_inside_obs(self):
        '''
            Detect if the y coordinate of each point is inside of an obstacle
        '''

        for i in range(0, len(self.obs_rect_list)):
            #  y_botton < X > y_up  (X.shape(num_particles, num_points))
            mask_collision = (self.obs_rect_list[i][1] < self.X) & (self.X < self.obs_rect_list[i][3])
            
            # x_bottom < x_fix < x_up
            mask_columns = (self.obs_rect_list[i][0] < self.x_fixed) & (self.x_fixed < self.obs_rect_list[i][2])
            mask_collision = mask_collision & mask_columns

        return mask_collision
    

    def fitness(self):

        # Shortest Path
        diff_yi = self.X[:, 1:] - self.X[:, :-1]
        diff_coord = np.stack( (self.diff_xi, diff_yi), axis=2 )
        norm_points = np.linalg.norm( diff_coord, axis=2 )

        self.distance = np.sum(norm_points, axis=1)
        self.cost_val = np.sum(norm_points, axis=1)

    def fitness_v2(self):
        '''
            It is included the collision as a penalty factor
            (In process)
        '''

        # Shortest Path
        diff_yi = self.X[:, 1:] - self.X[:, :-1]                        # (particles, resolution-1)
        diff_coord = np.stack( (self.diff_xi, diff_yi), axis=2 )        # (particles, resolution-1, (x,y))
        norm_points = np.linalg.norm( diff_coord, axis=2 )              # (particles, resolution-1 )

        self.distance = np.sum(norm_points, axis=1)                     # (particles)

        # Apply penalty to the cost val.
        mask_collision = self.collision_inside_obs()                    # Matrix with one values where a collision is detected
        mask_collision = np.sum(mask_collision, axis=1) > 0             # shape(particles,)
        mask_collision = mask_collision*2

        # When collided the distance is scaled to discard the path 
        self.cost_val = np.logical_not(mask_collision)*self.distance + mask_collision*self.distance


    def pso_compute(self):
        self.reset_vals()

        for i in range(0, self.iter):
            # self.validate_points()

            # Compute Velocity
            r_p = np.random.uniform(0, 1, (self.num_particles, self.resolution))
            r_g = np.random.uniform(0, 1, (self.num_particles, self.resolution))

            self.V = self.w*self.V + \
                    self.Cp*r_p*(self.P - self.X) + \
                    self.Cg*r_g*(self.G - self.X)   

            # Update X 
            self.X = self.X + self.V 

            self.validate_points()
            self.X = np.int32(self.X)                                                                   # Turns the value to integer (no decimal cordinates)
            self.X = np.float64(self.X)
            self.X = np.clip(self.X, 0, self.window_h)

            self.X[:, -1] = self.y_target
            self.X[:, 0] = self.y_init

            # Evaluate Cost value (Updating)
            # self.fitness()                                                                              # Compute current Cost value
            self.fitness_v2()        
            best_cost_mask = self.cost_val < self.p_cost                                                # Compare the current cost against the old value 
            self.p_cost = np.logical_not(best_cost_mask)*self.p_cost + best_cost_mask*self.cost_val
            best_cost_mask = best_cost_mask.reshape( (self.X.shape[0], 1) )
            # print(best_cost_mask.shape)
            self.P = np.logical_not(best_cost_mask)*self.P + best_cost_mask*self.X                      # Save old value if current > , and save current when current <
            

            best_index = np.argmin(self.cost_val)                                                       # Take the index of the best particle based on the cost function
            best_current_g_cost = np.min(self.cost_val)
            if best_current_g_cost < self.g_cost :                                                      # If the best current val. is better than the ald global best, then Update 
                self.G[:] = self.X[best_index, :]
                self.g_cost = best_current_g_cost


        # print("Last global best cost value = ", self.g_cost)
        self.output_path = np.stack( (self.x_fixed, self.G[0, :]) )
    


    def visualization(self):
        
        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1) 
        ax.plot(self.x_fixed, self.G[0, :], color ='tab:blue') 
        ax.scatter(self.x_fixed, self.G[0, :], c='red', alpha=0.5, linewidths=0.5)
        
        # for xi_dot in self.x_fixed:
            # ax.plot( [xi_dot, xi_dot], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 
        for i in range(1, ( self.x_fixed.shape[0] )-1 ):
            ax.plot( [self.x_fixed[i], self.x_fixed[i]], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 
            

        # Draw obstacles
        for i in range(0, len(self.obs_rect_list)):
            rect_w = self.obs_rect_list[i][2]
            rect_w = abs(rect_w - self.obs_rect_list[i][0])

            rect_h = self.obs_rect_list[i][3]
            rect_h = abs(rect_h - self.obs_rect_list[i][1])

            x_botton = self.obs_rect_list[i][0]
            y_botton = self.obs_rect_list[i][1]

            ax.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor='black'))
        
        ax.set_title('Global best Path') 
        plt.show() 


    def visualization_all(self):
        
        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1) 
        
        for i in range(0, self.num_particles):
            ax.plot(self.x_fixed, self.P[i, :], color ='tab:blue', alpha=0.5, linestyle='dashed') 
            ax.scatter(self.x_fixed, self.P[i, :], c='red', alpha=0.5, linewidths=0.5)
        
        for i in range(1, ( self.x_fixed.shape[0] )-1 ):
            ax.plot( [self.x_fixed[i], self.x_fixed[i]], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 

        ax.plot(self.x_fixed, self.G[0, :], color ='tab:red') 
        ax.scatter(self.x_fixed, self.G[0, :], c='red', alpha=0.5, linewidths=0.5)
        

        # Draw obstacles
        for i in range(0, len(self.obs_rect_list)):
            rect_w = self.obs_rect_list[i][2]
            rect_w = abs(rect_w - self.obs_rect_list[i][0])

            rect_h = self.obs_rect_list[i][3]
            rect_h = abs(rect_h - self.obs_rect_list[i][1])

            x_botton = self.obs_rect_list[i][0]
            y_botton = self.obs_rect_list[i][1]

            ax.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor='black'))


        sorted_cost = np.sort(self.p_cost)
        print("Min. cost val = ", np.min(self.p_cost))
        print("Max. cost val = ", np.max(self.p_cost[0]))

        ax.set_title('All Paths - Shortest '+ str(int(sorted_cost[0])) + " - second = "+ str(int(sorted_cost[1])) ) 
        
        plt.show() 


