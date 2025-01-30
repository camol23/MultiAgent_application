import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

'''
    * V_1.2 : It's made to apply some possible corrections
                over V_1.0  

    Action Space: 
                    1) Move to the ritgh (5 degrees)
                    2) Move to the left (5 degrees) 

    Observation Space:
                    1) Heading Error (angle) 
                    2) Distance to the goal 

    Reward Space:
                    1) Inverse proportional to the Error
                    2) Inverse proportional to the Distance

'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Policy function
class ActorNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        logits = self.output(outs)
        return logits


# Value V(s) function
class ValueNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value



class drl_model:
    def __init__(self):

        self.actor_model = ActorNet().to(device)
        self.critic_model = ValueNet().to(device)

        self.load_optimizers()

        # general
        self.stop_condition_flag = 0

        # record lists
        self.reward_records = []
        self.global_steps_T = 0
        self.TD_target_record = None



    # pick up action with above distribution policy_pi
    def pick_sample(self, s_batch):
        with torch.no_grad():
            #   --> size : (1, 4)
            # s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
            # print(s_batch)
            # Get logits from state
            #   --> size : (1, 2)
            logits = self.actor_model(s_batch)
            # print("Porbs = ", logits)
            #   --> size : (2)
            logits = logits.squeeze(dim=0)
            # From logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Pick up action's sample
            a = torch.multinomial(probs, num_samples=1)
            # Return
            # return a.tolist()[0]
            return a
        


    #### Training 
    def load_optimizers(self):
        self.opt_critic = torch.optim.AdamW(self.critic_model.parameters(), lr=0.001)
        self.opt_actor = torch.optim.AdamW(self.actor_model.parameters(), lr=0.001)


    def TD_target_1(rewards_list, gamma, reverse=False):
        '''
            
            reverse:
                a) True: cum_rewards goes from r_[t_last] to r_[0] 
                b) True: cum_rewards goes from r_[0] to r_[t_last] 
        '''

        cum_rewards = np.zeros_like(rewards_list)
        reward_len = len(rewards_list)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = rewards_list[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)

        if reverse :
            cum_rewards = np.flip(cum_rewards)

        return cum_rewards
    

    def TD_target_2(self, states, rewards_list, gamma):
        '''
            Use Critic network to compute V( S_[t+1] )
        '''

        values = self.critic_model(states).detach()

        # G = r_[t] + gamma * V(S_[t+1])
        td_target = np.zeros_like(rewards_list)
        td_target[0:-1] = rewards_list[0:-1] + gamma*values[1:] 

        # V(S_[last+1]) = 0
        td_target[-1] = rewards_list[-1] 

        return td_target



    def TD_target_3(self, values, rewards_list, gamma):
        '''
            Extract V( S_[t+1] ) from V(S_[t]) vector

                values: V( S_[t+1] )
        '''
        
        values_next = values.detach()
        
        # G = r_[t] + gamma * V(S_[t+1])
        td_target = np.zeros_like(rewards_list)
        td_target[0:-1] = rewards_list[0:-1] + gamma*values_next[1:] 
        
        # V(S_[last+1]) = 0
        td_target[-1] = rewards_list[-1] 

        return td_target


    def training_a2c(self, states_list, actions_list, rewards_list):
        '''
            Compute Losses, gradients, and update weigths

            Note:
                1) One Iteration
        '''

        gamma = 0.99                                                                 # Discount factor
        states = torch.tensor(states_list, dtype=torch.float).to(device)
        actions = torch.tensor(actions_list, dtype=torch.int64).to(device)
        
        # Get cumulative rewards (Return)

        td_target = self.TD_target_1(rewards_list, gamma, reverse=False)              # Using just rewards
        # td_target = self.TD_target_2(states, rewards_list, gamma)                   # Using Critic network
        td_target = torch.tensor(td_target, dtype=torch.float).to(device)
        

        # Compute Values
        values = self.critic_model(states)
        values = values.squeeze(dim=1)

        # Optimize value loss (Critic)
        vf_loss = F.mse_loss(
            values,
            td_target,
            reduction="none")
        
        self.opt_critic.zero_grad()
        vf_loss.sum().backward()
        self.opt_critic.step()


        # Optimize policy loss (Actor)
        with torch.no_grad():
            values = self.critic_model(states)

        advantages = (td_target - values).detach()
        # advantages = td_target - values
        
        logits = self.actor_model(states)
        logits = torch.squeeze(logits)
        actions = torch.squeeze(actions)
        print(actions.shape)
        print(logits.shape)
        log_probs = -F.cross_entropy(logits, actions, reduction="none")
        pi_loss = -log_probs * advantages
        
        self.opt_actor.zero_grad()
        pi_loss.sum().backward()
        self.opt_actor.step()

        # Output total rewards in episode (max 500)
        # print("Run episode{} with rewards {}".format(i, sum(rewards_list)), end="\r")

        total_iter_rewars = sum(rewards_list)
        print("Run episode {} with rewards {}".format(self.global_steps_T, total_iter_rewars))
        
        self.reward_records.append(total_iter_rewars) # by epoch
        self.global_steps_T += 1

        self.TD_target_record = td_target

        # print(self.global_steps_T)

        # stop if reward mean > 475.0
        # self.stop_condition()

    
    def stop_condition(self):
        # if np.average(self.reward_records[-50:]) > 475.0:
        #     self.stop_condition_flag = 1
        pass
        

    def preprossed_states():
        pass


    def plot_results(self):
        
        plt.plot(self.reward_records)
        plt.title("Sum. rewards by epoch - Epochs " + str(self.global_steps_T) )
        plt.show() 
