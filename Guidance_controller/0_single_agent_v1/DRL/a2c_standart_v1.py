import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Actor-Critic Network with:
            1) Separate actor and critic heads
            2) Common Feature extration (input network)
        
        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)                              # Probability distribution over actions
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                        # Estimate state value
        )
    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state (torch.Tensor): Input state
        
        Returns:
            action_probs (torch.Tensor): Probability distribution over actions
            state_value (torch.Tensor): Estimated state value
        """
        features = self.feature_extractor(state)

        # Actor-Critic
        action_probs = self.actor(features)
        state_value = self.critic(features)

        return action_probs, state_value
    

class ActorCritic_Agent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        """
        Actor-Critic Agent with training logic
        
        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            gamma    (float): Discount factor for future rewards
            learning_rate (float): Learning rate for optimization
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network and optimizer
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma

        # General use
        self.global_steps_T = 0

        # record lists
        self.reward_records = []
        self.pi_loss_record = []
        self.val_loss_record = []
        self.loss_record = []
        self.advantage_record = []
        self.cumulative_reward_record = [0]
        self.TD_target_record = []

    
    def select_action(self, state, vis_flag = False, vis_vals=False):
        """
        Select action based on current policy
        
        Args:
            state (numpy.ndarray): Current environment state
        
        Returns:
            action (int): Selected action
            action_prob (float): Probability of selected action
        """

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # state = torch.FloatTensor(state).to(self.device)
        probs, _ = self.network(state)
        action_probs = probs.detach().cpu().numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)


        if vis_flag :    
            print()
            if vis_vals :
                print("--------- Pick-Sample Function ---------")
                print("Input States = ", state.shape, state)
                print("Actor Output = ", probs.shape, probs)
                print("Squeeze OP.  = ", action_probs.shape, action_probs)
                print("Action       = ", action)
            else:
                print("--------- Pick-Sample Function ---------")
                print("Input States = ", state.shape)
                print("Actor Output = ", probs.shape)
                print("Squeeze OP.  = ", action_probs.shape)
                print("Action       = ", action)

            print("-----------------------------------------")
            print()

        return action, action_probs[action]
    

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single training step using Actor-Critic algorithm
        
        Args:
            state   (numpy): Current state
            action    (int): Action taken
            reward  (float): Reward received
            next_state (numpy): Next state
            done (bool): Episode termination flag
        """

        # Convert to tensors
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        
        # Compute target value
        _, current_value = self.network(state)
        _, next_value = self.network(next_state)
        
        target_value = reward + self.gamma * next_value * (1 - done)
        
        # Compute losses
        critic_loss = F.mse_loss(current_value, target_value.detach())
        
        # Compute advantage
        advantage = (target_value - current_value).detach()
        
        # Compute policy loss with entropy regularization
        action_probs, _ = self.network(state)
        log_probs = torch.log(action_probs[0, action])
        actor_loss = -log_probs * advantage
        
        # Entropy regularization to encourage exploration
        entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
        
        # Total loss
        loss = critic_loss + actor_loss + 0.01 * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_item = loss.item()
        pi_loss_mean = torch.mean(actor_loss).detach().numpy()
        val_loss_mean = torch.mean( torch.squeeze(critic_loss) ).detach().numpy()
        

        total_iter_rewars = reward
        self.cumulative_reward_record.append(self.umulative_reward_record[-1] + reward )
        self.reward_records.append(total_iter_rewars) 
        self.TD_target_record.append(target_value)
        self.pi_loss_record.append( pi_loss_mean )
        self.val_loss_record.append( val_loss_mean )
        self.loss_record.append( loss_item )

        # Visualization
        print()
        print("Run episode {} with rewards {}".format(self.global_steps_T, total_iter_rewars))
        print("     Pi Loss = {}  Val. Loss = {} Loss = {} ".format(pi_loss_mean, val_loss_mean, loss_item))
        print("___________________________________")
        self.global_steps_T += 1

        if vis_flag :
            print("--------- Training Function ---------")
            print("Input States Shape  = ", state.shape)
            print("Input Actions Shape = ", action.shape)
            print("Input Rewards       = ", reward)
            print()
            print("TD Target Shape    = ", target_value)
            print("Actor Output Shape = ", action_probs.shape)
            print("Log( Probs ) Shape = ", log_probs.shape)
        

