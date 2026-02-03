import torch
import torch.nn as nn
import torch.nn.functional as F


#The actor class is the one that decides the action, it is the policy

class Actor(nn.Module):
    
    #the input of the actor is the state and it should output  an action
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear2 = nn.Linear(400,300)
        self.linear3 = nn.Linear(300, action_dim)
 
    #the flow of the network
    def forward(self,state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x   #this is "param_key"
    
#The actor class decides how good the outputed action really is, in an effort to update the policy. 
# Q(s,a)
class Critic(nn.Module):

    #the init funciton differs from the actor because it takes in both the action and state to resolve to the Q-value
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, 1)

    #the flow of the crtic is the same
    def forward(self, state, action):
        # make sure both are [batch, dim] so cat doesnt break later
        if state.dim() > 2:
            state = state.view(state.shape[0], -1)
        if action.dim() > 2:
            action = action.view(action.shape[0], -1)


        x = F.relu(self.linear1(torch.cat([state,action], -1)))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
