import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, name,lr=0.001, hidden_dim=256, chkpt_dir='tmp',out_act_string="", net_type="MLP"):
        super(Actor, self).__init__()
        act_func_dict = {"softmax": lambda x: torch.softmax(x, dim=-1), "sigmoid": lambda x : torch.sigmoid(x),\
		 "tanh": lambda x : torch.tanh(x), "": lambda x:x}
        if net_type == "MLP":
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
        elif net_type == "LSTM":
            # Not tested
            self.lstm = nn.LSTM(state_dim, state_dim, batch_first=True)
            self.fc = nn.Linear(state_dim, action_dim)

        self.net_type = net_type
        self.out_act_func = act_func_dict[out_act_string]
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.h0 = torch.zeros(1, 1, state_dim, requires_grad=False).to(self.device)
        self.c0 = torch.zeros(1, 1, state_dim, requires_grad=False).to(self.device)





    def forward(self, state):
        if self.net_type == "MLP":
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.net_type == "LSTM":
            x, (self.h0, self.c0) = self.lstm(state.view(state.shape[0],1,-1),(self.h0, self.c0))
            self.h0 = self.h0.detach()
            self.c0 = self.c0.detach()
            x = self.fc(x.view(x.shape[0],-1))
        
        x = self.out_act_func(x)
        return x
    
    def save_checkpoint(self, text=""):
        print('... saving checkpoint ...')
        #torch.save(self.state_dict(), self.chkpt_dir+'/'+self.name+'.pth')
        torch.save(self.state_dict(), self.chkpt_dir+'/'+self.name+f'{text}.pth')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.chkpt_dir+'/'+self.name+ '.pth', map_location = torch.device("cpu")))

class Critic(nn.Module):
    def __init__(self, critic_dims, action_dim, n_agents, name,lr=0.001, hidden_dim=256, chkpt_dir='tmp'):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(critic_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.chkpt_dir = chkpt_dir
        self.name = name
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save_checkpoint(self, text):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.chkpt_dir+'/'+self.name+f'{text}.pth')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.chkpt_dir+'/'+self.name+'.pth', map_location = torch.device("cpu")))   

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions,n_agents, agent_idx, chkpt_dir='tmp', \
                 gamma=0.99, tau=0.01, seed = 0, noise_func = None, args = None):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.name = "agent_"+str(agent_idx)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
       
        out_act_string =  args.out_act
        self.actor = [Actor(actor_dims, n_actions, self.name+'_actor_policy'+str(i), chkpt_dir=chkpt_dir,lr=args.learning_rate, hidden_dim=args.actor_hidden, out_act_string = out_act_string, net_type=args.net_type) for i in range(args.sub_policy)]
        self.critic = [Critic(critic_dims, n_actions, n_agents=n_agents, name=self.name+'_critic'+str(i), chkpt_dir=chkpt_dir,lr=args.learning_rate,hidden_dim=args.critic_hidden) for i in range(args.sub_policy)]

        self.target_actor = [Actor(actor_dims, n_actions, self.name+'_target_actor'+str(i), chkpt_dir=chkpt_dir,lr=args.learning_rate,hidden_dim=args.actor_hidden, out_act_string = out_act_string, net_type=args.net_type)for i in range(args.sub_policy)]
        self.target_critic = [Critic(critic_dims, n_actions, n_agents=n_agents, name=self.name+'_target_critic'+str(i), chkpt_dir=chkpt_dir,lr=args.learning_rate,hidden_dim=args.critic_hidden)for i in range(args.sub_policy)]
        if noise_func is None:
            self.noise_func = lambda x: 0.1
        else:
            self.noise_func = noise_func


        self.update_target_networks(tau=self.tau)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for ta, a in zip(self.target_actor, self.actor):
            for target_param, param in zip(ta.parameters(), a.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for tc, c in zip(self.target_critic, self.critic):        
            for target_param, param in zip(tc.parameters(), c.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def get_action(self, state, idx, eval=False, ep=1, max_ep=100, WANDB=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.actor[idx].device)
        action = self.actor[idx](state).cpu().data.numpy()
        noise = self.noise_func(ep, max_ep)
        # if WANDB:
        #     wandb.log({"noise": noise, 'ep_noise': ep})
        if not eval:
            action += np.random.normal(0, noise, size=self.n_actions)
        return action.clip(0,1)

    def save_models(self, text="", act=False):
        for i in range(len(self.actor)):
            self.actor[i].save_checkpoint(text)
            if act: continue
            self.critic[i].save_checkpoint(text)
            self.target_actor[i].save_checkpoint(text)
            self.target_critic[i].save_checkpoint(text)

    def load_models(self):
        for i in range(len(self.actor)): 
            self.actor[i].load_checkpoint()
            self.critic[i].load_checkpoint()
            self.target_actor[i].load_checkpoint()
            self.target_critic[i].load_checkpoint() 



