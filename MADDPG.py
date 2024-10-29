import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from torch.nn.utils import clip_grad_norm_
# from gym import make_env
from actor_critic_nets import *
 	
from ma_replay_buffer import MultiAgenReplayBuffer
from pettingzoo.mpe import simple_adversary_v3


def noise_mul_func(ep, n_ep):
    ep_temp = ep/(n_ep*2/5000)
    return round((98.8/100)**((ep_temp)/4), 2)
class MADDPG:
    def __init__(self, actor_dims, critic_dims,  n_agents,n_actions, scenario="simple", gamma=0.99, tau=0.01, chkpt_dir='tmp', seed =0, args = None):
        self.gamma = args.gamma
        self.tau = args.tau
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.agents = []
        self.args = args
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if args.par_sharing:
            agent = Agent(actor_dims[0], critic_dims, n_actions[0], n_agents, 0, chkpt_dir=chkpt_dir+scenario, gamma=self.gamma, tau=self.tau, seed=seed, noise_func = noise_mul_func, args=self.args)
            for i in range(self.n_agents):
                # agent = Agent(actor_dims[i], critic_dims, n_actions[i], n_agents, i, chkpt_dir=chkpt_dir+scenario, gamma=self.gamma, tau=self.tau, seed=seed, noise_func = noise_mul_func, args=self.args)
                self.agents.append(agent)
        else: 
            for i in range(self.n_agents):
                agent = Agent(actor_dims[i], critic_dims, n_actions[i], n_agents, i, chkpt_dir=chkpt_dir+scenario, gamma=self.gamma, tau=self.tau, seed=seed, noise_func = noise_mul_func, args=self.args)
                self.agents.append(agent)
        self.update = 0
    
    def save_checkpoint(self, text = "", act = False):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models(text, act)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    def choose_action(self, raw_obs, k, eval=False, ep=1, max_ep=100, WANDB=False):
        actions = []
        if self.args.par_sharing:
            for agent_idx in range(len(self.agents)):
                action = self.agents[0].get_action(raw_obs[agent_idx].reshape(1, -1),k, eval, ep, max_ep, WANDB)
                actions.append(action)
        else:
            for agent_idx, agent in enumerate(self.agents):
                action = agent.get_action(raw_obs[agent_idx].reshape(1, -1),k, eval, ep, max_ep, WANDB)
                actions.append(action)
        return actions
    
    #TO BE TESTED
    # def learn_dial_par_sharing(self, memory_list):
    #     for k in range(self.args.sub_policy):
    #         #NO REPLAY BUFF
    #         state, new_state, reward, terminal, actor_action, actor_state_t0 = memory_list
    #         # obs, obs_, rewards, done,actions, actor_state_t0
    #         actor_state = [s.reshape(1,-1) for s in state]
    #         actor_new_state = [s_n.reshape(1,-1) for s_n in new_state]
    #         actor_state_t0 = [ast0.reshape(1,-1) for ast0 in actor_state_t0]
            
    #         device = self.agents[0].actor[k].device

    #         state = torch.tensor(np.array(state).reshape(1,-1), dtype=torch.float32).to(device)
    #         new_state = torch.tensor(np.array(new_state).reshape(1,-1), dtype=torch.float32).to(device)
    #         reward = torch.tensor(np.array(reward).reshape(1,-1), dtype=torch.float32).to(device)
    #         terminal = torch.tensor(np.array(terminal).reshape(1,-1), dtype=torch.float32).to(device)
    #         # target value y^j
    #         # critic update
    #         # memory = memory_list[k]
    #         # if not memory.ready():
    #         #     return
    #         # state, new_state, reward, terminal, actor_state, actor_new_state, actor_action, actor_state_t0 = memory.sample_buffer()
    #         # device = self.agents[0].actor[k].device

    #         # state = torch.tensor(state, dtype=torch.float32).to(device)
    #         # new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
    #         # reward = torch.tensor(reward, dtype=torch.float32).to(device)
    #         # terminal = torch.tensor(terminal.reshape(1,-1), dtype=torch.float32).to(device)
    #         #we just need one critic
            

    #         with torch.no_grad():
    #             target_action = torch.cat([self.agents[j].target_actor[k](torch.tensor(actor_new_state[j], dtype=torch.float32, device=device)) for j in range(self.n_agents)], dim=1)
    #             target_critic_value =  self.agents[0].target_critic[k](new_state, target_action).view(-1)
    #             # reward[i] = reward[i].view(-1, 1)
    #             next_q_value = reward[:,0].view(-1) +(1 - terminal[:, 0]) * self.gamma * target_critic_value
                    
    #         self.agents[0].critic[k].optimizer.zero_grad()
    #         old_actions = torch.tensor(np.concatenate(actor_action, axis=1), dtype=torch.float32, device=device)    
    #         q_values = self.agents[0].critic[k](state, old_actions).view(-1)
            
    #         loss = F.mse_loss(q_values, next_q_value)
                
    #         loss.backward(retain_graph=False)
    #         clip_grad_norm_(parameters=self.agents[0].critic[k].get_params(), max_norm=10)
    #         self.agents[0].critic[k].optimizer.step()

    #         self.agents[0].actor[k].optimizer.zero_grad()
    #         # actor_loss = 0
    #         # for i in range(self.n_agents):
    #         #     # actor update
    #         #     policy_action = torch.cat([self.agents[j].actor[k](torch.tensor(actor_state[j], dtype=torch.float32, device=device)) for j in range(self.n_agents)], dim=1)
    #         #     actor_loss += -self.agents[i].critic[k](state, policy_action).mean()

    #         policy_action = torch.cat([self.agents[0].actor[k](torch.tensor(actor_state[j], dtype=torch.float32, device=device)) for j in range(self.n_agents)], dim=1)
    #         actor_loss = -self.agents[0].critic[k](state, policy_action).mean()
    #         actor_loss.backward()
    #         clip_grad_norm_(parameters=self.agents[0].actor[k].get_params(), max_norm=10)
    #         self.agents[0].actor[k].optimizer.step()
            
    #             # target update
    #         for i in range(self.n_agents):
    #             if self.update % self.args.update_delay == 0:
    #                 self.agents[i].update_target_networks(self.tau)
    #     self.update+=1
    def learn(self, memory_list):
        for k in range(self.args.sub_policy):
            memory = memory_list[k]
            if not memory.ready():
                return
            state, new_state, reward, terminal, actor_state, actor_new_state, actor_action, actor_state_t0 = memory.sample_buffer()
            device = self.agents[0].actor[k].device

            state = torch.tensor(state, dtype=torch.float32).to(device)
            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).to(device)
            # target value y^j
            # critic update
            for i in range(self.n_agents):
                with torch.no_grad():
                    target_action = torch.cat([self.agents[j].target_actor[k](torch.tensor(actor_new_state[j], dtype=torch.float32, device=device)) for j in range(self.n_agents)], dim=1)
                    target_critic_value =  self.agents[i].target_critic[k](new_state, target_action).view(-1)
                    # reward[i] = reward[i].view(-1, 1)
                    next_q_value = reward[:,i].view(-1) +torch.tensor(1 - terminal[:, i], dtype=torch.float32, device=device) * self.gamma * target_critic_value
                        
                self.agents[i].critic[k].optimizer.zero_grad()
                old_actions = torch.tensor(np.concatenate(actor_action, axis=1), dtype=torch.float32, device=device)    
                q_values = self.agents[i].critic[k](state, old_actions).view(-1)
                
                loss = F.mse_loss(q_values, next_q_value)
                
                loss.backward()
                self.agents[i].critic[k].optimizer.step()

                
                policy_action = torch.cat([self.agents[j].actor[k](torch.tensor(actor_state[j], dtype=torch.float32, device=device, requires_grad=True)) for j in range(self.n_agents)], dim=1)
                actor_loss = -self.agents[i].critic[k](state, policy_action).mean()
                self.agents[i].actor[k].optimizer.zero_grad()
                actor_loss.backward()
                self.agents[i].actor[k].optimizer.step()
            
                # target update
                if self.update % self.args.update_delay == 0:
                    self.agents[i].update_target_networks(self.tau)
        self.update+=1
    def learn_dial(self, memory):
        for k in range(self.args.sub_policy):
            # memory = memory_list[k]
            # if not memory.ready():
            #     return
            state, new_state, reward, terminal, actor_action, actor_state_t0 = memory
            actor_state = [s.reshape(1,-1) for s in state]
            actor_new_state = [s_n.reshape(1,-1) for s_n in new_state]
            actor_state_t0 = [ast0.reshape(1,-1) for ast0 in actor_state_t0]
            
            device = self.agents[0].actor[k].device

            state = torch.tensor(state.reshape(1,-1), dtype=torch.float32).to(device)
            new_state = torch.tensor(new_state.reshape(1,-1), dtype=torch.float32).to(device)
            reward = torch.tensor(np.array(reward).reshape(1,-1), dtype=torch.float32).to(device)
            terminal = torch.tensor(np.array(terminal).reshape(1,-1), dtype=torch.float32).to(device)

            
            

            # target value y^j
            # critic update
            for i in range(self.n_agents):
                with torch.no_grad():
                    target_action = torch.cat([self.agents[j].target_actor[k](torch.tensor(actor_new_state[j], dtype=torch.float32, device=device)) for j in range(self.n_agents)], dim=1)
                    target_critic_value =  self.agents[i].target_critic[k](new_state, target_action).view(-1)
                    # reward[i] = reward[i].view(-1, 1)
                    next_q_value = reward[:,i].view(-1) +(1 - terminal[:, i]) * self.gamma * target_critic_value
                        
                self.agents[i].critic[k].optimizer.zero_grad()
                old_actions = torch.tensor(np.concatenate(actor_action, axis=1), dtype=torch.float32, device=device)    
                q_values = self.agents[i].critic[k](state, old_actions).view(-1)
                
                loss = F.mse_loss(q_values, next_q_value)
                
                loss.backward()
                self.agents[i].critic[k].optimizer.step()

                # actor update
                
                comm_shape = np.zeros((self.args.comm_channels,self.args.comm_target)).squeeze().shape
                comm_shape = comm_shape[0] if len(comm_shape) == 1 else comm_shape
                # dest_obs = torch.zeros(actor_state[i].shape)
                idxs = list(range(self.n_agents))
                idxs.remove(i)

                #need to recreate the communication values
                for ii in range(self.n_agents):
                    self.agents[ii].actor[k].optimizer.zero_grad()
                
                #shape (n_other_agent, batch_size, comm)
                other_agent_policy = torch.stack([self.agents[j].actor[k](torch.tensor(actor_state_t0[j], dtype=torch.float32, device=device)) for j in idxs])
                actions = other_agent_policy[:,:,-self.args.comm_channels*self.args.comm_target:]
                
                if i == 0:
                    agent_spec_obs = torch.cat([actions[0,:,0], actions[1,:,0]])
                elif i == 1:
                    agent_spec_obs = torch.cat([actions[0,:,0], actions[1,:,1]])
                elif i == 2:
                    agent_spec_obs = torch.cat([actions[0,:,1], actions[1,:,1]])
                # dest_obs[:,-self.args.comm_channels*self.args.comm_target:] = agent_spec_obs.reshape(-1,self.args.comm_channels*self.args.comm_target)
                
                tensor_actor_state = [torch.tensor(act_s, dtype=torch.float32, device=device) for act_s in actor_state]
                
                
                tensor_actor_state[i][:,-self.args.comm_channels*self.args.comm_target:] =agent_spec_obs.reshape(-1,self.args.comm_channels*self.args.comm_target)

                policy_action = torch.cat([self.agents[j].actor[k](tensor_actor_state[j]) for j in range(self.n_agents)], dim=1)
                actor_loss = -self.agents[i].critic[k](state, policy_action).mean()
                actor_loss.backward()
                
                for ii in range(self.n_agents):
                    self.agents[ii].actor[k].optimizer.step()


                
                # target update
                if self.update % self.args.update_delay == 0:
                    self.agents[i].update_target_networks(self.tau)
        self.update+=1

    def obs_list_to_state_vector(self, obs):
        state = np.array([])
        for s in obs:
            state = np.concatenate([state, s])
        return state

    def reset(self):
        for agent in self.agents:
            agent.reset()


