import numpy as np
import torch

class MultiAgenReplayBuffer:
    def __init__(self, critic_dims, actor_dims, n_actions, n_agents, buffer_size, batch_size = 512, seed = 0, args = None ):
        # self.critic_dims = critic_dims
        self.actor_dims =  actor_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pointer = 0

        self.state_memory = np.zeros((self.buffer_size, critic_dims ))
        self.new_state_memory = np.zeros((self.buffer_size, critic_dims))
        self.reward_memory = np.zeros((self.buffer_size, n_agents))
        self.terminal_memory = np.zeros((self.buffer_size, n_agents), dtype = np.bool_)
        # self.action_memory = np.zeros((self.buffer_size, n_agents, actor_dims))

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        if False:#self.args.par_sharing:
            self.actor_state_memory.append(np.zeros((self.buffer_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.buffer_size, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.buffer_size, self.n_actions[i])))
        else:
            for i in range(self.n_agents):
                self.actor_state_memory.append(np.zeros((self.buffer_size, self.actor_dims[i])))
                self.actor_new_state_memory.append(np.zeros((self.buffer_size, self.actor_dims[i])))
                self.actor_action_memory.append(np.zeros((self.buffer_size, self.n_actions[i])))


    def store_transition(self, state, action, reward, new_state, done):
        index = self.pointer % self.buffer_size
        # state = np.concatenate(state, axis = 0)
        # new_state = np.concatenate(new_state, axis = 0)
        # action = np.concatenate(action, axis = 0)

        self.state_memory[index] = np.concatenate(state)
        self.new_state_memory[index] = np.concatenate(new_state)
        self.reward_memory[index] = np.array(reward)
        self.terminal_memory[index] = np.array(done)

        # self.action_memory[index] = action

        if False:#self.args.par_sharing:
            self.actor_state_memory[0][index] = state[i]
            self.actor_new_state_memory[0][index] = new_state[i]
            self.actor_action_memory[0][index] = action[i]
        else:
            for i in range(self.n_agents):
                self.actor_state_memory[i][index] = state[i]
                self.actor_new_state_memory[i][index] = new_state[i]
                self.actor_action_memory[i][index] = action[i]

        self.pointer += 1

    def sample_buffer(self):
        max_memory = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_memory, self.batch_size, replace = False)

        state = self.state_memory[batch]
        new_state = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_state_t0 = [self.actor_state_memory[i][batch-1] for i in range(self.n_agents)]
        actor_state = [self.actor_state_memory[i][batch] for i in range(self.n_agents)]
        actor_new_state = [self.actor_new_state_memory[i][batch] for i in range(self.n_agents)]
        actor_action = [self.actor_action_memory[i][batch] for i in range(self.n_agents)]

        return state, new_state, reward, terminal, actor_state, actor_new_state, actor_action,actor_state_t0
    
    def sample_last_batch(self):
        max_memory = min(self.pointer, self.buffer_size)
        batch = self.batch_size

        state = self.state_memory[-batch:]
        new_state = self.new_state_memory[-batch:]
        reward = self.reward_memory[-batch:]
        terminal = self.terminal_memory[-batch:]

        actor_state_t0 = [self.actor_state_memory[i][-batch-1:] for i in range(self.n_agents)]
        actor_state = [self.actor_state_memory[i][-batch:] for i in range(self.n_agents)]
        actor_new_state = [self.actor_new_state_memory[i][-batch:] for i in range(self.n_agents)]
        actor_action = [self.actor_action_memory[i][-batch:] for i in range(self.n_agents)]

        return state, new_state, reward, terminal, actor_state, actor_new_state, actor_action,actor_state_t0
    
    def ready(self):
        if self.pointer >= self.batch_size:
            return True
        return False
        




               
