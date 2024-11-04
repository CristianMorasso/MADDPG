#!/usr/bin/env python3

#string to run: --env-id simple_spread_v3 --n-ep 250000 --learning-rate 0.0001 --critic-hidden 256 --out-act sigmoid --seed 3 --mod-params Base_250k_seed3
import os
import sys
import copy
import torch
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from MADDPG import MADDPG
from argParser import parse_args
from ma_replay_buffer import MultiAgenReplayBuffer
from pettingzoo.mpe import simple_reference_v3, simple_adversary_v3, simple_push_v3, simple_v3, simple_spread_v3, simple_speaker_listener_v4
#import wandb

 
def add_comm(obs,  actions, type="broadcast",shape=(3,18)):
    """
    Adds other agents communication to the observations.
    Args:
        obs: list of observations
        actions: list of actions
        type: type of communication
        shape: shape of the communication vector
    Returns:
        dest_obs: list of observations with communication added (new observations)
    """
    dest_obs = np.zeros(shape)
    if type== "broadcast":
        for i,o in enumerate(obs):
            dest_obs[i][:o.shape[0]] = o
            dest_obs[i][-3:] = actions.reshape(-1)
        
    elif type == "direct":
        for i, o in enumerate(obs):
            dest_obs[i][:o.shape[0]] = o
            idxs = list(range(len(obs)))
            idxs.remove(i)
            # direct_act = actions[idxs]
            dest_obs[i][-np.dot(*actions[idxs].shape):] = actions[idxs].reshape(-1)

    elif type == "directEsc":
        for i, o in enumerate(obs):
            dest_obs[i][:o.shape[0]] = o
            idxs = list(range(len(obs)))
            idxs.remove(i)
            direct_act = actions[idxs]
            if i == 0:
                agent_spec_obs = np.array([direct_act[0,0], direct_act[1,0]])
            elif i == 1:
                agent_spec_obs = np.array([direct_act[0,0], direct_act[1,1]])
            elif i == 2:
                agent_spec_obs = np.array([direct_act[0,1], direct_act[1,1]])
            dest_obs[i][-agent_spec_obs.reshape(-1).shape[0]:] = agent_spec_obs.reshape(-1)
            
    elif type == "vel_comm" or type == "self_vel_comm" :
        for i, o in enumerate(obs):
            dest_obs[i][:o.shape[0]] = o
            idxs = list(range(len(obs)))
            idxs.remove(i)
            dest_obs[i][-np.dot(*actions[idxs].shape):] = actions[idxs].reshape(-1)

    elif type == "closer_target":
        for i, o in enumerate(obs):
            dest_obs[i][:o.shape[0]] = o
            idxs = list(range(len(obs)))
            idxs.remove(i)
            dest_obs[i][-actions[idxs].reshape(-1).shape[0]:] = actions[idxs].reshape(-1)
            
            
    return dest_obs

def closer_target( target_deltas):
    """
    Takes the target deltas and returns the index of the closest target for each agent.
    Args:
        target_deltas: array of shape (n_agents, n_targets, 2) representing the delta between each agent and each target.
    Returns:
        closest_targets: array of shape (n_agents,) representing the index of the closest target for each agent.
    """
    target_deltas =  target_deltas.reshape(3,3,2)
    return np.argmin(np.linalg.norm(target_deltas, axis=2), axis=1)

def act_to_vel(actions):
    """
    Computes the agent intent according with env dynamics
    """
    axis_delta = np.zeros((actions.shape[0],2))
    
    axis_delta[:,0] += actions[:,1] - actions[:,0]
    axis_delta[:,1] += actions[:,3] - actions[:,2]
    axis_delta *= 5 #sensitivity
    return axis_delta

def dict_to_list(a):
    groups = []
    for item in a:
        groups.append(list(item.values()))
    return  groups

args = parse_args()
#Hyperparams
INFERENCE = False
PRINT_INTERVAL = 5000
MAX_EPISODES = args.n_ep
BATCH_SIZE = args.batch_size
MAX_STEPS = 25
SEED = args.seed
BUFFER_SIZE = args.buffer_size
total_steps = 0
score = -10
best_score = -100
worst_score = 100
low_pick = -1
actor_dims, action_dim = [], []
rewards_history = []
rewards_tot = collections.deque(maxlen=100)


#wandb params
WANDB = False
project_name = "MADDPG"

#storage dirs 
out_dir = "out_csv" #if args.seed == 1 else "seeds_test"
nets_out_dir = "nets" #if args.seed == 1 else "nets/seeds_test"

#env selection from args
params = f"_{args.mod_params}"
env_name = args.env_id
if args.env_id == "simple_spread_v3":
    env_class= simple_spread_v3
if args.env_id == "simple_speaker_listener_v4":
    env_class= simple_speaker_listener_v4
if args.env_id == "simple_adversary_v3":
    env_class= simple_adversary_v3


# if WANDB:
#     wandb.init(
#         project=project_name,
#         name=f"{env_name}_fastUp_{SEED}",
#         group=env_name, 
#         job_type=env_name,
#         reinit=True
#     )

#check dirs
if not os.path.isdir(f"{out_dir}"):
    os.mkdir(f"{out_dir}")
if not os.path.isdir(f"{nets_out_dir}"):
    os.mkdir(f"{nets_out_dir}")
if not os.path.isdir(f"{nets_out_dir}/{env_name}{params}"):
    os.mkdir(f"{nets_out_dir}/{env_name}{params}")

#log file
sys.stdout = open('file_out.txt', 'w')

if INFERENCE:
    env = env_class.parallel_env(max_cycles=100, n_agents=3,continuous_actions=True, render_mode="human")
else:
    env = env_class.parallel_env(continuous_actions=True)
    
obs = env.reset(seed=SEED)
print(env)
print("num agents ", env.num_agents)

#for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


n_agents = env.num_agents

#setup communication specific
comm_type = args.comm_type
if not comm_type is None:
    comm_target = env.num_agents -1
    comm_channels= args.comm_ch
    comm_channels_net = args.comm_ch
else: 
    comm_target = 0
    comm_channels= 0
    comm_channels_net = 0
if comm_type == "broadcast":
    comm_target = 1
if comm_type == "directEsc":
    args.comm_channels = comm_channels
    args.comm_target = comm_target
if comm_type == "vel_comm" or comm_type == "self_vel_comm"or comm_type == "closer_target":
    comm_channels_net=0

#load info from env
for i in range(n_agents):
    actor_dims.append(env.observation_space(env.agents[i]).shape[0]+((comm_channels_net-2 )*comm_target if comm_channels_net > 2 else 0)) 
    action_dim.append(env.action_space(env.agents[i]).shape[0]+comm_channels_net*comm_target)# comm_channels is the comunication channel
critic_dims = sum(actor_dims)

maddpg = MADDPG(actor_dims, critic_dims+sum(action_dim), n_agents, action_dim,chkpt_dir=f"{nets_out_dir}", scenario=f"/{env_name}{params}", seed=SEED, args=args)
if INFERENCE:
    maddpg.load_checkpoint()
memory = [MultiAgenReplayBuffer(critic_dims, actor_dims, action_dim,n_agents, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE,seed = SEED, args =args) for _ in range(args.sub_policy)]

for i in tqdm(range(MAX_EPISODES)):
    #init episode
    k = np.random.randint(0, args.sub_policy)
    step = 0
    obs, info = env.reset(seed=SEED+i)
    obs=list(obs.values())
    done = [False] * n_agents
    rewards_ep_list = []
    if not comm_type is None:
        comm_actions = np.zeros((n_agents,comm_target, 1 if comm_type == "vel_comm" else comm_channels ))
        if comm_type == "broadcast": comm_actions = np.zeros(3)
        obs = add_comm(obs, comm_actions.squeeze(), comm_type, shape=(n_agents,actor_dims[0]))
    
    while not any(done):
        #compute new action
        actions = maddpg.choose_action(obs, k=k, eval=INFERENCE,ep=i, max_ep=MAX_EPISODES, WANDB=WANDB)
        #parsing communication
        if comm_type == "directEsc":
            comm_actions = np.array(actions).squeeze()[:,-comm_channels*comm_target:].reshape(n_agents, comm_target, comm_channels)
            actions_dict = {agent:action[0,:-comm_channels*comm_target].reshape(-1) for agent, action in zip(env.agents, actions)}
        elif comm_type == "direct":
            comm_actions = np.array(actions).squeeze()[:,-comm_channels:]
            actions_dict = {agent:action[0,:-comm_channels].reshape(-1) for agent, action in zip(env.agents, actions)}
        elif comm_type == "vel_comm":
            comm_actions = act_to_vel(np.array(actions).squeeze()[:,1:].reshape(n_agents, 4))
            actions_dict = {agent:action[0].reshape(-1) for agent, action in zip(env.agents, actions)}
        elif comm_type == "self_vel_comm":
            comm_actions = obs[:,:2]
            actions_dict = {agent:action[0].reshape(-1) for agent, action in zip(env.agents, actions)}
        elif  comm_type == "closer_target":
            comm_actions = closer_target(obs[:,4:10])
            actions_dict = {agent:action[0].reshape(-1) for agent, action in zip(env.agents, actions)}
        elif  comm_type == "broadcast":
            
            comm_actions = np.array(actions).squeeze()[:,-1].reshape(n_agents, 1)
            actions_dict = {agent:action[0,:-1].reshape(-1) for agent, action in zip(env.agents, actions)}
        else: 
            actions_dict = {agent:action[0].reshape(-1) for agent, action in zip(env.agents, actions)}
        #step
        data = env.step(actions_dict)
        #save data
        data_processed = dict_to_list(data)
        obs_, rewards, terminations, truncations, info = data_processed
        done = (terminations or truncations)
        if not comm_type is None:
            #create next obs
            obs_ = add_comm(obs_, comm_actions, comm_type, shape=(n_agents,actor_dims[0]))

        if INFERENCE and done:
            env.render(render_mode="human")

        if step >= MAX_STEPS-1 and not INFERENCE:
            done = [True] * n_agents
        #store data
        if not INFERENCE :
            memory[k].store_transition(obs, actions, rewards, obs_, done)

        if args.dial and step % BATCH_SIZE == 0:
            #not really working (learn func are commented in MADDPG.py) 
            if step > 0:
                
                if args.par_sharing:
                    
                    maddpg.learn_dial_par_sharing(memory[0].sample_last_batch())
                else:
                    maddpg.learn_dial(memory)
             
        elif (not INFERENCE) and total_steps % args.learn_delay == 0:
            maddpg.learn(memory)

        actor_state_t0 = copy.deepcopy(obs)
        obs = copy.deepcopy(obs_)
        rewards_ep_list.append(rewards) 
        
        step += 1
        total_steps += 1
    #compute rewards over ep
    rewards_tot.append(np.sum(rewards_ep_list))
    avg_score = np.mean(rewards_tot)
    rewards_history.append(avg_score)
    # if WANDB and i % 100 == 0:    
    #     wandb.log({#'avg_score_adversary':np.mean(np.array(rewards_history)[:,0][0]),\
    #             # 'avg_score_agents':np.mean(np.array(rewards_history)[:,0][0]),\
    #             # 'avg_score_agent1':np.mean(np.array(rewards_history)[:,1][0]),\
    #             'total_rew':avg_score,'episode':i} )
    
    #saving model, good model after some eps
    if i > MAX_EPISODES/20 and avg_score > best_score:
        print("episode: ", i, "avg: ", avg_score, "best: ", best_score)
        best_score = avg_score
        if not INFERENCE:
            print("Saving best model")
            maddpg.save_checkpoint("_best")
    #saving model, after some eps from wrost model
    if low_pick > 0 and ((i - low_pick )% slice_fraq == 0):
        maddpg.save_checkpoint(f"_fraq_{int(avg_score)}_ep_{i}", True)
    #saving model, wrost model afer few eps
    if i > MAX_EPISODES/3 and avg_score < worst_score:
        print("episode: ", i, "avg: ", avg_score, "worst: ", worst_score)
        worst_score = avg_score
        if not INFERENCE:
            print("Saving worst model")
            maddpg.save_checkpoint(f"_worst_{int(worst_score)}_ep_{i}", True)
            low_pick = i
            slice_fraq = (MAX_EPISODES -i )//3 
    #saving model, every 5k eps over 150k
    if i > 150000 and (i % 5000 == 0):
        maddpg.save_checkpoint(f"_fixed_{int(avg_score)}_ep_{i}", True)
    #log file print 
    if i % PRINT_INTERVAL == 0 and i > 0:
        print('episode ', i, 'avg score %.1f' % avg_score)

#saving rewards
reward_history_df = pd.DataFrame(rewards_history)
reward_history_df.to_csv(f"{out_dir}/{env_name}{params}.csv")
print("-----END-----")
sys.stdout.close()
