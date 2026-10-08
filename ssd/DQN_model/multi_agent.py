from DQN_model.single_agent import dqn

import torch
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

class mult_agent_dqn:
    def __init__(self,device,env, epsilon_decay, conv_filters, 
                 fc_filters, lr, replay_capacity, batch_size,gamma,
                 grad_clip,num_episode,tau,episode_len,log_file,
                 update_freq = 1, use_cnn = False, debug = False,
                 epsilon_decay_style="LINEAR",):
        self.device = device
        obs, info = env.reset()
        self.agent_ids = list(env.agents.keys())
        self.sorted_agent_ids = sorted(self.agent_ids)
        self.env = env
        self.dqn_lst = {}
        self.update_freq = update_freq
        img_shape = 0
        # rep_shape = 0
        # self.extra_obs = extra_obs
        # if self.extra_obs:
        #     img_shape = obs[self.agent_ids[0]]["curr_obs"].shape
        #     # rep_shape = obs[self.agent_ids[0]]["agent_reputations"].shape[0]
        # else:
        print("obs is", obs[self.agent_ids[0]].keys())
        img_shape = obs[self.agent_ids[0]]["curr_obs"].shape
        action_shape = env.action_space.n
        self.num_episodes = num_episode
        # self.executor = ThreadPoolExecutor(max_workers=len(self.agent_ids))

        self.episode_len = episode_len
        self.log_file = log_file
        self.log = []
        with open(self.log_file, 'w', newline='') as f:
            f.write("")

        for id in self.agent_ids:
            mod = dqn(self.env, device = device,epsilon_decay=epsilon_decay, 
                      update_freq = self.update_freq, img_shape=img_shape, 
                      action_shape=action_shape,conv_filters=conv_filters,
                      fc_filters=fc_filters,lr=lr,replay_capacity=replay_capacity,
                      batch_size=batch_size,gamma=gamma,grad_clip=grad_clip,
                      num_episode=num_episode,tau=tau, episode_len=episode_len,
                      log_file=None, use_cnn=use_cnn,debug=debug,
                      epsilon_decay_style=epsilon_decay_style,
                      use_amp=True)
            self.dqn_lst[id] = mod

    # ------------------------------------------------------------------ #
    # action selection
    # ------------------------------------------------------------------ #
    def multi_agent_select_action(self):
        return {
        id: self.dqn_lst[id].select_action().item()
        for id in self.sorted_agent_ids
        }
     
    # ------------------------------------------------------------------ #
    # state setup
    # ------------------------------------------------------------------ #
    def multi_agent_set_state(self, state):
        img_list = [state[id]["curr_obs"] for id in self.sorted_agent_ids]
        img_batch = (
            torch.from_numpy(np.stack(img_list))
            .to(self.device, non_blocking=True)
            .float()
        )
        for i, id in enumerate(self.sorted_agent_ids):
            self.dqn_lst[id].set_state(img_batch[i:i+1]) 


    def train_agent_step(self, agent_id, t, action, obs, rew, term, trunc):
        """Train single agent - can be parallelized"""
        self.dqn_lst[agent_id].train_one_step(t, action, obs, rew, term, trunc)
    

    def process_multi_agent_obs(self,observation, reward, terminated, truncated,action):
        img_list = []
        # rep_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        act_list = []
        for id in sorted(self.agent_ids):
            img = observation[id]["curr_obs"]
            img_list.append(img)
            reward_list.append(reward[id])
            terminated_list.append(terminated[id])
            truncated_list.append(truncated[id])
            act_list.append(torch.tensor([action[id]]))
        img_batch = (
            torch.from_numpy(np.stack(img_list))
            .to(self.device, non_blocking=True)
            .float()
        )        
        reward_batch = torch.as_tensor(
            reward_list, dtype=torch.float32, device=self.device
        )        
        truncated_batch = torch.tensor(np.stack(truncated_list), dtype=torch.float32, device=self.device)
        terminated_batch = torch.tensor(np.stack(terminated_list), dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(
            act_list, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        return img_batch,reward_batch,truncated_batch, terminated_batch, action_batch
        
    # ------------------------------------------------------------------ #
    # internal helper used by the executor
    # ------------------------------------------------------------------ #
    def _train_one(self, id, t, ac, im, rw, ter, tru):
        self.dqn_lst[id].train_one_step(t, ac, im, None, rw, ter, tru)

        
    def train(self):
        # start_time = time.time()
        state, info = self.env.reset()
        self.multi_agent_set_state(state)
        start_time = time.time()

        for i_episode in range(self.num_episodes):
            for t in range(self.episode_len):
                action_dict = self.multi_agent_select_action()
                observation, reward, terminated, truncated, _ = self.env.step(action_dict)
                img,rew,term,trunc,act = self.process_multi_agent_obs(observation,reward,terminated,truncated,action_dict)
                for i, id in enumerate(sorted(self.agent_ids)):
                    im = img[i:i+1]
                    # re = rep[i:i+1]
                    rw = rew[i:i+1]
                    ter = term[i:i+1]
                    tru = trunc[i:i+1]
                    ac = act[i:i+1]
                    self.dqn_lst[id].train_one_step(t,ac, im,rw,ter,tru)
            state, info = self.env.reset()
            self.log_harvest(i_episode, info)
            self.multi_agent_set_state(state)
        self.log_harvest(i_episode, info, force_log=True)

        print("complete")
    
    # TODO make this customizable
    def log_harvest(self,i_episode, info,force_log = False):
        if not force_log:
            line = ""
            for id in sorted(self.agent_ids):
                reward, time_out, beam_attempt,beam_success,rep,total_good_hit,collection_time=info[id]
                # action = info[str(id) + "a"]
                line += f"{reward},{time_out},{beam_attempt},{beam_success},{rep},{total_good_hit},{collection_time},"
            self.log.append(line[:-1]+"\n")
            apple_hist = info["map"]
            apple_hist_str = ', '.join(map(str, apple_hist))
            self.log.append(apple_hist_str + "\n")
        if i_episode % 100 == 0 or force_log:
            with open(self.log_file, 'a', newline='') as f:
                f.writelines(self.log)
            self.log = []
