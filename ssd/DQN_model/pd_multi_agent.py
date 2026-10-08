"""Independent DQN learners for the n-agent Prisoner's Dilemma.

Mirrors DQN_model/multi_agent.py: one `dqn` per agent, all stepping a shared
environment, no parameter sharing, so heterogeneous strategies (and
reputations) can emerge.

Differences from the harvest version:

  * the observation is a flat (num_agents - 1,) reputation vector instead of an
    RGB view, so the networks are plain MLPs (use_cnn=False),
  * PDEnv's action space is already DiscreteWithDType(2), so there is NO action
    wrapper anywhere in this file -- the Q head has one output per action and
    the index the policy picks IS the action (0 = cooperate, 1 = defect),
  * action tensors are built explicitly as (1, 1) long tensors so that
    gather(1, action_batch) in optimize_model always lines up,
  * logging follows the PD info schema, which uses the same six slots as the
    public goods logs so the existing parsers work unchanged.

A note on reward scale
----------------------
This depends entirely on the env's payoff_mode, and the advice has changed.

  * payoff_mode="threshold" with the n-dependent defaults: payoffs grow with
    the agent count -- a defector in a mixed round earns n + 1 -- so with
    n = 10 and a 200-step episode the return dwarfs anything the harvest or
    public goods agents see, and Q targets scale with it. If training diverges,
    set reward_scale to about 1 / n before touching the learning rate.
  * payoff_mode="fraction", and play_mode="pairwise" in particular: the matrix
    is fixed at t/r/p/s regardless of n, so one round pays at most t (3.0 by
    default) plus rep_weight * rep in [-rep_weight, rep_weight]. Nothing scales
    with the agent count and reward_scale=1.0 is correct. Scaling it down here
    only shrinks the reputation signal against the exploration noise.

A note on the observation
-------------------------
The env emits "own_rep" -- the agent's own standing, the figure its reputation
bonus is actually computed from. This file discovers observation keys
dynamically, so it is picked up automatically, and __init__ now refuses to
start if any key the env emits fails to reach the network. That check exists
because silently dropping "own_rep" would leave the agent paid for a quantity
it cannot see, which looks like a training failure rather than a plumbing bug.
"""

import time

import numpy as np
import torch

from DQN_model.single_agent import dqn
from social_dilemmas.envs.pgg_utils import flatten_obs, obs_keys_present, obs_dim


class pd_mult_agent_dqn:
    def __init__(self, device, env, epsilon_decay, conv_filters,
                 fc_filters, lr, replay_capacity, batch_size, gamma,
                 grad_clip, num_episode, tau, episode_len, log_file,
                 update_freq=1, use_cnn=False, debug=False,
                 epsilon_decay_style="LINEAR", reward_scale=1.0,
                 target_update_freq=1, mask_sitouts=True,
                 refund_epsilon=True, allow_dropped_obs_keys=()):
        self.device = device
        obs, info = env.reset()
        self.agent_ids = list(env.agents.keys())
        self.sorted_agent_ids = sorted(self.agent_ids)
        self.env = env
        self.dqn_lst = {}
        self.update_freq = update_freq
        self.reward_scale = reward_scale
        self.debug = debug
        # keep rounds the agent was not drawn to play out of the replay buffer
        self.mask_sitouts = mask_sitouts
        # give back the epsilon tick an agent spent on a round it turned out
        # not to play -- see skip_one_step
        self.refund_epsilon = refund_epsilon
        self.playing_now = {id: True for id in sorted(self.agent_ids)}

        print("obs is", obs[self.agent_ids[0]].keys())
        # every key the env emits reaches the network, not just curr_obs
        self.obs_keys = obs_keys_present(obs[self.agent_ids[0]])
        obs_shape = (obs_dim(obs[self.agent_ids[0]], self.obs_keys),)
        print("obs keys used:", self.obs_keys, "-> input width", obs_shape[0])
        # A key the env emits but obs_keys_present does not list never reaches
        # the network, and nothing downstream complains -- the run just quietly
        # trains on a smaller observation. That is how "own_rep" would go
        # missing, and an agent paid for a standing it cannot observe has no
        # way to attribute the bonus to its own behaviour. Fail loudly instead.
        dropped = (set(obs[self.agent_ids[0]].keys())
                   - set(self.obs_keys) - set(allow_dropped_obs_keys))
        if dropped:
            raise ValueError(
                "the env emits {} but obs_keys_present() does not list "
                "{}, so {} would never reach the network. Add {} to "
                "pgg_utils.obs_keys_present (and obs_dim/flatten_obs), or pass "
                "allow_dropped_obs_keys={} if the omission is deliberate."
                .format(sorted(obs[self.agent_ids[0]].keys()),
                        sorted(dropped), sorted(dropped), sorted(dropped),
                        tuple(sorted(dropped)))
            )
        # Discrete(2): cooperate / defect. No ladder, no wrapper.
        action_shape = env.action_space.n
        assert action_shape == 2, \
            "PDEnv should expose two actions, got {}".format(action_shape)
        self.num_episodes = num_episode

        self.episode_len = episode_len
        self.log_file = log_file
        self.log = []
        if self.log_file is not None:
            with open(self.log_file, 'w', newline='') as f:
                f.write("")

        for id in self.agent_ids:
            mod = dqn(self.env, device=device, epsilon_decay=epsilon_decay,
                      update_freq=self.update_freq, img_shape=obs_shape,
                      action_shape=action_shape, conv_filters=conv_filters,
                      fc_filters=fc_filters, lr=lr, replay_capacity=replay_capacity,
                      batch_size=batch_size, gamma=gamma, grad_clip=grad_clip,
                      num_episode=num_episode, tau=tau, episode_len=episode_len,
                      log_file=None, use_cnn=use_cnn, debug=debug,
                      epsilon_decay_style=epsilon_decay_style,
                      target_update_freq=target_update_freq,
                      use_amp=True)
            self.dqn_lst[id] = mod

    # ------------------------------------------------------------------ #
    # action selection
    # ------------------------------------------------------------------ #
    def multi_agent_select_action(self):
        """epsilon-greedy cooperate/defect, per agent.

        An agent that was not drawn this round does not call select_action at
        all. That matters for more than compute: `dqn.select_action` is what
        advances `steps_done`, and `steps_done` is what decays epsilon. Letting
        it tick on rounds the agent cannot act would anneal exploration at the
        full rate while the agent collects only k/n of the experience. The
        dummy action is ignored by the env.
        """
        actions = {}
        for id in self.sorted_agent_ids:
            if self.playing_now.get(id, True):
                actions[id] = int(self.dqn_lst[id].select_action().item())
            else:
                actions[id] = 0   # dummy; the env ignores it
        return actions

    def multi_agent_greedy_action(self):
        """argmax action, used for evaluation rollouts"""
        actions = {}
        with torch.no_grad():
            for id in self.sorted_agent_ids:
                q = self.dqn_lst[id].policy_net(self.dqn_lst[id].state)
                actions[id] = int(q.max(1).indices.item())
        return actions

    def q_values(self):
        """Q(cooperate), Q(defect) per agent -- the most direct read on what the
        agents have actually learned about the dilemma."""
        out = {}
        with torch.no_grad():
            for id in self.sorted_agent_ids:
                q = self.dqn_lst[id].policy_net(self.dqn_lst[id].state)
                out[id] = q.squeeze(0).cpu().numpy().tolist()
        return out

    # ------------------------------------------------------------------ #
    # state setup
    # ------------------------------------------------------------------ #
    def multi_agent_set_state(self, state):
        # who is drawn for the round the agents are about to play; the env puts
        # this in the observation precisely so the policy can condition on it
        self.playing_now = {
            id: bool(state[id]["is_playing"][0]) if "is_playing" in state[id] else True
            for id in self.sorted_agent_ids
        }
        obs_list = [flatten_obs(state[id], self.obs_keys)
                    for id in self.sorted_agent_ids]
        obs_batch = (
            torch.from_numpy(np.stack(obs_list))
            .to(self.device, non_blocking=True)
            .float()
        )
        for i, id in enumerate(self.sorted_agent_ids):
            self.dqn_lst[id].set_state(obs_batch[i:i + 1])

    def process_multi_agent_obs(self, observation, reward, terminated, truncated, action):
        obs_list, reward_list = [], []
        terminated_list, truncated_list, act_list = [], [], []
        for id in self.sorted_agent_ids:
            obs_list.append(flatten_obs(observation[id], self.obs_keys))
            reward_list.append(reward[id] * self.reward_scale)
            terminated_list.append(bool(terminated[id]))
            truncated_list.append(bool(truncated[id]))
            act_list.append(int(action[id]))
        obs_batch = (
            torch.from_numpy(np.stack(obs_list))
            .to(self.device, non_blocking=True)
            .float()
        )
        reward_batch = torch.as_tensor(
            reward_list, dtype=torch.float32, device=self.device
        )
        truncated_batch = torch.as_tensor(
            truncated_list, dtype=torch.float32, device=self.device
        )
        terminated_batch = torch.as_tensor(
            terminated_list, dtype=torch.float32, device=self.device
        )
        # (N, 1) so a per-agent slice is the (1, 1) index gather() wants
        action_batch = torch.as_tensor(
            act_list, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        return obs_batch, reward_batch, truncated_batch, terminated_batch, action_batch

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def skip_one_step(self, agent_id, t, obs_slice, acted=False):
        """A round this agent sat out.

        The transition is inert -- the action changed nothing and the reward is
        0 -- so it never enters replay, where it would otherwise crowd out real
        experience. The state still has to advance, or the next transition would
        pair a stale observation with a fresh action, and the optimisation
        schedule is kept identical so an agent does not learn more slowly merely
        because it was drawn less often.

        `acted` covers a case partial participation does not. Under
        play_mode="pairwise" an agent can be left without an opponent -- when
        the number of players is not a multiple of pair_size and exactly one is
        left over -- and that is decided inside step(), when the tables are
        drawn. It cannot be known when the action was selected, so unlike a
        num_players sit-out it is not announced in "is_playing": the agent
        already called select_action and already spent an epsilon tick on a
        round it could not play. Giving the tick back keeps exploration
        annealing at the rate the agent actually gathers experience.

        Choosing num_agents as a multiple of pair_size avoids the situation
        entirely -- no agent is ever stranded -- and is the cleaner fix.
        """
        mod = self.dqn_lst[agent_id]
        if acted and self.refund_epsilon and getattr(mod, "steps_done", 0) > 0:
            mod.steps_done -= 1
        mod.set_state(obs_slice)
        if t % mod.update_freq == 0:
            mod.optimize_model()
        if t % mod.target_update_freq == 0:
            mod._soft_update_target()

    def train(self):
        state, info = self.env.reset()
        self.multi_agent_set_state(state)
        start_time = time.time()

        for i_episode in range(self.num_episodes):
            for t in range(self.episode_len):
                action_dict = self.multi_agent_select_action()
                observation, reward, terminated, truncated, infos = self.env.step(
                    action_dict
                )
                obs, rew, trunc, term, act = self.process_multi_agent_obs(
                    observation, reward, terminated, truncated, action_dict
                )
                for i, id in enumerate(self.sorted_agent_ids):
                    played = infos[id].get("played", 1) if self.mask_sitouts else 1
                    if played:
                        self.dqn_lst[id].train_one_step(
                            t, act[i:i + 1], obs[i:i + 1], rew[i:i + 1],
                            term[i:i + 1], trunc[i:i + 1]
                        )
                    else:
                        # acted=True means this agent did select an action --
                        # it was stranded by the table draw rather than left
                        # out by num_players, so its epsilon tick is refunded
                        self.skip_one_step(
                            id, t, obs[i:i + 1],
                            acted=self.playing_now.get(id, True),
                        )
            state, info = self.env.reset()
            self.log_pd(i_episode, info)
            self.multi_agent_set_state(state)
            if self.debug and i_episode % 100 == 0:
                coop = np.mean([info[id][2] for id in self.sorted_agent_ids])
                print("episode {} | coop rate {:.3f} | epsilon {:.3f} | {:.1f}s".format(
                    i_episode, coop,
                    self.dqn_lst[self.sorted_agent_ids[0]].epsilon,
                    time.time() - start_time,
                ))
        self.log_pd(i_episode, info, force_log=True)

        print("complete")

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, num_episodes=1, episode_len=None):
        """Greedy rollouts; returns the cooperation rate and return per agent.

        Cooperation is counted over the rounds an agent actually PLAYED, not
        over every round of the rollout. A greedy action is selected for every
        agent each round, including ones the env then ignores -- a num_players
        sit-out, or an agent stranded without an opponent in pairwise play --
        and counting those dummy actions would report the env's masking as
        behaviour. With 0 = COOPERATE a dummy action reads as a cooperation,
        so the bias is upward, which is the direction that flatters the result.
        """
        episode_len = episode_len or self.episode_len
        coop = {id: 0 for id in self.sorted_agent_ids}
        played = {id: 0 for id in self.sorted_agent_ids}
        returns = {id: 0.0 for id in self.sorted_agent_ids}
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            self.multi_agent_set_state(state)
            for _t in range(episode_len):
                action_dict = self.multi_agent_greedy_action()
                state, reward, _term, _trunc, info = self.env.step(action_dict)
                for id in self.sorted_agent_ids:
                    if not info[id].get("played", 1):
                        continue
                    played[id] += 1
                    coop[id] += int(info[id]["cooperated"])
                    returns[id] += reward[id]
                self.multi_agent_set_state(state)
        coop_rate = {id: coop[id] / max(1, played[id]) for id in coop}
        mean_return = {id: returns[id] / num_episodes for id in returns}
        return coop_rate, mean_return

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #
    def log_pd(self, i_episode, info, force_log=False):
        """One CSV line of per-agent episode stats, one line of the cooperation
        history. Same six slots as the public goods logs:
        return, total cooperations, coop rate, defect steps, mutual-coop steps,
        reputation."""
        if self.log_file is None:
            return
        if not force_log:
            line = ""
            for id in self.sorted_agent_ids:
                (reward, total_coop, coop_rate, defect_steps,
                 mutual_coop, rep) = info[id]
                line += (f"{reward},{total_coop},{coop_rate},"
                         f"{defect_steps},{mutual_coop},{rep},")
            self.log.append(line[:-1] + "\n")
            self.log.append(', '.join(map(str, info["coop_hist"])) + "\n")
        if i_episode % 100 == 0 or force_log:
            with open(self.log_file, 'a', newline='') as f:
                f.writelines(self.log)
            self.log = []

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def save(self, path):
        torch.save(
            {id: self.dqn_lst[id].policy_net.state_dict() for id in self.agent_ids},
            path,
        )

    def load(self, path):
        state_dicts = torch.load(path, map_location=self.device)
        for id, sd in state_dicts.items():
            self.dqn_lst[id].policy_net.load_state_dict(sd)
            self.dqn_lst[id].target_net.load_state_dict(sd)