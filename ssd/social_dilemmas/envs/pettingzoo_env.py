from functools import lru_cache

from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# from pettingzoo.utils import wrappers
# from pettingzoo.utils.conversions import from_parallel_wrapper
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper, OrderEnforcingWrapper
from pettingzoo import ParallelEnv

from social_dilemmas.envs.env_creator import get_env_creator

MAX_CYCLES = 10


def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
    return _parallel_env(max_cycles, **ssd_args)

def multi_agent_env(max_cycles=MAX_CYCLES, **ssd_args):
    return _parallel_env_multi_agent(max_cycles, **ssd_args)

def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
    # Get the ParallelEnv
    parallel = parallel_env(max_cycles, **ssd_args)

    # Convert to AECEnv using built-in method
    aec = parallel.parallel_to_aec()

    return aec


def env(max_cycles=MAX_CYCLES, **ssd_args):
    aec_env = raw_env(max_cycles, **ssd_args)

    aec_env = AssertOutOfBoundsWrapper(aec_env)
    aec_env = OrderEnforcingWrapper(aec_env)

    return aec_env

class ssd_parallel_env(ParallelEnv):
    def __init__(self, env, max_cycles):
        self.ssd_env = env
        self.max_cycles = max_cycles
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
        self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
        self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
        self.action_spaces = {agent: env.action_space for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.agents}
        result = self.ssd_env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result

        else:
            obs, info = result, {}
        
        return obs, info

    def seed(self, seed=None):
        return self.ssd_env.seed(seed)

    def render(self, mode="human"):
        return self.ssd_env.render(mode=mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions):  

        obss, rews, self.dones, trunc, infos = self.ssd_env.step(actions)
        del self.dones["__all__"]
        self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            self.dones = {agent: True for agent in self.agents}
        self.agents = [agent for agent in self.agents if not self.dones[agent]]
        return obss, rews, self.dones, trunc,infos

class ssd_multi_agent_env(MultiAgentEnv):
    def __init__(self, env, max_cycles, config=None):
        super().__init__()
        self.ssd_env = env
        self.max_cycles = max_cycles
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_spaces = {
            agent: env.observation_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: env.action_space for agent in self.possible_agents
        }
        
        self._agent_ids = self.possible_agents

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.agents}
        obs,info = self.ssd_env.reset()

        return obs, info

    def seed(self, seed=None):
        return self.ssd_env.seed(seed)

    def render(self, mode="human"):
        return self.ssd_env.render(mode=mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions): 
        obs, rews, dones, trunc, infos = self.ssd_env.step(actions)
        # del self.dones["__all__"]
        self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            dones = {agent: True for agent in self._agent_ids}

        # RLlib expects all agents plus "__all__" in dones
        dones_with_all = {agent: dones.get(agent, True) for agent in self._agent_ids}
        dones_with_all["__all__"] = all(dones_with_all.values())

        return obs, rews, dones_with_all, trunc, infos  
    

class _parallel_env(ssd_parallel_env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, max_cycles, **ssd_args):
        EzPickle.__init__(self, max_cycles, **ssd_args)
        env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
        super().__init__(env, max_cycles)

class _parallel_env_multi_agent(ssd_multi_agent_env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, max_cycles, **ssd_args):
        EzPickle.__init__(self, max_cycles, **ssd_args)
        env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
        super().__init__(env, max_cycles)

class CustomStepWrapper(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "CustomStepWrapper"}

    def __init__(self, env, max_cycles):
        self.env = env
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.max_cycles = max_cycles
        self.num_cycles = 0

    def reset(self, seed=None, options=None):
        self.num_cycles = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self.env.step(action)
        self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            for agent in self.env.agents:
                self.env._cumulative_rewards[agent] = 0  # optional: cleanup
                self.env.dones[agent] = True

    def observe(self, agent):
        return self.env.observe(agent)

    def render(self, *args, **kwargs): return self.env.render(*args, **kwargs)
    def close(self): return self.env.close()
    def state(self): return self.env.state()
    def __getattr__(self, name): return getattr(self.env, name)