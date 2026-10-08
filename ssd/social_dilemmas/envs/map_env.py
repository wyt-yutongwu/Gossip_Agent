"""Base map class that defines the rendering process
"""
 
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box, Dict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import MultiAgentEnv
from gymnasium.spaces.utils import flatdim, flatten
from math import tanh
import copy
from collections import deque
 
_MAP_ENV_ACTIONS = {
    "MOVE_LEFT": [0, -1],  # Move left
    "MOVE_RIGHT": [0, 1],  # Move right
    "MOVE_UP": [-1, 0],  # Move up
    "MOVE_DOWN": [1, 0],  # Move down
    "STAY": [0, 0],  # don't move
    "TURN_CLOCKWISE": [[0, 1], [-1, 0]],  # Clockwise rotation matrix
    "TURN_COUNTERCLOCKWISE": [[0, -1], [1, 0]],
}  # Counter clockwise rotation matrix
# Positive Theta is in the counterclockwise direction
 
ORIENTATIONS = {"LEFT": [0, -1], "RIGHT": [0, 1], "UP": [-1, 0], "DOWN": [1, 0]}
BASE_ACTIONS = {
    0: "MOVE_LEFT",  # Move left
    1: "MOVE_RIGHT",  # Move right
    2: "MOVE_UP",  # Move up
    3: "MOVE_DOWN",  # Move down
    4: "STAY",  # don't move
    5: "TURN_CLOCKWISE",  # Rotate counter clockwise
    6: "TURN_COUNTERCLOCKWISE",
}  # Rotate clockwise
# DEFAULT_COLOURS = {
#     b" ": np.array([0, 0, 0], dtype=np.uint8),  # Black background
#     b"0": np.array([0, 0, 0], dtype=np.uint8),  # Black background beyond map walls
#     b"": np.array([180, 180, 180], dtype=np.uint8),  # Grey board walls
#     b"@": np.array([180, 180, 180], dtype=np.uint8),  # Grey board walls
#     b"A": np.array([0, 255, 0], dtype=np.uint8),  # Green apples
#     b"F": np.array([255, 255, 0], dtype=np.uint8),  # Yellow firing beam
#     b"P": np.array([159, 67, 255], dtype=np.uint8),  # Generic agent (any player)
#     # Colours for agents. R value is a unique identifier
#     b"1": np.array([0, 0, 255], dtype=np.uint8),  # Pure blue
#     b"2": np.array([2, 81, 154], dtype=np.uint8),  # Sky blue
#     b"3": np.array([204, 0, 204], dtype=np.uint8),  # Magenta
#     b"4": np.array([216, 30, 54], dtype=np.uint8),  # Red
#     b"5": np.array([254, 151, 0], dtype=np.uint8),  # Orange
#     b"6": np.array([100, 255, 255], dtype=np.uint8),  # Cyan
#     b"7": np.array([99, 99, 255], dtype=np.uint8),  # Lavender
#     b"8": np.array([250, 204, 255], dtype=np.uint8),  # Pink
#     b"9": np.array([238, 223, 16], dtype=np.uint8),  # Yellow
# }
 
DEFAULT_COLOURS = {
    b" ": np.array([0, 0, 0], dtype=np.uint8),  # Black background
    b"0": np.array([0, 0, 0], dtype=np.uint8),  # Black background beyond map walls
    b"": np.array([180, 180, 180], dtype=np.uint8),  # Grey board walls
    b"@": np.array([180, 180, 180], dtype=np.uint8),  # Grey board walls
    b"A": np.array([0, 255, 0], dtype=np.uint8),  # Green apples
    b"F": np.array([255, 255, 0], dtype=np.uint8),  # Yellow firing beam
    b"P": np.array([159, 67, 255], dtype=np.uint8),  # Generic agent (any player)
    # Colours for agents. R value is a unique identifier
    b"+": np.array([238, 223, 16], dtype=np.uint8),  # Yellow, positive or zero reputation
    b"-": np.array([216, 30, 54], dtype=np.uint8),  # Red, negative reputation
    b"1": np.array([0, 0, 255], dtype=np.uint8),  # Pure blue, agent itself
    # b"1": np.array([0, 0, 255], dtype=np.uint8),  # Pure blue
    # b"2": np.array([2, 81, 154], dtype=np.uint8),  # Sky blue
    # b"3": np.array([204, 0, 204], dtype=np.uint8),  # Magenta
    # b"4": np.array([216, 30, 54], dtype=np.uint8),  # Red
    # b"5": np.array([254, 151, 0], dtype=np.uint8),  # Orange
    # b"6": np.array([100, 255, 255], dtype=np.uint8),  # Cyan
    # b"7": np.array([99, 99, 255], dtype=np.uint8),  # Lavender
    # b"8": np.array([250, 204, 255], dtype=np.uint8),  # Pink
    # b"9": np.array([238, 223, 16], dtype=np.uint8),  # Yellow
}
 
# the axes look like this when printed out
# WARNING: increasing array position in the direction of down
# so for example if you move_left when facing left
# your y position decreases.
#         ^
#         |
#         U
#         P
# <--LEFT * RIGHT---->
#         D
#         O
#         W
#         N
#         |
 
 
class MapEnv(MultiAgentEnv):
    def __init__(
        self,
        ascii_map,
        extra_actions,
        view_len,
        num_agents=1,
        color_map=None,
        use_collective_reward=False,
        inequity_averse_reward=False,
        use_reputation = False,
        alpha=0.0,
        beta=0.0,
        # how to broadcast reputation: 0 no broadcast, 1 broadcast to all, 2 word-of-mouth
        broadcast_mode = -1,
        broadcast_radius = 0,
        network_mode = False,
        adjacency_matrix = None,

    ):
        """
 
        Parameters
        ----------
        ascii_map: list of strings
            Specify what the map should look like. Look at constant.py for
            further explanation
        extra_actions: dict with action name-value pair
            Environment-specific actions that are not present in _MAP_ENV_ACTIONS
        num_agents: int
            Number of agents to have in the system.
        color_map: dict
            Specifies how to convert between ascii chars and colors
        """
        self.debug = False
        self._num_agents = num_agents 
        self.base_map = self.ascii_to_numpy(ascii_map)
        self.view_len = view_len
        self.map_padding = view_len
        self.use_collective_reward = use_collective_reward
        self.inequity_averse_reward = inequity_averse_reward
        self.use_reputation = use_reputation
        self.alpha = alpha
        self.beta = beta
        self.all_actions = _MAP_ENV_ACTIONS.copy()
        self.all_actions.update(extra_actions)
        self.decay = 0.99
        self.num_spawn_apples = 0
        self.dist_agent_to_apple = 1
        self.apple_hist = []

        # keep track of episode and step num
        self.ep_count = 0
        self.count = 0

        # broadcast param
        self.broadcast_mode = broadcast_mode
        self.broadcast_radius = broadcast_radius

        # network param: adjacency_matrix[i][j] nonzero means agent-i and agent-j are neighbors
        self.network_mode = network_mode
        print("in init, network mode is: ", self.network_mode)
        if self.network_mode:
            assert adjacency_matrix is not None, "network_mode requires an adjacency_matrix"
            self.adjacency_matrix = np.asarray(adjacency_matrix)
            assert self.adjacency_matrix.shape == (num_agents, num_agents), \
                "adjacency_matrix must have shape (num_agents, num_agents)"
            # pre-compute the neighbor sets once, since the network is static
            self.network_neighbor_map = {
                "agent-" + str(i): {
                    "agent-" + str(j)
                    for j in range(num_agents)
                    if i != j and self.adjacency_matrix[i][j]
                }
                for i in range(num_agents)
            }
            print("agent_neighbor map: ", self.network_neighbor_map)
        else:
            self.adjacency_matrix = None
            
        # Map without agents or beams
        self.world_map = np.full(
            (len(self.base_map), len(self.base_map[0])), fill_value=b" ", dtype="c"
        )
        # Color mapping
        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS.copy()
        # World map image
        self.world_map_color = np.full(
            (len(self.base_map) + view_len * 2, len(self.base_map[0]) + view_len * 2, 3),
            fill_value=0,
            dtype=np.uint8,
        )
        self.beam_pos = []
 
        self.agents = {}
 
        # returns the agent at a desired position if there is one
        self.pos_dict = {}
        self.spawn_points = []  # where agents can appear
 
        self.wall_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"@":
                    self.wall_points.append([row, col])
        self.setup_agents()
        print(self.agents.keys())
        self.succ_hits = {}
        for agent in self.agents.values():
            self.succ_hits[agent.agent_id] = 0


    @property
    def num_agents(self):
        return self._num_agents
            
    @property
    def observation_space(self):
        obs_space = {
            "curr_obs": Box(
                low=0,
                high=255,
                shape=(2 * self.view_len + 1, 2 * self.view_len + 1, 3),
                dtype=np.float32,
            )
        }
        print("obs_space", obs_space)
        return obs_space
    
    # def agent_id_to_index(self, agent_id):
    #     """Convert an agent id like 'agent-3' to its integer index 3"""
    #     return int(agent_id.split("-")[-1])
    
    # def is_network_neighbor(self, agent_id, other_agent_id):
    #     """Check the adjacency matrix to see if two agents are neighbors"""
    #     i = self.agent_id_to_index(agent_id)
    #     j = self.agent_id_to_index(other_agent_id)
    #     return bool(self.adjacency_matrix[i][j])
    
    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn apples and build walls"""
        pass
 
    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like fire or clean
 
        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken
 
        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        pass
 
    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass
 
    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError
 
    # FIXME(ev) move this to a utils eventually
    def ascii_to_numpy(self, ascii_list):
        """converts a list of strings into a numpy array
 
 
        Parameters
        ----------
        ascii_list: list of strings
            List describing what the map should look like
        Returns
        -------
        arr: np.ndarray
            numpy array describing the map with ' ' indicating an empty space
        """
 
        arr = np.full((len(ascii_list), len(ascii_list[0])), b" ", dtype="c")
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    def run_rep(self):
        # agent change rep -> calculate true rep -> broadcast and update rep again based on some weight
        apple_in_view = self.num_apple_in_view()
        reps = {}
        rep_change_count = {}
        rep_change = set()
        good_hit = dict()
        if self.network_mode:
            # neighbors are static and pre-computed, stored on each agent
            agent_neighbor = self.network_neighbor_map
        else:
            agent_neighbor = {aid: set() for aid in self.agents}
        for agent_id in self.agents:
            if agent_id not in good_hit:
                good_hit[agent_id] = self.agents[agent_id].good_hit_this_turn()
            curr_num_apple = apple_in_view[agent_id]
            agent_pos = self.agents[agent_id].pos
            for other_agent_id in self.agents:
                if other_agent_id not in good_hit:
                    good_hit[other_agent_id] = self.agents[other_agent_id].good_hit_this_turn()
                if other_agent_id not in reps:
                    reps[other_agent_id] = 0
                if agent_id != other_agent_id:
                    total_rep = 0
                    other_agnet_pos = self.agents[other_agent_id].pos
                    if not self.network_mode and abs(agent_pos[0] - other_agnet_pos[0]) < self.broadcast_radius and abs(agent_pos[1] - other_agnet_pos[1]) < self.broadcast_radius:
                        agent_neighbor[agent_id].add(other_agent_id)
                    if abs(agent_pos[0] - other_agnet_pos[0]) < self.view_len and abs(agent_pos[1] - other_agnet_pos[1]) < self.view_len:
                        # apple related reputation
                        if self.agents[other_agent_id].reward_this_turn == 1 and curr_num_apple <= 1:
                            total_rep += -1
                        elif curr_num_apple == 0:
                            total_rep += -1
                        elif self.agents[other_agent_id].reward_this_turn == 0 and curr_num_apple <= 1:
                            total_rep += 1

                        if good_hit[other_agent_id] == -1:
                            total_rep += -1
                        if total_rep != 0:
                            rep_change.add(other_agent_id)
                            if other_agent_id not in rep_change_count:
                                rep_change_count[other_agent_id] = 1
                            else:
                                rep_change_count[other_agent_id] += 1
                            self.agents[agent_id].adjust_other_agent_rep(other_agent_id, total_rep)
                    reps[other_agent_id] += self.agents[agent_id].other_agent_rep[other_agent_id]
        
        # broadcast average rep for each agent
        for id in rep_change:
            avg = reps[id] / (self.num_agents - 1)
            self.agents[id].add_rep(avg)
            curr_rep = self.agents[id].calculate_rep()
            self.agents[id].reputation = curr_rep

        for id in self.agents:
            # broadcast rep
            neighbor_avg = dict(self.agents[id].other_agent_rep)
            neighbor_count = 1
            self.agents[id].tmp_other_agent_rep = neighbor_avg
            for agent_id in self.agents:
                if agent_id in agent_neighbor[id]:
                    neighbor_count += 1
                    new_neighbor_avg = {i: neighbor_avg[i] + self.agents[agent_id].other_agent_rep[i] for i in neighbor_avg.keys()}
                    neighbor_avg = new_neighbor_avg
            if neighbor_count > 1:
                for i in neighbor_avg.keys():
                    if i == id:
                        neighbor_avg[i] = 0
                    elif i in agent_neighbor[id]:
                        neighbor_avg[i] = neighbor_avg[i] / (neighbor_count - 1)
                    else:
                        neighbor_avg[i] = neighbor_avg[i] / neighbor_count
                self.agents[id].tmp_other_agent_rep = neighbor_avg
        for id in self.agents:
            self.agents[id].other_agent_rep = self.agents[id].tmp_other_agent_rep


    def step(self, actions):
        """Takes in a dict of actions and converts them to a map update
 
        Parameters
        ----------
        actions: dict {agent-id: int}
            dict of actions, keyed by agent-id that are passed to the agent. The agent
            interprets the int and converts it to a command
 
        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym
        """
        self.count += 1
        if self.ep_count >= 500 and self.debug:
            print("===== step =====")
            print("map is")
            print(self.get_map_with_agents())
        # Update time_out time for the agent
        for agent in self.agents.values():
            agent.update_time_out()
        self.beam_pos = []
        agent_actions = {}
        # Only append action if the agent is active
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            if agent.active:
                # agent.add_to_action_hist(action)    
                agent_action = agent.action_map(action)
                agent_actions[agent_id] = agent_action
            else:
                agent_actions[agent_id] = None
 
        # Remove agents from color map
        for agent in self.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            self.single_update_world_color_map(row, col, self.world_map[row, col])
 
        self.update_moves(agent_actions)
 
        for agent in self.agents.values():
            if not agent.active:
                continue
            pos = agent.pos
            new_char = agent.consume(self.world_map[pos[0], pos[1]],self.count)
            self.single_update_map(pos[0], pos[1], new_char)
 
 
        # execute custom moves like firing
        self.update_custom_moves(agent_actions)
 
        # execute spawning events
        self.custom_map_update()
        # update reputation
        # also runs reputation in baseline, but will not affect the reward function
        self.run_rep()
                
        map_with_agents = self.get_map_with_agents()
        # Add agents to color map
        for agent in self.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            # Firing beams have priority over agents and should cover them
            if self.world_map[row, col] not in [b"F", b"C"]:
                self.single_update_world_color_map(row, col, agent.get_char_id(), agent.return_reputation())
 
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        num_apples = self.find_num_apples()
        self.update_apple_hist(num_apples)
 
        # self.min_num_apples = min(num_apples, self.min_num_apples)
        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = self.color_view(agent) / 255.0
 
            # concatenate on the prev_actions to the observations
            if agent.active:
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                }
            else:
                observations[agent.agent_id] = self.timed_out_obs()
            # Return rewards so far, and cumulative timed out time
            reward, reward_so_far = agent.compute_reward()
            if agent.active and self.use_reputation:
                reward = reward +  self.alpha * agent.return_reputation()
            rewards[agent.agent_id] = reward
                
 
            dones[agent.agent_id] = False
            time_out = agent.compute_time_out()
            beam_attempt = agent.get_beam_attemp()
            successful_hit = agent.get_successful_hit()
            self.succ_hits[agent.agent_id] = successful_hit
 
            if self.use_reputation:
                reputation = agent.return_reputation()
                infos[agent.agent_id] = {
                    "total_time_out_steps": time_out,
                    "reward_this_episode": reward_so_far,
                    "reputation": reputation,
                }
            else:
                infos[agent.agent_id] = {
                    "total_time_out_steps": time_out,
                    "reward_this_episode": reward_so_far,
                    "beam_attempt": beam_attempt,
                    "successful_hit": successful_hit,
                    "spawned_apples": self.num_spawn_apples
                }
    
        dones["__all__"] = np.any(list(dones.values()))
        episode_done = dones["__all__"]
        # Set truncated to match episode end, explicitly specifying truncation
        trunc = {agent_id: episode_done for agent_id in dones}
        trunc["__all__"] = episode_done

 
        return observations, rewards, dones, trunc,infos
    
    def simple_obs(self):
        arr = np.zeros(4, dtype = np.float32)
        count= 0
        map = self.get_map_with_agents()
        for i in range(1,3):
            for j in range(1,3):
                if map[i,j] == b"A":
                    arr[count] = 1.0
                elif map[i,j] == b"1":
                    arr[count] = 2
                count += 1
        return arr
 
 
 
    def reset(self):
        """Reset the environment.
 
        This method is performed in between rollouts. It resets the state of
        the environment.
 
        Returns
        -------
        observation: dict of numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        info = {}
        for agent in self.agents.values():
            reward, reward_so_far = agent.compute_reward()
            time_out = agent.compute_time_out()
            beam_attempt = agent.get_beam_attemp()
            beam_success = agent.get_successful_hit()
            rep = agent.return_reputation()
            collection_time = agent.return_collection_time()
            # action_hist = agent.return_action_hist()
            total_good_hit = agent.return_total_good_hit()
            info[agent.agent_id] = [reward_so_far, time_out,beam_attempt,beam_success,rep,total_good_hit,collection_time] 
            # info[str(agent.agent_id)+"a"] = action_hist
        info["map"] = self.return_apple_hist()
        self.apple_hist = []
        if self.ep_count >= 500 and self.debug:
            print("reset")
        self.ep_count += 1
        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.rep_hist = []
        self.min_num_apples = 10000
        self.custom_map_update()
        self.num_spawn_apples = 0
        self.count = 0
        map_with_agents = self.get_map_with_agents()
        for agent in self.agents.values():
            self.succ_hits[agent.agent_id] = 0
        self.dist_agent_to_apple = self.distance_to_apple()
        observations = {}
        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = self.color_view(agent) / 255.0
            observations[agent.agent_id] = {"curr_obs": rgb_arr}
        return observations, info
 
    def seed(self, seed=None):
        np.random.seed(seed)
 
    def close(self):
        plt.close()
 
    @property
    def agent_pos(self):
        return [agent.pos.tolist() for agent in self.agents.values() if agent.active]
    
    def timed_out_obs(self):
 
        output = {  "curr_obs": np.array(np.zeros((2 * self.view_len + 1, 2 * self.view_len + 1, 3)),dtype=np.float32),
                    }
        return output
    def get_map_with_agents(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.
 
        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)
 
        for agent in self.agents.values():
            char_id = agent.get_char_id()
 
            # If agent is not within map, skip.
            if not (0 <= agent.pos[0] < grid.shape[0] and 0 <= agent.pos[1] < grid.shape[1]):
                continue
 
            grid[agent.pos[0], agent.pos[1]] = char_id
 
        # beams should overlay agents
        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]
 
        return grid
 
    def check_agent_map(self, agent_map):
        """Checks the map to make sure agents aren't duplicated"""
        unique, counts = np.unique(agent_map, return_counts=True)
        count_dict = dict(zip(unique, counts))
 
        # check for multiple agents
        for i in range(self.num_agents):
            if count_dict[chr(i + 1)] != 1:
                print("Error! Wrong number of agent", i, "in map!")
                return False
        return True
 
    def full_map_to_colors(self):
        map_with_agents = self.get_map_with_agents()
        rgb_arr = np.zeros((map_with_agents.shape[0], map_with_agents.shape[1], 3), dtype=int)
        return self.map_to_colors(map_with_agents, self.color_map, rgb_arr)
 
    def color_view(self, agent):
        other_agent_rep = agent.other_agent_rep
        row, col = agent.pos[0], agent.pos[1]
        world_map_copy = copy.deepcopy(self.world_map_color)
        world_map_copy[row + self.map_padding,col + self.map_padding] = DEFAULT_COLOURS[b"1"]
        if self.broadcast_mode >= 0:
            agent_pos = {a_id: a.pos for a_id, a in self.agents.items()}
            for a_id in agent_pos:
                x, y = agent_pos[a_id]
                if row - self.view_len <= x or row + self.view_len >= x:
                    if col - self.view_len <= y or col + self.view_len >= y:
                        if other_agent_rep[a_id] < 0:
                            world_map_copy[x +self.map_padding, y+self.map_padding] = self.color_map[b"-"]
                        else:
                            world_map_copy[x +self.map_padding, y+self.map_padding] = self.color_map[b"+"]
 
        view_slice = world_map_copy[
                row + self.map_padding - self.view_len : row + self.map_padding + self.view_len + 1,
                col + self.map_padding - self.view_len : col + self.map_padding + self.view_len + 1,
            ]
        if agent.orientation == "UP":
            rotated_view = view_slice
        elif agent.orientation == "LEFT":
            rotated_view = np.rot90(view_slice)
        elif agent.orientation == "DOWN":
            rotated_view = np.rot90(view_slice, k=2)
        elif agent.orientation == "RIGHT":
            rotated_view = np.rot90(view_slice, k=1, axes=(1, 0))
        
        return rotated_view
 
    def map_to_colors(self, mmap, color_map, rgb_arr, orientation="UP"):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        mmap: np.ndarray
            map to convert to colors
            Double m to avoid shadowing map.
        color_map: dict
            mapping between array elements and desired colors
        rgb_arr: np.array
            Variable to store the mapping in
        orientation:
            The way in which the output should be oriented.
             UP = no rotation.
             RIGHT = Clockwise 90 degree rotation.
             DOWN = Clockwise 180 degree rotation.
             LEFT = Clockwise 270 degree rotation.
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        x_len = mmap.shape[0]
        y_len = mmap.shape[1]
        if orientation == "UP":
            for row_elem in range(x_len):
                for col_elem in range(y_len):
                    rgb_arr[row_elem, col_elem, :] = color_map[mmap[row_elem, col_elem]]
        elif orientation == "LEFT":
            for row_elem in range(x_len):
                for col_elem in range(y_len):
                    rgb_arr[row_elem, col_elem, :] = color_map[mmap[col_elem, x_len - 1 - row_elem]]
        elif orientation == "DOWN":
            for row_elem in range(x_len):
                for col_elem in range(y_len):
                    rgb_arr[row_elem, col_elem, :] = color_map[
                        mmap[x_len - 1 - row_elem, y_len - 1 - col_elem]
                    ]
        elif orientation == "RIGHT":
            for row_elem in range(x_len):
                for col_elem in range(y_len):
                    rgb_arr[row_elem, col_elem, :] = color_map[mmap[y_len - 1 - col_elem, row_elem]]
        else:
            raise ValueError("Orientation {} is not valid".format(orientation))
 
        return rgb_arr
 
    def render(self, title, filename=None,mode="human"):
        """Creates an image of the map to plot or save.
 
        Args:
            filename: If a string is passed, will save the image
                      to disk at this location.
        """
        rgb_arr = self.world_map_color.astype(np.uint8)#self.full_map_to_colors()
        if mode == "human":
            plt.cla()
            # plt.title(title)
            YELLOW = DEFAULT_COLOURS[b"+"]
            BLUE = DEFAULT_COLOURS[b"1"]
            BEAM = DEFAULT_COLOURS[b"F"]
            yellow_coords = np.argwhere(np.all(rgb_arr == YELLOW, axis=-1))
 
            if len(yellow_coords) > 0:
                r, c = yellow_coords[np.random.randint(len(yellow_coords))]
                rgb_arr[r, c] = BLUE
                for i in range(-1,2):
                    for j in range(1,4):
                        if r + i > len(rgb_arr) or c + j > len(rgb_arr[0]) or r + i < 0:
                            continue
                        rgb_arr[r+i,c+j] = BEAM
            else:
                return rgb_arr
            plt.imshow(rgb_arr, interpolation="nearest")
            plt.axis('off')
            if filename is None:
                plt.show(block=False)
            else:
                plt.savefig(filename,transparent=True)
            return None
        return rgb_arr
 
    def update_moves(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions.
         Also resolves conflicts over multiple agents wanting a cell.
 
         This method works by finding all conflicts over a cell and randomly assigning them
        to one of the agents that desires the slot. It then sets all of the other agents
        that wanted the cell to have a move of staying. For moves that do not directly
        conflict with another agent for a cell, but may not be temporarily resolvable
        due to an agent currently being in the desired cell, we continually loop through
        the actions until all moves have been satisfied or deemed impossible.
        For example, agent 1 may want to move from [1,2] to [2,2] but agent 2 is in [2,2].
        Agent 2, however, is moving into [3,2]. Agent-1's action is first in the order so at the
        first pass it is skipped but agent-2 moves to [3,2]. In the second pass, agent-1 will
        then be able to move into [2,2].
 
         Parameters
         ----------
         agent_actions: dict
             dict with agent_id as key and action as value
        """
 
        reserved_slots = []
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            # if agent is timed out, no action is taken
            if not agent.active:
                continue
            selected_action = self.all_actions[action]
            # TODO(ev) these two parts of the actions
            if "MOVE" in action or "STAY" in action:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(selected_action, agent.get_orientation())
                new_pos = agent.pos + rot_action
                # allow the agents to confirm what position they can move to
                new_pos = agent.return_valid_pos(new_pos)
                reserved_slots.append((*new_pos, b"P", agent_id))
            elif "TURN" in action:
                new_rot = self.update_rotation(action, agent.get_orientation())
                agent.update_agent_rot(new_rot)
 
        # now do the conflict resolution part of the process
 
        # helpful for finding the agent in the conflicting slot, ignore timed-out agents
        agent_by_pos = {tuple(agent.pos): agent.agent_id for agent in self.agents.values() if agent.active}
 
        # agent moves keyed by ids
        agent_moves = {}
 
        # lists of moves and their corresponding agents
        move_slots = []
        agent_to_slot = []
 
        for slot in reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == b"P":
                agent_id = slot[3]
                agent_moves[agent_id] = [row, col]
                move_slots.append([row, col])
                agent_to_slot.append(agent_id)
 
        # cut short the computation if there are no moves
        if len(agent_to_slot) > 0:
 
            # first we will resolve all slots over which multiple agents
            # want the slot
 
            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(
                move_slots, return_index=True, return_counts=True, axis=0
            )
            search_list = np.array(move_slots)
 
            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another
 
            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        conflict_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent_id in all_agents_id:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            if move.tolist() in self.agent_pos:
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent_id = agent_by_pos[tuple(move)]
                                curr_pos = self.agents[agent_id].pos.tolist()
                                curr_conflict_pos = self.agents[conflicting_agent_id].pos.tolist()
                                conflict_move = agent_moves.get(
                                    conflicting_agent_id, curr_conflict_pos
                                )
                                # Condition (1):
                                # a STAY command has been issued
                                if agent_id == conflicting_agent_id:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command
                                elif (
                                    conflicting_agent_id not in moves_copy.keys()
                                    or curr_conflict_pos == conflict_move
                                ):
                                    conflict_cell_free = False
 
                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent_id in moves_copy.keys():
                                    if (
                                        agent_moves[conflicting_agent_id] == curr_pos
                                        and move.tolist()
                                        == self.agents[conflicting_agent_id].pos.tolist()
                                    ):
                                        conflict_cell_free = False
 
                        # if the conflict cell is open, let one of the conflicting agents
                        # move into it
                        if conflict_cell_free:
                            self.agents[agent_to_slot[index]].update_agent_pos(move)
                            agent_by_pos = {
                                tuple(agent.pos): agent.agent_id for agent in self.agents.values()
                            }
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id].pos.tolist()
 
            # make the remaining un-conflicted moves
            while len(agent_moves.items()) > 0:
                agent_by_pos = {tuple(agent.pos): agent.agent_id for agent in self.agents.values()}
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent_id, move in moves_copy.items():
                    # Check time out
                    if agent_id in del_keys or not self.agents[agent_id].active:
                        continue
                    if move in self.agent_pos:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        conflicting_agent_id = agent_by_pos[tuple(move)]
                        curr_pos = self.agents[agent_id].pos.tolist()
                        curr_conflict_pos = self.agents[conflicting_agent_id].pos.tolist()
                        conflict_move = agent_moves.get(conflicting_agent_id, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent_id == conflicting_agent_id:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif (
                            conflicting_agent_id not in moves_copy.keys()
                            or curr_conflict_pos == conflict_move
                        ):
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent_id in moves_copy.keys():
                            if (
                                agent_moves[conflicting_agent_id] == curr_pos
                                and move == self.agents[conflicting_agent_id].pos.tolist()
                            ):
                                del agent_moves[conflicting_agent_id]
                                del agent_moves[agent_id]
                                del_keys.append(agent_id)
                                del_keys.append(conflicting_agent_id)
                    # this move is unconflicted so go ahead and move
                    else:
                        self.agents[agent_id].update_agent_pos(move)
                        del agent_moves[agent_id]
                        del_keys.append(agent_id)
 
                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent_id, move in agent_moves.items():
                        self.agents[agent_id].update_agent_pos(move)
                    break
 
 
    def update_custom_moves(self, agent_actions):
        """
        This function executes non-movement actions like firing, cleaning etc.
        The order in which agent actions are resolved is random to ensure homogeneity, similar to
        update_moves, otherwise a race condition occurs which prioritizes lower-numbered agents
        """
        agent_ids = list(agent_actions.keys())
        np.random.shuffle(agent_ids)
        # Check if the agent is timed-out
        for agent_id in agent_ids:        
            agent = self.agents[agent_id]
            if not agent.active:
                continue
            action = agent_actions[agent_id]
            # check its not a move based action
            if "MOVE" not in action and "STAY" not in action and "TURN" not in action:
                agent = self.agents[agent_id]
                updates = self.custom_action(agent, action)
                # print("in custom", updates)
                if len(updates) > 0:
                    self.update_map(updates)
 
    def update_map(self, new_points):
        """For points in new_points, place desired char on the map
        Update the color map as well"""
        for point in new_points:
            self.single_update_map(*point)
 
    def single_update_map(self, row, col, char, rep = None):
        self.world_map[row, col] = char 
        if rep is not None and rep < 0: 
            self.world_map_color[row + self.map_padding, col + self.map_padding] = self.color_map[b"-"]
        elif rep is not None and rep >= 0:
            self.world_map_color[row + self.map_padding, col + self.map_padding] = self.color_map[b"+"]
        else:            
            self.world_map_color[row + self.map_padding, col + self.map_padding] = self.color_map[char]
 
 
    def single_update_world_color_map(self, row, col, char, rep = None):
        """Only update the color map. This is done separately when agents move, because their own
        position state is not contained in self.world_map, but in their own Agent objects"""
        if rep is not None and rep < 0: 
            self.world_map_color[row + self.map_padding, col + self.map_padding] = self.color_map[b"-"]
        elif rep is not None and rep >= 0:
            self.world_map_color[row + self.map_padding, col + self.map_padding] = self.color_map[b"+"]
        else:            
            self.world_map_color[row + self.map_padding, col + self.map_padding] = self.color_map[char]
 
    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), b" ", dtype="c")
        self.world_map_color = np.full(
            (len(self.base_map) + self.view_len * 2, len(self.base_map[0]) + self.view_len * 2, 3),
            fill_value=0,
            dtype=np.float32,
        )
        self.build_walls()
        self.custom_reset()
 
    def update_map_fire(
        self,
        firing_pos,
        firing_orientation,
        fire_len,
        fire_char,
        cell_types=[],
        update_char=[],
        blocking_cells=b"P",
        beam_width=3,
    ):
        """From a firing position, fire a beam that may clean or hit agents
 
        Notes:
            (1) Beams are blocked by agents
            (2) A beam travels along until it hits a blocking cell at which beam the beam
                covers that cell and stops
            (3) If a beam hits a cell whose character is in cell_types, it replaces it with
                the corresponding index in update_char
            (4) As per the rules, the beams fire from in front of the agent and on its
                sides so the beam that starts in front of the agent travels out one
                cell further than it does along the sides.
            (5) This method updates the beam_pos, an internal representation of how
                which cells need to be rendered with fire_char in the agent view
 
        Parameters
        ----------
        firing_pos: (list)
            the row, col from which the beam is fired
        firing_orientation: (string)
            the direction the beam is to be fired in
        fire_len: (int)
            the number of cells forward to fire
        fire_char: (bytes)
            the cell that should be placed where the beam goes
        cell_types: (list of bytes)
            the cells that are affected by the beam
        update_char: (list of bytes)
            the character that should replace the affected cells.
        blocking_cells: (list of bytes)
            cells that block the firing beam
        Returns
        -------
        updates: (tuple (row, col, char))
            the cells that have been hit by the beam and what char will be placed there
        """
        agent_by_pos = {tuple(agent.pos): agent_id for agent_id, agent in self.agents.items()}
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        # compute the other two starting positions
        right_shift = self.rotate_right(firing_direction)
        if beam_width == 1:
            firing_pos = [start_pos]
        elif beam_width == 3:
            firing_pos = [
                start_pos,
                start_pos + right_shift - firing_direction,
                start_pos - right_shift - firing_direction,
            ]
        else:
            raise NotImplementedError()
        firing_points = []
        updates = []
        hit_agent_rep = None
        for pos in firing_pos:
            next_cell = pos + firing_direction
            for i in range(fire_len):
                if (
                    self.test_if_in_bounds(next_cell)
                    and self.world_map[next_cell[0], next_cell[1]] != b"@"
                ):
                    # Update the cell if needed
                    firing_points.append((next_cell[0], next_cell[1], fire_char))
                    for c in range(len(cell_types)):
                        if self.world_map[next_cell[0], next_cell[1]] == cell_types[c]:
                            updates.append((next_cell[0], next_cell[1], update_char[c]))
                            break
 
                    # agents absorb beams
                    # activate the agents hit function if needed
                    # If agent is timed-out, will not get hit again
                    if [next_cell[0], next_cell[1]] in self.agent_pos:
                        agent_id = agent_by_pos[(next_cell[0], next_cell[1])]
                        if self.agents[agent_id].active:
                            hit_agent_rep = self.agents[agent_id].return_reputation()
                            self.agents[agent_id].hit(fire_char)
                            break
 
                    # check if the cell blocks beams. For example, waste blocks beams.
                    if self.world_map[next_cell[0], next_cell[1]] in blocking_cells:
                        break
 
                    # increment the beam position
                    next_cell += firing_direction
 
                else:
                    break
 
        self.beam_pos += firing_points
        return updates, hit_agent_rep
 
    def spawn_point(self):
        """Returns a randomly selected spawn point."""
        spawn_index = 0
        is_free_cell = False
        curr_agent_pos = [agent.pos.tolist() for agent in self.agents.values()]
        np.random.shuffle(self.spawn_points)
        for i, spawn_point in enumerate(self.spawn_points):
            if [spawn_point[0], spawn_point[1]] not in curr_agent_pos:
                spawn_index = i
                is_free_cell = True
        assert is_free_cell, "There are not enough spawn points! Check your map?"
        return np.array(self.spawn_points[spawn_index])
 
    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[rand_int]
 
    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.single_update_map(row, col, b"@")
 
    ########################################
    # Utility methods, move these eventually
    ########################################
 
    # TODO(ev) this can be a general property of map_env or a util
    def rotate_action(self, action_vec, orientation):
        # WARNING: Note, we adopt the physics convention that \theta=0 is in the +y direction
        if orientation == "UP":
            return action_vec
        elif orientation == "LEFT":
            return self.rotate_left(action_vec)
        elif orientation == "RIGHT":
            return self.rotate_right(action_vec)
        else:
            return self.rotate_left(self.rotate_left(action_vec))
 
    def rotate_left(self, action_vec):
        return np.dot(self.all_actions["TURN_COUNTERCLOCKWISE"], action_vec)
 
    def rotate_right(self, action_vec):
        return np.dot(self.all_actions["TURN_CLOCKWISE"], action_vec)
 
    # TODO(ev) this should be an agent property
    def update_rotation(self, action, curr_orientation):
        if action == "TURN_COUNTERCLOCKWISE":
            if curr_orientation == "LEFT":
                return "DOWN"
            elif curr_orientation == "DOWN":
                return "RIGHT"
            elif curr_orientation == "RIGHT":
                return "UP"
            else:
                return "LEFT"
        else:
            if curr_orientation == "LEFT":
                return "UP"
            elif curr_orientation == "UP":
                return "RIGHT"
            elif curr_orientation == "RIGHT":
                return "DOWN"
            else:
                return "LEFT"
 
    # TODO(ev) this definitely should go into utils or the general agent class
    def test_if_in_bounds(self, pos):
        """Checks if a selected cell is outside the range of the map"""
        return 0 <= pos[0] < self.world_map.shape[0] and 0 <= pos[1] < self.world_map.shape[1]
 
    def find_num_apples(self):
        count = 0
        for i in range (len(self.world_map)):
            for j in range (len(self.world_map[0])):
                if self.world_map[i,j] == b"A":
                    count += 1
        return np.float32(count)
 
    def num_apple_in_view(self):
        apple_in_view = dict()
        for agent_id, agent in self.agents.items():
            count = 0
            row, col = agent.pos[0], agent.pos[1]
            view_slice = self.world_map[
                max(0, row - self.view_len) : min(row +  self.view_len + 1, len(self.world_map)),
                max(0, col - self.view_len) : min(col + self.view_len + 1, len(self.world_map[0])),
            ]
            if agent.orientation == "UP":
                rotated_view = view_slice
            elif agent.orientation == "LEFT":
                rotated_view = np.rot90(view_slice)
            elif agent.orientation == "DOWN":
                rotated_view = np.rot90(view_slice, k=2)
            elif agent.orientation == "RIGHT":
                rotated_view = np.rot90(view_slice, k=1, axes=(1, 0))
            for i in range (len(rotated_view)):
                for j in range (len(rotated_view[0])):
                    if rotated_view[i,j] == b"A":
                        count += 1
            apple_in_view[agent_id] = count
        return apple_in_view
 
 
    def find_other_agent_reward(self, reward_list,agent_id):
        output = list()
        for id in sorted(self.agents.keys()):
            if id != agent_id:
                output.append(reward_list[id]["reward_this_episode"])
        return np.float32(output)
 
    # Returns whether other agents are active
    def find_active_agent_info(self,agent_id):
        output = list()
        for id in sorted(self.agents.keys()):
            if id != agent_id:
                output.append(self.agents[agent_id].active)
        return output
    
    def distance_to_apple(self):
        agent_pos = list()
        apple_pos = list()
        map = self.get_map_with_agents()
        if self.ep_count >= 500 and self.debug:
            print("=====in distance to apple=====")
            print(" map is:")
            print(map)
        for i in range(0, len(map)):
            for j in range(0, len(map[0])):
                if map[i,j] == b"1":
                    agent_pos = [i,j]
                    if self.ep_count >= 500 and self.debug:
                        print("agent pos is:", agent_pos)
                elif map[i,j] == b"A":
                    apple_pos = [i,j]
                    if self.ep_count >= 500 and self.debug:
                        print("apple pos is:", apple_pos)
        return abs(apple_pos[0] - agent_pos[0]) + abs((apple_pos[1] - agent_pos[1]))
    
    def find_visible_agents(self, agent_id):
        """Returns all the agents that can be seen by agent with agent_id
        Args
        ----
        agent_id: str
            The id of the agent whose visible agents we are asking about
        Returns
        -------
        visible_agents: list
            which agents can be seen by the agent with id "agent_id"
        """
        agent_pos = self.agents[agent_id].pos
        upper_lim = int(agent_pos[0] + self.agents[agent_id].row_size)
        lower_lim = int(agent_pos[0] - self.agents[agent_id].row_size)
        left_lim = int(agent_pos[1] - self.agents[agent_id].col_size)
        right_lim = int(agent_pos[1] + self.agents[agent_id].col_size)
 
        # keep this sorted so the visibility matrix is always in order
        other_agent_pos = [
            self.agents[other_agent_id].pos
            for other_agent_id in sorted(self.agents.keys())
            # Agent is not visible to others if timed-out
            if other_agent_id != agent_id# and self.agents[other_agent_id].active
        ]
        active_agent = [True if (self.agents[other_agent_id].active) else False for other_agent_id in sorted(self.agents.keys())]
        return np.array(
            [
                1
                if (lower_lim <= other_agent_pos[i][0] <= upper_lim and left_lim <= other_agent_pos[i][1] <= right_lim and active_agent[i] )
                else 0
                for i in range(0, len(other_agent_pos)) 
            ],
            dtype=np.float32,
        )
 
    def find_other_agents_reputation(self, agent_id):
        """Return the reputation of all other agents within the observation window, if not observable, return 0"""
        agent_pos = self.agents[agent_id].pos
        upper_lim = int(agent_pos[0] + self.agents[agent_id].row_size)
        lower_lim = int(agent_pos[0] - self.agents[agent_id].row_size)
        left_lim = int(agent_pos[1] - self.agents[agent_id].col_size)
        right_lim = int(agent_pos[1] + self.agents[agent_id].col_size)
 
        # keep this sorted so the visibility matrix is always in order
        other_agent_pos = [
            self.agents[other_agent_id].pos
            for other_agent_id in sorted(self.agents.keys())
            # Agent is not visible to others if timed-out
            if other_agent_id != agent_id and self.agents[other_agent_id].active
        ]
        other_agent_reputation = [
            self.agents[other_agent_id].reputation
            for other_agent_id in sorted(self.agents.keys())
            # Agent is not visible to others if timed-out
            if other_agent_id != agent_id and self.agents[other_agent_id].active
        ]
 
        return np.array(
            [
                other_agent_reputation[i]
                if (lower_lim <= other_agent_pos[i][0] <= upper_lim and left_lim <= other_agent_pos[i][1] <= right_lim)
                else 0
                for i in range(0, len(other_agent_pos))
            ],
        )
 
    @staticmethod
    def get_environment_callbacks():
        return DefaultCallbacks
 
    def update_apple_hist(self, count):
        self.apple_hist.append(count)
 
    def return_apple_hist(self):
        return self.apple_hist