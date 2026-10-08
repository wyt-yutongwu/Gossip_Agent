"""Base class for an agent that defines the possible actions. """
 
import numpy as np
from collections import Counter, deque
 
import utility_funcs as util
 
# basic moves every agent should do
BASE_ACTIONS = {
    0: "MOVE_LEFT",  # Move left
    1: "MOVE_RIGHT",  # Move right
    2: "MOVE_UP",  # Move up
    3: "MOVE_DOWN",  # Move down
    4: "STAY",  # don't move
    5: "TURN_CLOCKWISE",  # Rotate counter clockwise
    6: "TURN_COUNTERCLOCKWISE",
}  # Rotate clockwise
 

 
class Agent(object):
    def __init__(self, agent_id, start_pos, start_orientation, full_map, row_size, col_size, num_agents, max_len = 10, network_neighbor = None):
        """Superclass for all agents.
 
        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        start_pos: (np.ndarray)
            a 2d array indicating the x-y position of the agents
        start_orientation: (np.ndarray)
            a 2d array containing a unit vector indicating the agent direction
        full_map: (2d array)
            a reference to this agent's view of the environment
        row_size: (int)
            how many rows up and down the agent can look
        col_size: (int)
            how many columns left and right the agent can look
        """
        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.orientation = start_orientation
        self.full_map = full_map
        self.row_size = row_size
        self.col_size = col_size
        self.reward_this_turn = 0
        self.prev_visible_agents = None
        self.active = True
        self.decay = 0.99
        self.max_len = max_len
        self.network_neighbor = network_neighbor

        #peacefulness
        self.time_out_this_turn = 0
        # utility
        self.reward_so_far = 0
        # reputation
        self.reputation = 0
        self.other_agent_rep = {"agent-"+str(i):0 for i in range(num_agents)}
        self.rep_hist = deque([], maxlen=self.max_len)
        self.total_good_hit = 0
        # sustainability
        self.collection_time = 0
        self.tmp_other_agent_rep = {"agent-"+str(i):0 for i in range(num_agents)}
        self.action_hist = []
 
    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.
 
        MUST BE implemented in new environments.
 
        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError
 
    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.
 
        MUST BE implemented in new environments.
 
        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError
 
    def add_to_action_hist(self, action):
        self.action_hist.append(action)
    
    def return_action_hist(self):
        return self.action_hist
        
    # def post_process_action_hist(self):
    #     output = list()
    #     item_counts = Counter(self.action_hist)
    #     for i in range(0, 8):
    #         if i not in item_counts:
    #             output.append(0)
    #         else:
    #             output.append(item_counts[i])
    #     return output
 
    def return_total_good_hit(self):
        return self.total_good_hit
    
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError
 
    def get_char_id(self):
        return bytes(str(int(self.agent_id[-1]) + 1), encoding="ascii")
 
    def get_state(self):
        return util.return_view(self.full_map, self.pos, self.row_size, self.col_size)
 
    def compute_reward(self):
        reward = self.reward_this_turn
        self.reward_so_far += reward
        self.reward_this_turn = 0
        return reward, self.reward_so_far
 
    def compute_time_out(self):
        time_out = self.time_out_this_turn
        return time_out
 
    def return_reputation(self):
        return self.reputation
 
    def set_pos(self, new_pos):
        self.pos = np.array(new_pos)
 
    def get_pos(self):
        return self.pos
 
    def add_rep(self, amount):
        self.rep_hist.append(amount)
 
    def calculate_rep(self):
        avg = sum(self.rep_hist) / len(self.rep_hist)
        return avg
 
    def translate_pos_to_egocentric_coord(self, pos):
        offset_pos = pos - self.pos
        ego_centre = [self.row_size, self.col_size]
        return ego_centre + offset_pos
 
    def set_orientation(self, new_orientation):
        self.orientation = new_orientation
 
    def get_orientation(self):
        return self.orientation
    
    def return_collection_time(self):
        return self.collection_time
    
    def adjust_other_agent_rep(self, agent_id, amount):
        if amount != 0:
            self.other_agent_rep[agent_id] = 0.9 * self.other_agent_rep[agent_id] + amount * 0.1
            
    def adjust_other_agent_rep_broadcast(self, agent_id, amount):
        self.other_agent_rep[agent_id] = amount
            
    def calculate_tmp_rep(self, agent_id, amount):
        return 0.5 * self.other_agent_rep[agent_id] + amount * 0.5

    def get_other_agent_rep(self):
        return self.other_agent_rep
 
    def return_valid_pos(self, new_pos):
        """Checks that the next pos is legal, if not return current pos"""
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # You can't walk through walls, closed doors or switches
        if self.is_tile_walkable(new_row, new_col):
            return new_pos
        else:
            return self.pos
 
    def update_agent_pos(self, new_pos):
        """Updates the agents internal positions
 
        Returns
        -------
        old_pos: (np.ndarray)
            2 element array describing where the agent used to be
        new_pos: (np.ndarray)
            2 element array describing the agent positions
        """
        old_pos = self.pos
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        if self.is_tile_walkable(new_row, new_col):
            validated_new_pos = new_pos
        else:
            validated_new_pos = self.pos
        self.set_pos(validated_new_pos)
        # TODO(ev) list array consistency
        return self.pos, np.array(old_pos)
 
    def is_tile_walkable(self, row, column):
        return (
            0 <= row < self.full_map.shape[0]
            and 0 <= column < self.full_map.shape[1]
            # You can't walk through walls, closed doors or switches
            and self.full_map[row, column] not in [b"@", b"D", b"w", b"W"]
        )
 
    def update_agent_rot(self, new_rot):
        self.set_orientation(new_rot)
 
    def hit(self, char):
        """Defines how an agent responds to being hit by a beam of type char"""
        raise NotImplementedError
 
    def consume(self, char, step_num):
        """Defines how an agent interacts with the char it is standing on"""
        raise NotImplementedError
 
 
HARVEST_ACTIONS = BASE_ACTIONS.copy()
HARVEST_ACTIONS.update({7: "FIRE"})  # Fire a penalty beam
 
 
class HarvestAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, full_map, view_len, time_out_duration, num_agents, beam_penalty):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, full_map, view_len, view_len, num_agents=num_agents)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
        self.time_out_remaining = 0
        self.time_out_duration = time_out_duration
        self.beam_attempt = 0
        self.successful_hit = 0
        self.good_hit = 0
        # self.beam_penalty = beam_penalty
 
 
    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return HARVEST_ACTIONS[action_number]
    
    def update_time_out(self):
        if self.time_out_remaining > 0:
            self.time_out_remaining -= 1
            if self.time_out_remaining == 0:
                self.active = True
 
    def hit(self, char):
        if self.time_out_remaining > 0:
            print("ERROR")
        if char == b"F":
            self.time_out_remaining = self.time_out_duration
            self.time_out_this_turn += self.time_out_duration
            self.active = False
 
    def get_beam_attemp(self):
        return self.beam_attempt
    
    def get_successful_hit(self):
        return self.successful_hit
 
    def good_hit_this_turn(self):
        good_hit = self.good_hit
        self.good_hit = 0
        if good_hit > 0:
            self.total_good_hit += good_hit
        return good_hit
 
    def fire_beam(self, char, hit_agent_rep):
        if self.time_out_remaining > 0:
            print("Error in fire beam")
        # deducts reputation when a good agent (reputation >= 0) is hit
        # increase reputation when a bad agent (reputation < 0) is hit 
        if char == b"F":
            self.beam_attempt += 1
            if hit_agent_rep is None:
                return
            elif hit_agent_rep >= 0:
                self.successful_hit += 1
                self.good_hit = -1
            else:
                self.successful_hit += 1
                self.good_hit = 1
    

 
    def get_done(self):
        return False
 
    def consume(self, char,step_num):
        if not self.active:
            return char
        """Defines how an agent interacts with the char it is standing on"""
        if char == b"A":
            self.reward_this_turn += 1
            collection_time = step_num
            self.collection_time += collection_time
            return b" "
        else:
            return char

COOPERATE = 0
DEFECT = 1
PD_ACTIONS = {COOPERATE: "COOPERATE", DEFECT: "DEFECT"}
 
 
class PDAgent(Agent):
    def __init__(self, agent_id, num_agents, max_len=10, network_neighbor=None):
        """Agent for the n-player Prisoner's Dilemma.
 
        Parameters
        ----------
        agent_id: (str)
            unique id of the agent, e.g. "agent-0"
        num_agents: (int)
            number of agents in the game, used to size the reputation table
        max_len: (int)
            length of the reputation history used to smooth the reputation
        network_neighbor: (set or None)
            the agents this one is connected to; in the PD environment this is
            the ONLY channel it can observe or broadcast through
        """
        dummy_map = np.full((1, 1), b" ", dtype="c")
        super().__init__(
            agent_id,
            start_pos=[0, 0],
            start_orientation="UP",
            full_map=dummy_map,
            row_size=0,
            col_size=0,
            num_agents=num_agents,
            max_len=max_len,
            network_neighbor=network_neighbor,
        )
        self.num_agents = num_agents
 
        # action bookkeeping
        self.action = COOPERATE          # action chosen this turn
        self.prev_action = COOPERATE     # action chosen last turn
        self.num_steps = 0               # steps played this episode
        self.coop_steps = 0              # steps spent cooperating
        self.defect_steps = 0            # steps spent defecting
        self.last_reward = 0.0           # payoff last turn, after shaping
 
        # kept so code written for the harvest agents keeps working
        self.beam_attempt = 0
        self.successful_hit = 0
        self.good_hit = 0
 
    @property
    def action_space(self):
        """Two discrete actions; the gym space itself lives on the environment."""
        return (COOPERATE, DEFECT)
 
    def action_map(self, action_number):
        """Maps a raw policy output to COOPERATE (0) or DEFECT (1).
 
        Accepts an int, a float, a 0-d/1-d numpy array or a list. Floats are
        rounded, so a policy head that emits 0.0/1.0 -- e.g. a categorical PPO
        head routed through an investment ladder, or a DQN index wrapper -- maps
        onto the two actions without any glue code.
        """
        action = np.asarray(action_number, dtype=np.float64).reshape(-1)
        if action.size != 1:
            raise ValueError(
                "PD action must be a single scalar, got shape {}".format(
                    np.shape(action_number)
                )
            )
        value = float(action[0])
        if not np.isfinite(value):
            value = float(COOPERATE)
        return int(np.clip(round(value), COOPERATE, DEFECT))
 
    def act(self, action_number):
        """Commits this turn's action and updates the action statistics.
 
        Returns
        -------
        action: (int) COOPERATE or DEFECT
        """
        action = self.action_map(action_number)
        self.prev_action = self.action
        self.action = action
        self.num_steps += 1
        if action == COOPERATE:
            self.coop_steps += 1
        else:
            self.defect_steps += 1
        self.add_to_action_hist(action)
        return action
 
    def sit_out(self):
        """This round the agent was not drawn to play.
 
        None of the episode statistics move: the round is not counted as a step
        it played, so coop_rate() stays the average over the rounds it was
        actually in, and defect_steps is not inflated by rounds it never got to
        act in.
        """
        self.prev_action = self.action
        return None
 
    def add_reward(self, amount):
        """Adds this turn's matrix-game payoff to the agent"""
        self.reward_this_turn += float(amount)
 
    def cooperated(self):
        return self.action == COOPERATE
 
    def coop_rate(self):
        if self.num_steps == 0:
            return 0.0
        return self.coop_steps / self.num_steps
 
    # ------------------------------------------------------------------
    # API parity with the grid-world agents
    # ------------------------------------------------------------------
    def update_time_out(self):
        """No time-out mechanic in the Prisoner's Dilemma"""
        return
 
    def get_char_id(self):
        """No map rendering in the Prisoner's Dilemma"""
        return b"P"
 
    def get_state(self):
        """The agent's state is its reputation view, not a map slice"""
        return np.array(
            [self.other_agent_rep["agent-" + str(i)] for i in range(self.num_agents)],
            dtype=np.float32,
        )
 
    def get_beam_attemp(self):
        return self.beam_attempt
 
    def get_successful_hit(self):
        return self.successful_hit
 
    def good_hit_this_turn(self):
        return 0
 
    def fire_beam(self, char, hit_agent_rep=None):
        """There is no punishment beam in the Prisoner's Dilemma"""
        return
 
    def hit(self, char):
        """The agent cannot be hit in the Prisoner's Dilemma"""
        return
 
    def consume(self, char, step_num=None):
        """Nothing to consume: the payoff is handed out by the environment"""
        return char
 
    def get_done(self):
        return False