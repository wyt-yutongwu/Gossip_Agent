import numpy as np
from numpy.random import rand
 
from ssd.envs.agent import HarvestAgent
from ssd.envs.gym.discrete_with_dtype import DiscreteWithDType
from ssd.envs.map_env import MapEnv
from ssd.maps import HARVEST_MAP, HARVEST_TEST_MAP
 
APPLE_RADIUS = 2
TIME_OUT_DURATION = 25
 
# Add custom actions to the agent
_HARVEST_ACTIONS = {"FIRE": 5}  # length of firing range
 
SPAWN_PROB = [0, 0.01,0.01,0.05, 0.05, 0.1] 
 
HARVEST_VIEW_SIZE = 5
 
 
class HarvestEnv(MapEnv):
    def __init__(
        self,
        ascii_map=HARVEST_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        use_reputation = False,
        alpha=0.0,
        beta=0.0,
        beam_penalty = -0.2,
        broadcast_mode = 0,
        broadcast_radius = 0,
        network_mode = False,
        adjacency_matrix = None,
    ):
        self.beam_penalty = beam_penalty
 
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
            use_reputation=use_reputation,
            broadcast_mode=broadcast_mode,
            broadcast_radius=broadcast_radius,
            network_mode=network_mode,
            adjacency_matrix=adjacency_matrix
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])
        self.apple_hist = []
 
 
    @property
    def action_space(self):
        return DiscreteWithDType(8, dtype=np.uint8)
 
    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()
 
        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE, time_out_duration=TIME_OUT_DURATION, num_agents = self.num_agents, beam_penalty=self.beam_penalty)
            self.agents[agent_id] = agent
 
    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")
        self.apple_hist = []
 
    def custom_action(self, agent, action):
        updates, hit_agent_rep = self.update_map_fire(
            agent.pos.tolist(),
            agent.get_orientation(),
            self.all_actions["FIRE"],
            fire_char=b"F",
        )
        agent.fire_beam(b"F", hit_agent_rep)
        return updates
    
 
    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)
 
    def spawn_apples(self):
        """Construct the apples spawned in this step.
 
        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """
 
        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                0 <= x + j < self.world_map.shape[0]
                                and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1
 
                spawn_prob = SPAWN_PROB[min(num_apples, 5)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points
 
    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples
 