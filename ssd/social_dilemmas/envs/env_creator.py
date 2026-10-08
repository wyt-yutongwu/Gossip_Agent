from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.switch import SwitchEnv
from social_dilemmas.maps import HARVEST_TEST_MAP,HARVEST_MAP_PAPER, HARVEST_SINGLE_AGENT
from social_dilemmas.envs.harvest_simple import HarvestSimpleEnv

def get_env_creator(
    env,
    num_agents,
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0,
    num_switches=6,
    use_reputation = False
):
    if env == "harvest":

        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                return_agent_actions=False,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                use_reputation = use_reputation
            )
    else:
        raise ValueError(f"env must be one of harvest, cleanup, switch, not {env}")

    return env_creator

def create_env(
    env,
    num_agents,
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0,
    num_switches=6,
    use_reputation = False,
    beam_penalty = -0.2,
    broadcast_mode = 0,
    broadcast_radius = 0,
    network_mode = False,
    adjacency_matrix = None
):
    if env == "harvest" and not use_reputation:
        return HarvestEnv(
            num_agents=num_agents,
            # return_agent_actions=False,
            return_agent_actions=False,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            ascii_map = HARVEST_MAP_PAPER,
            beta=beta,
            use_reputation = use_reputation,
            beam_penalty=beam_penalty
        )
    elif env == "harvest" and use_reputation:
        return HarvestEnv(
            num_agents=num_agents,
            # return_agent_actions=False,
            return_agent_actions=True,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
            use_reputation = use_reputation,
            ascii_map = HARVEST_MAP_PAPER,
            beam_penalty=beam_penalty,
            broadcast_mode = broadcast_mode,
            broadcast_radius = broadcast_radius,
            network_mode = network_mode,
            adjacency_matrix= adjacency_matrix
        )

    else:
        raise ValueError(f"env must be one of harvest, cleanup, switch, not {env}")