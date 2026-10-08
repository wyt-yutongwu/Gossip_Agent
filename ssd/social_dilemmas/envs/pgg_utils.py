"""Shared, algorithm-agnostic helpers for the public goods game.

Lives next to the environment because both things here are properties of the
*game*, not of a learner:

  * the network topologies reputation is broadcast over (PGG agents always live
    on a network),
  * the optional discrete investment ladder, for value-based learners that
    cannot act on a continuous interval.

Nothing in this module imports torch, and neither DQN nor PPO code needs to
import the other's package to get at it. Continuous-action learners (PPO with a
Beta or Gaussian head) ignore DiscreteInvestment entirely and act on the real
interval [0, 1] that PGGEnv exposes.
"""

import numpy as np


# ---------------------------------------------------------------------------
# discrete investment ladder (only for value-based / categorical policies)
# ---------------------------------------------------------------------------
class DiscreteInvestment(object):
    """A Discrete(n) space whose indices map onto investment levels."""

    def __init__(self, num_bins=11, low=0.0, high=1.0, dtype=np.int64):
        assert num_bins >= 2, "need at least two investment levels"
        self.n = int(num_bins)
        self.low = float(low)
        self.high = float(high)
        self.dtype = dtype
        self.bins = np.linspace(self.low, self.high, self.n)

    def sample(self):
        return int(np.random.randint(self.n))

    def contains(self, x):
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def to_investment(self, index):
        """index -> investment level."""
        return float(self.bins[int(index)])

    def to_index(self, investment):
        """investment level -> nearest index, for scripted/bootstrap policies."""
        return int(np.argmin(np.abs(self.bins - float(investment))))

    def __repr__(self):
        return "DiscreteInvestment({}, [{}, {}])".format(self.n, self.low, self.high)


# ---------------------------------------------------------------------------
# network topologies
# ---------------------------------------------------------------------------
def complete_graph(n):
    adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, 0)
    return adj


def ring_graph(n):
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj[i][(i + 1) % n] = 1
        adj[i][(i - 1) % n] = 1
    np.fill_diagonal(adj, 0)
    return adj


def path_graph(n):
    adj = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj[i][i + 1] = 1
        adj[i + 1][i] = 1
    return adj


def star_graph(n):
    adj = np.zeros((n, n), dtype=int)
    for i in range(1, n):
        adj[0][i] = 1
        adj[i][0] = 1
    return adj


def erdos_renyi_graph(n, p=0.5, seed=None):
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i][j] = 1
                adj[j][i] = 1
    return adj


NETWORKS = {
    "complete": complete_graph,
    "ring": ring_graph,
    "path": path_graph,
    "star": star_graph,
}


def build_network(name, num_agents, p=0.5, seed=None):
    if name == "random":
        return erdos_renyi_graph(num_agents, p=p, seed=seed)
    if name not in NETWORKS:
        raise ValueError(
            "unknown network '{}', pick one of {} or 'random'".format(
                name, sorted(NETWORKS)
            )
        )
    return NETWORKS[name](num_agents)


# ---------------------------------------------------------------------------
# observation flattening
# ---------------------------------------------------------------------------
# The environments return a dict observation:
#
#   "curr_obs"   the agent's opinions about everyone else. Always present.
#   "own_rep"    the agent's OWN standing -- the figure its reputation bonus is
#                computed from. Emitted by PDEnv when observe_own_rep is on;
#                PGGEnv does not emit it, so it simply drops out of the list
#                there and that env's input width is unchanged.
#   "table_mask" which slots of curr_obs hold a real opponent, under
#                PDEnv's obs_scope="next_table". Appears only when the table
#                size does not divide the player count, so that a short table
#                or a stranded agent can be told from a neutral opinion.
#   "is_playing" whether the agent is drawn for the coming round. Appears only
#                when num_players < num_agents.
#
# A learner that reads only "curr_obs" silently throws the rest away, so the
# agent cannot condition on whether it is in this round, nor on how it is
# currently regarded -- while still being PAID for how it is regarded. That
# leaves the reputation term as an unattributable drift in the reward. Adding a
# key here is what makes it reach the network.
#
# THIS TUPLE IS THE WHITELIST. A key the env emits but this tuple omits is
# dropped without any error, so it must be extended whenever an env grows an
# observation entry. pd_multi_agent raises at construction if that happens.
# The order is fixed because flatten_obs concatenates in it; changing the
# order or the contents changes the MLP's input layout, which invalidates any
# checkpoint trained against the previous one.
OBS_KEYS = ("curr_obs", "own_rep", "table_mask", "is_playing")


def obs_keys_present(obs, keys=OBS_KEYS):
    """The subset of OBS_KEYS this environment actually emits, in fixed order."""
    return [k for k in keys if k in obs]


def flatten_obs(obs, keys):
    """Concatenate the named entries of one agent's observation into a vector."""
    return np.concatenate(
        [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in keys]
    ).astype(np.float32)


def obs_dim(obs, keys):
    """Width of the flattened vector, i.e. the MLP's input size."""
    return int(sum(np.asarray(obs[k]).size for k in keys))