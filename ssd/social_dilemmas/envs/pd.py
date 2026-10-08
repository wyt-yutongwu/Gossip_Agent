"""n-agent Prisoner's Dilemma with group-based reputation communication.

Payoffs
-------
The scheme from the paper, set by four corner values:

      every agent cooperates  ->  every agent gets all_coop_payoff
      every agent defects     ->  every agent gets all_defect_payoff
      mixed                   ->  cooperators get coop_mixed_payoff,
                                  defectors get defect_mixed_payoff

Leaving all_coop_payoff and defect_mixed_payoff as None uses the paper's
n-dependent defaults, n and n + 1; setting them fixes a matrix that does not
change with the agent count. play_mode="pairwise" reads the same four values
as the 2x2 matrix, so they are the only payoff knobs in either shape.

For n = 2 the threshold scheme is the classic matrix, T = 3 > R = 2 > P = 1 >
S = 0. All-C is the social optimum and all-D is the unique Nash equilibrium.

Like PGGEnv this is a non-spatial repeated game, so it subclasses MultiAgentEnv
directly rather than MapEnv, and it carries the same reputation machinery with
"invested below the fair share" replaced by "defected".

Communication
-------------
Every agent plays every round. What changes from round to round is who an agent
can see and talk to, set by network_mode:

  "static"  -- a fixed graph from adjacency_matrix (the complete graph by
               default). The classic setting.
  "groups"  -- NO fixed network. Each round the whole population is randomly
               partitioned into disjoint groups of `group_size`, and an agent
               sees and gossips with exactly its groupmates for that round.
               Information about anyone else can only arrive second-hand, one
               group-hop per round, so `group_size` is the single knob
               controlling how fast reputation spreads through the population.

               NB this makes group_size control BOTH the gossip structure and
               how much an agent observes first-hand. Set
               observe_neighbors_only=False to hold the observation window
               fixed across group sizes and vary only the gossip.
  "sample"  -- a pure gossip channel. Each round every agent hears reputations
               from `comm_j` random others, drawn fresh and directed. Nothing
               about observation is sampled: who saw you act is decided by the
               round itself -- your table under play_mode="pairwise", everyone
               who played under play_mode="group". This is the mode to use when
               the gossip reach has to vary with the observation window held
               fixed.
  "table"   -- pairwise play only: you gossip with exactly the opponents you
               just played. The gossip channel and the observation window
               collapse onto the same set, so second-hand news travels only
               through agents who judged you first-hand -- one table-hop per
               round. Not tunable: the reach is fixed at pair_size - 1.
  "players" -- the complete graph over whoever was drawn to play, for the
               partial-participation setting (num_players < num_agents).

In every mode the three channels -- who you see act, whose reputation you can
form an opinion about, and whose opinion table yours is averaged with -- run
along the same edges.
"""

import numpy as np
from gymnasium.spaces import Box, Dict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import MultiAgentEnv

from ssd.envs.agent import COOPERATE, DEFECT, PDAgent
from ssd.envs.gym.discrete_with_dtype import DiscreteWithDType


class PDEnv(MultiAgentEnv):
    def __init__(
        self,
        num_agents=5,
        episode_length=1000,
        # payoffs: None means "use the scheme from the paper", which depends on n
        all_coop_payoff=None,        # n
        all_defect_payoff=1.0,       # 1
        defect_mixed_payoff=None,    # n + 1
        coop_mixed_payoff=0.0,       # 0
        use_reputation=False,
        rep_weight=1.0,
        # communication
        adjacency_matrix=None,
        observe_neighbors_only=True,
        network_mode="static",
        group_size=None,
        comm_j=None,
        # partial participation: k of the n agents are drawn to play each round
        nonplayers_observe=True,
        # how the four payoff levels are laid out
        # pairwise play inside randomly drawn tables
        play_mode="group",
        pair_size=None,
        pair_payoff="mean",
        observe_own_rep=True,
        gossip_mode="average",
        opinion_rule="image",
        obs_scope="all",
        rep_memory=10,
    ):
        """
        Parameters
        ----------
        num_agents: int
            Number of agents playing the game.
        episode_length: int or None
            Rounds per episode; the episode is truncated (not terminated) at
            this many steps. None runs forever.
        all_coop_payoff, all_defect_payoff, defect_mixed_payoff, coop_mixed_payoff:
            The four payoff levels. Leaving the two None values alone gives
            the paper's n-dependent scheme; set them to fix a matrix
            independent of the agent count. play_mode="pairwise" reads the
            same four values as the corners of its 2x2 matrix -- R, P, T, S in
            that order -- so they govern both shapes of the round.
        use_reputation: bool
            Add rep_weight * own reputation to the reward.
            False is the no-reputation baseline, and it changes the
            OBSERVATION as well as the reward: an agent with no reputation
            system has no opinions to look at, so "curr_obs" holds what its
            coming opponents did LAST ROUND instead -- +1 cooperated, -1
            defected, 0 did not play (or there was no last round). Under
            play_mode="pairwise" these are the pair_size - 1 agents it is
            about to share a table with, sorted ascending exactly like the
            obs_scope="next_table" reputations, so the tables are cut one
            round early whatever obs_scope says. Under play_mode="group"
            everyone is an opponent, so it is every other agent in id order.
            "own_rep" is left out of the baseline observation.
        rep_weight: float
            Weight of the reputation term added to a player's reward, so the
            bonus swings over +-rep_weight as the agent's public reputation
            moves over [-1, 1]. Only paid to agents that actually played.
        adjacency_matrix: (num_agents, num_agents) array or None
            adjacency_matrix[i][j] != 0 means agent-i and agent-j are
            neighbours. None defaults to the complete graph. Only used by
            network_mode="static"; the other modes draw their own graph and
            require adjacency_matrix to be None.
        observe_neighbors_only: bool
            True (default): an agent only sees the actions of the agents it is
            currently connected to -- its static neighbours, or its groupmates
            under network_mode="groups".
            False: actions are public and only the broadcast is network-bound.
        network_mode: str
            Who an agent can see and communicate with.
            "static" (default): the fixed graph from adjacency_matrix.
            "groups": no fixed network at all. Every round the population is
            randomly partitioned into disjoint groups of `group_size`, and an
            agent's neighbours are its groupmates for that round only. Every
            agent is in exactly one group every round, so nobody is left out;
            what varies is who they are with.
            "sample": gossip only. Every round each agent picks comm_j others
            at random to hear reputations from. Observation is NOT sampled --
            it is whatever the round's play structure implies -- so j moves
            the gossip reach with the observation window held fixed, which
            "groups" cannot do because there one number sets both.
            "table": play_mode="pairwise" only -- your gossip partners ARE the
            opponents you just played. Equivalent to "groups" under pairwise
            play, which takes its partition from the tables; "table" is the
            name that says so, and it refuses group_size and the group game
            rather than silently accepting them. Use it for the tightest
            locality available: news about you reaches an agent only via
            someone who faced you, one table-hop per round. The reach is
            pinned at pair_size - 1, so this is one design point rather than a
            knob -- "sample" with comm_j is what varies the gossip reach with
            the observation window held fixed.
            "players": the complete graph over the agents drawn to play, for
            the partial-participation setting.
        group_size: int or None
            k, the size of the reputation-sharing groups drawn each round.
            Required by network_mode="groups" and ignored otherwise. Must be
            between 2 and num_agents. When k does not divide num_agents the
            last group holds the remainder; a remainder of exactly one agent is
            folded into the previous group instead of being left alone, so no
            agent is ever isolated while it has someone to be grouped with.
            k = num_agents reproduces the complete graph, k = 2 gives random
            pairings, and everything between interpolates.
        comm_j: int or None
            j, how many other agents each agent averages its opinion table with
            per round under network_mode="sample". Drawn fresh every round,
            independently of the observation draw, so an agent can gossip with
            agents it did not watch and vice versa. 0 switches gossip off
            entirely, leaving pure first-hand reputation. Also directed: i
            hears from its j picks, who do not thereby hear from i.
        nonplayers_observe: bool
            Whether an agent that sat out still forms first-hand opinions about
            the agents that played. Only relevant with num_players < num_agents.
        play_mode: str
            The shape of the round.
            "group" (default): one simultaneous n-player game. Everybody at
            the table moves at once and the payoff is a function of how many
            cooperated.
            "pairwise": the population is cut into random tables of pair_size
            and every table plays a round robin of two-player games. An agent
            picks ONE action and plays that same action against each of its
            tablemates, so the choice is still a single commitment but the
            payoff is the aggregate of |table| - 1 separate encounters, scored
            with the ordinary 2x2 matrix, whose corners are the four payoff
            levels. payoff_over is not consulted in this mode.
        pair_size: int or None
            k, the number of agents at a table in play_mode="pairwise". The
            population is shuffled and cut into consecutive blocks of k; the
            final block keeps whatever is left over and plays as a smaller
            table. A leftover of exactly one agent has nobody to play, so that
            agent sits the round out -- it acts on nothing, earns nothing and
            is judged by nobody, exactly like a non-participant under
            num_players.
        pair_payoff: str
            How the encounters at a table are aggregated into one reward.
            "mean" (default): average over opponents, so an agent at a table
            of 3 and an agent at a table of 7 are on the same scale. This
            matters because the leftover table is usually smaller than the
            rest, and because a k-sweep otherwise confounds the dilemma with
            the number of games played.
            "sum": total over opponents, the convention in the spatial-MARL
            literature (e.g. LR2's environmental reward). Reward magnitude
            then grows roughly linearly in k.
        gossip_mode: str
            How a broadcast combines what an agent already believes with what
            it hears. This is the knob that decides whether comm_j can matter
            at all.
            "average" (default): replace your table with the mean of yours and
            your neighbours'. This is row-stochastic, so it contracts the
            spread of opinion toward consensus, and doubly stochastic in
            expectation, so it preserves the population mean exactly. Raising
            comm_j therefore destroys the disagreement a private reward reads
            while provably moving no aggregate -- gossip cannot help, and
            measurably hurts.
            "delta": add the mean CHANGE in your neighbours' tables since the
            last broadcast to your own level. cur - prev is what they learned
            first-hand this round, so the news arrives without overwriting
            what you saw yourself. Neither row-stochastic nor mean-preserving.
            Follows the gossip update in Song et al.'s COOPER baseline.
        observe_own_rep: bool
            True (default): the observation carries an "own_rep" entry holding
            the reputation figure this agent is currently paid on -- the same
            number reward_reputation() returns -- the public reputation.
            Without it an agent is rewarded for its standing while being
            unable to see that standing, which leaves it nothing to condition
            on: the reputation term becomes an unattributable drift in the
            reward rather than a consequence of its own last action. Both LR2
            and COOPER feed the agent its own standing for this reason.
            False reproduces the older, blind observation.
        rep_memory: int
            Length of the reputation history used to smooth the reputation.
        """
        self.debug = False
        self._num_agents = num_agents
        self.episode_length = episode_length

        # ---------------- participation ----------------
        self.num_players = num_agents 
        assert 1 <= self.num_players <= num_agents, \
            "num_players must be between 1 and num_agents"
        self.partial_participation = self.num_players < num_agents
        print("partial participation:", self.partial_participation)
        self.nonplayers_observe = bool(nonplayers_observe)
        self.rng = np.random.default_rng()

        # ---------------- payoffs ----------------
        payoff_n = self.num_players 
        self.payoff_n = payoff_n
        self.all_coop_payoff = float(payoff_n if all_coop_payoff is None
                                     else all_coop_payoff)
        self.all_defect_payoff = float(all_defect_payoff)
        self.defect_mixed_payoff = float(payoff_n + 1 if defect_mixed_payoff is None
                                         else defect_mixed_payoff)
        self.coop_mixed_payoff = float(coop_mixed_payoff)
        print("payoff levels:", self.payoff_levels())

        # ---------------- reputation ----------------
        self.use_reputation = use_reputation
        print("use reputation:", self.use_reputation)
        self.rep_weight = rep_weight
        print("rep weight is:", self.rep_weight)
        self.rep_memory = rep_memory
        self.observe_neighbors_only = observe_neighbors_only
        self.observe_own_rep = bool(observe_own_rep)
        print("agents observe their own reputation:", self.observe_own_rep)
        assert obs_scope in ("all", "next_table"), \
            "obs_scope must be 'all' or 'next_table'"
        if obs_scope != "all":
            assert play_mode == "pairwise", \
                ("obs_scope='next_table' needs play_mode='pairwise' -- there "
                 "is no table to look ahead to in the group game")
        self.obs_scope = obs_scope
        # a short table, or a stranded agent, is possible only when the table
        # size does not divide the number of players; when it does, every
        # agent has exactly pair_size - 1 opponents every round and the
        # validity mask would be a constant
        # no reputation -> nothing to hold an opinion in, so the agent is
        # shown its coming opponents' last actions instead
        self.observe_actions = not self.use_reputation
        # the observation names the coming opponents, so their tables have to
        # be cut one round early
        self.lookahead = play_mode == "pairwise" and (
            obs_scope == "next_table" or self.observe_actions)
        self.table_remainder = (self.num_players % int(pair_size)
                                if self.lookahead and pair_size is not None
                                else 0)
        self.table_mask = self.table_remainder != 0
        print("observation scope:", self.obs_scope)
        if self.observe_actions:
            if self.lookahead and pair_size is not None:
                print("observation (no reputation): last-round actions of the "
                      "{} coming opponents{}, sorted ascending; +1 cooperated, "
                      "-1 defected, 0 did not play".format(
                          int(pair_size) - 1,
                          " + a validity mask (tables are not all full)"
                          if self.table_mask else " (every table is full)"))
            else:
                print("observation (no reputation): last-round actions of "
                      "every other agent; +1 cooperated, -1 defected, 0 did "
                      "not play")
        elif obs_scope == "next_table":
            print("observation: {} coming-opponent reputations{}, sorted "
                  "ascending, plus own_rep".format(
                      pair_size - 1,
                      " + a validity mask (tables are not all full)"
                      if self.table_mask else " (every table is full)"))
        assert opinion_rule in ("image", "clean_hands"), \
            "opinion_rule must be 'image' or 'clean_hands'"
        self.opinion_rule = opinion_rule
        print("opinion rule:", self.opinion_rule)
        assert gossip_mode in ("average", "delta"), \
            "gossip_mode must be 'average' or 'delta'"
        self.gossip_mode = gossip_mode
        print("gossip mode:", self.gossip_mode)
        self.prev_other_agent_rep = {}

        # ---------------- shape of the round ----------------
        assert play_mode in ("group", "pairwise"), \
            "play_mode must be 'group' or 'pairwise'"
        self.play_mode = play_mode
        assert pair_payoff in ("mean", "sum"), \
            "pair_payoff must be 'mean' or 'sum'"
        self.pair_payoff = pair_payoff
        if play_mode == "pairwise":
            assert pair_size is not None, "play_mode='pairwise' needs a pair_size"
            pair_size = int(pair_size)
            assert 2 <= pair_size <= num_agents, \
                "pair_size must be between 2 and num_agents, got {}".format(pair_size)
            # the 2x2 corners are the four payoff levels
            t, r = self.defect_mixed_payoff, self.all_coop_payoff
            p, s = self.all_defect_payoff, self.coop_mixed_payoff
            assert t > r > p > s, \
                ("pairwise play needs a proper 2x2 dilemma t > r > p > s, got "
                 "t={} r={} p={} s={}. Left as None these come from the "
                 "n-dependent defaults; pass the four levels explicitly."
                 .format(t, r, p, s))
            if all_coop_payoff is None or defect_mixed_payoff is None:
                print("WARNING: pairwise play with the n-dependent defaults "
                      "makes every encounter scale with num_agents. Pass "
                      "all_coop_payoff and defect_mixed_payoff explicitly for "
                      "a matrix that is fixed across agent counts.")
        self.pair_size = None if pair_size is None else int(pair_size)
        if play_mode == "pairwise":
            print("pairwise play: random tables of {}, scored by the {} of "
                  "{} encounters".format(self.pair_size, self.pair_payoff,
                                         self.pair_size - 1))
            print("pair matrix: R={} S={} T={} P={}".format(
                self.all_coop_payoff, self.coop_mixed_payoff,
                self.defect_mixed_payoff, self.all_defect_payoff))

        # ---------------- communication structure ----------------
        assert network_mode in ("static", "groups", "sample", "players", "table"), \
            ("network_mode must be 'static', 'groups', 'sample', 'players' or "
             "'table'")
        if network_mode == "table":
            assert play_mode == "pairwise", \
                ("network_mode='table' means 'gossip with the opponents you "
                 "just played', so it only exists under play_mode='pairwise'. "
                 "Use 'groups' for a gossip partition in the group game.")
            assert group_size is None, \
                ("network_mode='table' takes its groups from the tables "
                 "themselves, so group_size must be left unset -- pair_size is "
                 "the only group size in this setup")
        self.network_mode = network_mode
        if network_mode != "static":
            assert adjacency_matrix is None, \
                ("network_mode='{}' draws its own graph, so a fixed "
                 "adjacency_matrix cannot also apply".format(network_mode))
        if network_mode == "groups" and play_mode == "pairwise":
            # the tables already partition the population, so they supply the
            # gossip groups too -- a separate group_size would be a second,
            # invisible partition that draw_groups never consults
            assert group_size is None, \
                ("network_mode='groups' with play_mode='pairwise' gossips "
                 "along the tables themselves, so group_size must be left "
                 "unset -- pair_size is the only group size in this setup")
        elif network_mode == "groups":
            assert group_size is not None, \
                "network_mode='groups' needs a group_size"
            group_size = int(group_size)
            assert 2 <= group_size <= num_agents, \
                "group_size must be between 2 and num_agents, got {}".format(group_size)
        self.group_size = None if group_size is None else int(group_size)
        if network_mode == "sample":
            assert comm_j is not None, "network_mode='sample' needs a comm_j"
            comm_j = int(comm_j)
            assert 0 <= comm_j <= num_agents - 1, \
                "comm_j must be between 0 and num_agents - 1"
        self.comm_j = None if comm_j is None else int(comm_j)
        if network_mode == "sample":
            print("communication: each agent hears reputations from {} others, "
                  "redrawn each round; observation is set by the round's play "
                  "structure".format(self.comm_j))
        if network_mode == "groups":
            print("communication: random disjoint groups of {} redrawn "
                  "each round".format(self.group_size))
        if network_mode == "table":
            print("communication: each agent gossips with exactly the {} "
                  "opponents it just played".format(pair_size - 1))

        # keep track of episode and step num
        self.ep_count = 0
        self.count = 0

        # the static network; only consulted by network_mode="static"
        if adjacency_matrix is None:
            self.adjacency_matrix = np.ones((num_agents, num_agents), dtype=int)
            np.fill_diagonal(self.adjacency_matrix, 0)
        else:
            self.adjacency_matrix = np.asarray(adjacency_matrix)
        assert self.adjacency_matrix.shape == (num_agents, num_agents), \
            "adjacency_matrix must have shape (num_agents, num_agents)"
        self.network_neighbor_map = {
            "agent-" + str(i): {
                "agent-" + str(j)
                for j in range(num_agents)
                if i != j and self.adjacency_matrix[i][j]
            }
            for i in range(num_agents)
        }
        if network_mode == "static":
            print("agent_neighbor map: ", self.network_neighbor_map)

        self.agents = {}
        self.setup_agents()
        print(self.agents.keys())
        self._agent_ids = set(self.agents.keys())
        self.agent_ids_sorted = sorted(self.agents.keys())

        # actions of the current round, keyed by agent id
        self.actions = {agent_id: COOPERATE for agent_id in self.agents}
        # what each agent did in the round just played; None if it sat out,
        # or if no round has been played yet this episode
        self.last_actions = {agent_id: None for agent_id in self.agents}
        # who is drawn for the upcoming round, and for the round being played
        self.playing = self.sample_players()
        self.players_this_round = set(self.playing)
        # this round's tables (play_mode="pairwise") and who sat out for lack
        # of an opponent
        self.pair_groups = []
        self.pair_group_map = {}
        self.pair_sitouts = set()
        # the tables for the round the agents are ABOUT to play, drawn one
        # step early so the observation can name the coming opponents
        self.next_pair_groups = []
        self.next_pair_group_map = {}
        self.next_pair_sitouts = set()
        # this round's communication structure
        self.groups = []
        self.group_map = {}
        self.talk_map = {}        # listener -> the agents it hears from
        self.redraw_network()
        if self.lookahead:
            self.draw_next_pair_groups()
        # cooperation rate per round, for the whole episode
        self.coop_hist = []
        self.prev_other_agent_rep = {
            a: dict(self.agents[a].other_agent_rep) for a in self.agents
        }

    # ------------------------------------------------------------------
    # spaces
    # ------------------------------------------------------------------
    @property
    def num_agents(self):
        return self._num_agents

    @property
    def action_space(self):
        """0 = COOPERATE, 1 = DEFECT."""
        return DiscreteWithDType(2, dtype=np.uint8)

    @property
    def observation_space(self):
        """The agent observes the reputations it privately holds about others.

        curr_obs[k] is this agent's opinion of the k-th other agent (ids
        sorted, itself excluded). Opinions are EMAs of +-1 signals, so they are
        bounded by [-1, 1]. Entries for agents it has never heard about -- never
        grouped with, and nothing relayed yet -- stay at 0.

        With use_reputation=False, curr_obs holds the coming opponents' last
        actions instead (+1 cooperated, -1 defected, 0 did not play), which
        fits the same [-1, 1] box, and there is no own_rep entry.
        """
        table = self.lookahead
        width = (self.pair_size - 1) if table else (self.num_agents - 1)
        spaces = {
            "curr_obs": Box(low=-1.0, high=1.0, shape=(width,), dtype=np.float32)
        }
        if table and self.table_mask:
            spaces["table_mask"] = Box(
                low=0.0, high=1.0, shape=(width,), dtype=np.float32
            )
        if self.observe_own_rep and not self.observe_actions:
            spaces["own_rep"] = Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        if self.partial_participation:
            spaces["is_playing"] = Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        obs_space = Dict(spaces)
        print("obs_space", obs_space)
        return obs_space

    def setup_agents(self):
        """Construct all the agents for the environment"""
        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            self.agents[agent_id] = PDAgent(
                agent_id,
                num_agents=self.num_agents,
                max_len=self.rep_memory,
                network_neighbor=self.network_neighbor_map[agent_id],
            )

    # ------------------------------------------------------------------
    # payoffs
    # ------------------------------------------------------------------
    def payoff(self, action, num_cooperators, num_players=None):
        """One agent's payoff for this round, given how the table played.

        `num_cooperators` counts the cooperators among the `num_players` agents
        that were actually drawn -- including this agent. Everyone at the table
        moves simultaneously; there is no pairing and no sequencing in either
        mode, only a different shape for the same one-shot group game.

        The paper's all-or-nothing scheme: "everyone cooperated" and "everyone
        defected" are judged over the agents that were actually drawn.
        """
        n = self.num_agents if num_players is None else num_players
        if num_cooperators == n:
            return self.all_coop_payoff
        if num_cooperators == 0:
            return self.all_defect_payoff
        return (self.coop_mixed_payoff if action == COOPERATE
                else self.defect_mixed_payoff)

    def reward_reputation(self, agent_id):
        """The reputation figure that enters this agent's reward.

        The public reputation: the mean opinion the whole population holds
        about the agent, smoothed over rep_memory rounds. One scalar per
        agent, identical for everyone who looks at it -- which is what makes
        it a reputation rather than a private assessment.

        Kept as a method rather than inlined because it is the single point
        of truth for three call sites -- the reward, the "own_rep"
        observation and the logged info -- and they must not drift apart.
        """
        return self.agents[agent_id].return_reputation()

    def payoff_levels(self):
        """The four corners of this round's game, for printing and logging."""
        return {
            "all_coop": self.all_coop_payoff,
            "all_defect": self.all_defect_payoff,
            "lone_defector": self.defect_mixed_payoff,
            "lone_cooperator": self.coop_mixed_payoff,
        }

    # ------------------------------------------------------------------
    # participation
    # ------------------------------------------------------------------
    def sample_players(self):
        """Draw the agents that play this round, without replacement."""
        if not self.partial_participation:
            return set(self.agents)
        ids = sorted(self.agents)
        picked = self.rng.choice(len(ids), size=self.num_players, replace=False)
        return {ids[int(j)] for j in picked}

    def is_playing(self, agent_id):
        return agent_id in self.players_this_round

    # ------------------------------------------------------------------
    # communication groups
    # ------------------------------------------------------------------
    def sample_groups(self):
        """Randomly partition every agent into disjoint groups of group_size.

        Shuffle the population and cut it into consecutive blocks, so each
        agent is in exactly one group and the groups never overlap. Two agents
        are grouped together with probability (group_size - 1) / (n - 1) in any
        given round, which is the quantity that governs how fast first-hand
        information about a given agent reaches a given observer.

        When group_size does not divide num_agents the last block holds the
        remainder. A remainder of exactly one agent would leave that agent with
        nobody to talk to, so it is folded into the previous block instead --
        that block is then one larger than group_size for the round.
        """
        ids = self.agent_ids_sorted
        if self.group_size is None or self.group_size >= len(ids):
            return [set(ids)]
        order = self.rng.permutation(len(ids))
        shuffled = [ids[int(j)] for j in order]
        k = self.group_size
        groups = [set(shuffled[i:i + k]) for i in range(0, len(shuffled), k)]
        if len(groups) > 1 and len(groups[-1]) < 2:
            # pop first: `groups[-2] |= groups.pop()` would resolve the index
            # against the already-shortened list and clobber the wrong group
            leftover = groups.pop()
            groups[-1] |= leftover
        return groups

    # ------------------------------------------------------------------
    # pairwise tables
    # ------------------------------------------------------------------
    def draw_pair_groups(self):
        """Cut this round's participants into random tables of pair_size.

        Shuffle, then take consecutive blocks of k. The final block holds the
        remainder and plays as a smaller table. A remainder of exactly one
        agent has nobody to play against, so that agent sits the round out --
        it is returned in pair_sitouts and the caller drops it from
        players_this_round, which makes it a non-participant for every
        downstream purpose: no action recorded, no payoff, no observers, no
        reputation bonus.

        NB this is a different rule from sample_groups(), which folds a lone
        remainder INTO the previous group. There a singleton would have nobody
        to gossip with but the round still happens; here it would have nobody
        to play, which is a reason not to play at all.
        """
        ids = sorted(self.players_this_round)
        self.pair_groups, self.pair_group_map, self.pair_sitouts = [], {}, set()
        if len(ids) < 2:
            self.pair_sitouts = set(ids)
            return self.pair_groups
        order = self.rng.permutation(len(ids))
        shuffled = [ids[int(j)] for j in order]
        k = min(self.pair_size, len(shuffled))
        groups = [set(shuffled[i:i + k]) for i in range(0, len(shuffled), k)]
        if len(groups[-1]) < 2:
            self.pair_sitouts = groups.pop()
        self.pair_groups = groups
        self.pair_group_map = {a: g for g in groups for a in g}
        return self.pair_groups

    def draw_next_pair_groups(self):
        """Cut the tables for the round that is about to be played.

        Called at the END of step(), after self.playing has been resampled, so
        obs_scope="next_table" can show an agent who it will face. It writes
        only the next_* fields: this round's pair_groups still has to stand,
        because own_rep is the opinion of the audience that just watched this
        agent act, not of the opponents it has yet to meet.
        """
        keep = (self.pair_groups, self.pair_group_map, self.pair_sitouts)
        self.players_this_round, saved = set(self.playing), self.players_this_round
        self.draw_pair_groups()
        (self.next_pair_groups, self.next_pair_group_map,
         self.next_pair_sitouts) = (self.pair_groups, self.pair_group_map,
                                    self.pair_sitouts)
        (self.pair_groups, self.pair_group_map, self.pair_sitouts) = keep
        self.players_this_round = saved
        return self.next_pair_groups

    def adopt_next_pair_groups(self):
        """Promote the pre-drawn tables to this round's."""
        (self.pair_groups, self.pair_group_map, self.pair_sitouts) = (
            self.next_pair_groups, self.next_pair_group_map,
            self.next_pair_sitouts)
        return self.pair_groups

    def next_opponents_of(self, agent_id):
        """Who this agent is about to play, under obs_scope='next_table'."""
        return self.next_pair_group_map.get(agent_id, set()) - {agent_id}

    def pair_group_of(self, agent_id):
        """The table agent_id played at this round, itself included."""
        return self.pair_group_map.get(agent_id, set())

    def opponents_of(self, agent_id):
        """The agents agent_id actually played a two-player game against."""
        return self.pair_group_of(agent_id) - {agent_id}

    def pair_matrix(self, own_action, other_action):
        """One two-player encounter, scored with the ordinary 2x2 matrix."""
        if own_action == COOPERATE:
            return (self.all_coop_payoff if other_action == COOPERATE
                    else self.coop_mixed_payoff)          # R or S
        return (self.defect_mixed_payoff if other_action == COOPERATE
                else self.all_defect_payoff)              # T or P

    def pairwise_payoff(self, agent_id):
        """Aggregate this agent's round robin at its table.

        The agent committed to one action and plays it against every opponent,
        so this is sum_j M[a_i][a_j] -- divided by the number of opponents
        when pair_payoff="mean", which puts a small leftover table on the same
        scale as a full one.
        """
        opponents = self.opponents_of(agent_id)
        if not opponents:
            return 0.0
        own = self.actions[agent_id]
        total = sum(self.pair_matrix(own, self.actions[o]) for o in opponents)
        if self.pair_payoff == "mean":
            return total / len(opponents)
        return float(total)

    def draw_groups(self):
        """Redraw this round's groups and the agent -> group lookup."""
        if self.play_mode == "pairwise":
            # the table you played at is also the group you talk to; a
            # separate communication partition would be a second, invisible
            # structure nobody asked for
            self.groups = [set(g) for g in self.pair_groups]
            if self.pair_sitouts:
                self.groups = self.groups + [set(self.pair_sitouts)]
        elif self.network_mode == "groups":
            self.groups = self.sample_groups()
        else:
            self.groups = [set(self.agents)]
        self.group_map = {a: g for g in self.groups for a in g}
        return self.groups

    def sample_directed(self, size):
        """For every agent, draw `size` distinct others uniformly at random.

        One argsort of an (n, n) matrix of uniforms with the diagonal pushed to
        the end, so an agent is never drawn by itself and each row is an
        independent uniform sample without replacement.
        """
        ids = self.agent_ids_sorted
        n = len(ids)
        if size <= 0:
            return {a: set() for a in ids}
        if size >= n - 1:
            return {a: {b for b in ids if b != a} for a in ids}
        r = self.rng.random((n, n))
        np.fill_diagonal(r, np.inf)          # never pick yourself
        order = np.argsort(r, axis=1)[:, :size]
        return {ids[i]: {ids[int(c)] for c in order[i]} for i in range(n)}

    def draw_sample_graph(self):
        """Redraw this round's gossip channel.

        talk_map[i] : the comm_j agents whose opinion tables i averages with.

        Only gossip is sampled. Who saw i act is decided by the round itself --
        i's table under play_mode="pairwise", the agents that played under
        play_mode="group" -- so j varies the gossip reach with the observation
        window held fixed, which network_mode="groups" cannot do because there
        one partition sets both.
        """
        self.talk_map = self.sample_directed(self.comm_j)
        return self.talk_map

    def redraw_network(self, use_next_tables=False):
        """Redraw whichever per-round structure this network_mode uses."""
        if self.play_mode == "pairwise":
            if use_next_tables and self.next_pair_groups:
                # already cut at the end of the last step and shown to the
                # agents; redrawing here would make the observation a lie
                self.adopt_next_pair_groups()
            else:
                # tables first: they define who plays, and therefore who is left
                self.draw_pair_groups()
        if self.network_mode == "sample":
            self.draw_sample_graph()
        self.draw_groups()

    def group_of(self, agent_id):
        """The set of agents sharing this round's group with agent_id."""
        return self.group_map.get(agent_id, set(self.agents))

    def group_index(self, agent_id):
        """Which group agent_id is in this round, for logging."""
        for i, g in enumerate(self.groups):
            if agent_id in g:
                return i
        return -1

    # ------------------------------------------------------------------
    # network / reputation
    # ------------------------------------------------------------------
    def current_neighbor_map(self):
        """Who talks to whom this round.

        "static":  the pre-computed graph, identical every round.
        "groups":  this round's random partition -- an agent's neighbours are
                   its groupmates, and nobody else.
        "table":   pairwise play only: your neighbours are the opponents you
                   just played, so the gossip channel and the observation
                   window are the same set of agents.
        "players": the complete graph over the agents drawn to play.
        """
        if self.network_mode == "sample":
            return dict(self.talk_map)
        if self.network_mode in ("groups", "table"):
            return {agent_id: self.group_of(agent_id) - {agent_id}
                    for agent_id in self.agents}
        if self.network_mode == "players":
            players = self.players_this_round
            return {
                agent_id: (players - {agent_id}) if agent_id in players else set()
                for agent_id in self.agents
            }
        return self.network_neighbor_map

    def neighbors_of(self, agent_id):
        """The network neighbours agent_id broadcasts its opinions to."""
        return self.current_neighbor_map()[agent_id]

    def observers_of(self, agent_id):
        """Which agents see agent_id's action this round.

        An agent that sat out did not act, so there is nothing to judge it on
        and it has no observers.
        """
        if not self.is_playing(agent_id):
            return set()
        if self.play_mode == "pairwise":
            # you are judged by exactly the opponents who faced your action.
            # This overrides network_mode -- nobody outside the table played
            # you, so nothing about the gossip graph can widen the window.
            watchers = self.opponents_of(agent_id)
            if not self.observe_neighbors_only:
                watchers = {o for o in self.agents if o != agent_id}
                if not self.nonplayers_observe:
                    watchers &= self.players_this_round
            return watchers
        if self.network_mode == "sample":
            # "sample" is a gossip-only mode: it says nothing about who saw
            # what, so in the group game everyone who played saw everyone
            # else, exactly as under network_mode="players"
            watchers = {o for o in self.agents if o != agent_id}
            if not self.nonplayers_observe:
                watchers &= self.players_this_round
            return watchers
        if self.network_mode == "groups":
            # the group bounds what you can see as well as who you gossip with
            watchers = self.group_of(agent_id) - {agent_id}
            if not self.observe_neighbors_only:
                watchers = {o for o in self.agents if o != agent_id}
        elif self.network_mode == "players":
            return self.players_this_round - {agent_id}
        elif self.observe_neighbors_only:
            watchers = set(self.network_neighbor_map[agent_id])
        else:
            watchers = {other for other in self.agents if other != agent_id}
        if not self.nonplayers_observe:
            watchers &= self.players_this_round
        return watchers

    def is_defecting(self, agent_id):
        return self.actions[agent_id] == DEFECT

    def opinion_signal(self, observer_id, target_id):
        """The +1 / 0 / -1 an observer emits about a target this round.

        "image" (default) -- image scoring, unconditional. A cooperating
        target is praised and a defecting target is condemned by every
        observer that saw it. Neither the observer's own action nor the
        population's standing enters the judgement.

        "clean_hands" -- only an observer who cooperated this round has
        standing to condemn. A cooperating target still earns +1 from
        everyone; a defecting target is condemned by cooperators but draws a
        neutral 0 from a fellow defector, who is in no position to judge.
        This is the more defensible model of human judgement, and it is the
        rule in the indirect-reciprocity tradition where condemnation must
        itself be justified.

        The cost is a positive feedback: as the population defects, fewer
        observers have clean hands, so condemnation dries up exactly when it
        is most needed.

        A 0 is not a judgement: run_rep does not apply it, so a
        mutual-defection encounter leaves both tables untouched.
        """
        if not self.is_defecting(target_id):
            return 1
        if self.opinion_rule == "clean_hands" and self.is_defecting(observer_id):
            return 0
        return -1

    def run_rep(self):
        """agent forms an opinion -> average it into the true reputation ->
        broadcast the opinions along the network and update them again.
 
        Who is in the conversation depends on network_mode. With "static" it
        is the fixed adjacency graph every round. With "groups" it is this
        round's random partition: an agent forms first-hand opinions only about
        its own groupmates and averages its table only with theirs, so an
        opinion about an agent outside the group can only have arrived
        second-hand, one group-hop per round.
        """
        reps = {}
        rep_change = set()
        # the static graph, or this round's redrawn one
        agent_neighbor = self.current_neighbor_map()
        if self.network_mode != "static":
            # each agent's own record of who it talks to is refreshed here
            for _aid, _agent in self.agents.items():
                _agent.network_neighbor = agent_neighbor[_aid]
        for agent_id in self.agents:
            for other_agent_id in self.agents:
                if agent_id == other_agent_id:
                    continue
                if agent_id in self.observers_of(other_agent_id):
                    total_rep = self.opinion_signal(agent_id, other_agent_id)
                    if total_rep != 0:
                        rep_change.add(other_agent_id)
                        self.agents[agent_id].adjust_other_agent_rep(
                            other_agent_id, total_rep
                        )

        # snapshot the tables after first-hand observation and before the
        # broadcast, so the public reputation is collapsed from the same
        # state in every mode
        for agent_id in self.agents:
            for other_agent_id in self.agents:
                if other_agent_id not in reps:
                    reps[other_agent_id] = 0
                if agent_id == other_agent_id:
                    continue
                reps[other_agent_id] += self.agents[agent_id].other_agent_rep[
                    other_agent_id
                ]
 
        # collapse the opinions held about each agent into its public
        # reputation: the mean over every other agent's table, including
        # agents that never observed this one and hold a stale or zero view
        for id in rep_change:
            avg = reps[id] / (self.num_agents - 1)
            self.agents[id].add_rep(avg)
            self.agents[id].reputation = self.agents[id].calculate_rep()
 
        if self.gossip_mode == "delta":
            # Pass on the CHANGE, not the level. cur - prev is exactly what a
            # neighbour learned first-hand since the last broadcast, so adding
            # the mean of those changes to your own level carries the news
            # without overwriting what you saw yourself.
            #
            # This matters because averaging levels is row-stochastic: it
            # contracts the spread of opinion toward consensus and preserves
            # the population mean, so raising comm_j destroys the disagreement
            # the private reward reads and moves no aggregate in exchange. A
            # sum of changes is neither row-stochastic nor mean-preserving,
            # so the gossip degree can actually do something.
            prev = self.prev_other_agent_rep
            updated = {}
            for id in self.agents:
                cur = self.agents[id].other_agent_rep
                nbrs = [m for m in agent_neighbor[id] if m != id]
                new = dict(cur)
                if nbrs:
                    for tgt in cur:
                        if tgt == id:
                            new[tgt] = 0.0
                            continue
                        step = np.mean([
                            self.agents[m].other_agent_rep[tgt] - prev[m][tgt]
                            for m in nbrs
                        ])
                        new[tgt] = float(np.clip(cur[tgt] + step, -1.0, 1.0))
                updated[id] = new
            for id in self.agents:
                self.agents[id].other_agent_rep = updated[id]
            self.prev_other_agent_rep = {
                a: dict(self.agents[a].other_agent_rep) for a in self.agents
            }
            return
 
        # broadcast: average your table with your neighbours'
        for id in self.agents:
            neighbor_avg = dict(self.agents[id].other_agent_rep)
            neighbor_count = 1
            self.agents[id].tmp_other_agent_rep = neighbor_avg
            for agent_id in self.agents:
                if agent_id in agent_neighbor[id]:
                    neighbor_count += 1
                    neighbor_avg = {
                        i: neighbor_avg[i] + self.agents[agent_id].other_agent_rep[i]
                        for i in neighbor_avg.keys()
                    }
            if neighbor_count > 1:
                for i in neighbor_avg.keys():
                    if i == id:
                        neighbor_avg[i] = 0
                    elif i in agent_neighbor[id]:
                        # i's own table holds 0 about itself, so only
                        # neighbor_count - 1 tables carry an opinion about i
                        neighbor_avg[i] = neighbor_avg[i] / (neighbor_count - 1)
                    else:
                        neighbor_avg[i] = neighbor_avg[i] / neighbor_count
                self.agents[id].tmp_other_agent_rep = neighbor_avg
        for id in self.agents:
            self.agents[id].other_agent_rep = self.agents[id].tmp_other_agent_rep
        self.prev_other_agent_rep = {
            a: dict(self.agents[a].other_agent_rep) for a in self.agents
        }
        
    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def last_action_signal(self, agent_id):
        """An agent's action in the round just played, on the opinion scale.

        +1 cooperated, -1 defected -- the same signs image scoring gives --
        and 0 when there is nothing to report: the agent sat the round out,
        or no round has been played yet this episode.
        """
        action = self.last_actions.get(agent_id)
        if action is None:
            return 0.0
        return 1.0 if action == COOPERATE else -1.0

    def action_obs(self, agent):
        """The no-reputation observation: what the coming opponents just did.

        Under play_mode="pairwise" these are the agents this one is about to
        share a table with, sorted ascending (defectors first) and padded to
        pair_size - 1, the same layout as the obs_scope="next_table"
        reputations. The actions are public: the agent is shown them whether
        or not it shared a table with those opponents last round.

        Under play_mode="group" every other agent is an opponent, so it is
        all of them in id order.
        """
        if self.lookahead:
            coming = self.next_opponents_of(agent.agent_id)
            width = self.pair_size - 1
            vals = sorted(self.last_action_signal(o) for o in coming)[:width]
            pad = width - len(vals)
            obs = {"curr_obs": np.array(vals + [0.0] * pad, dtype=np.float32)}
            if self.table_mask:
                obs["table_mask"] = np.array(
                    [1.0] * len(vals) + [0.0] * pad, dtype=np.float32
                )
            return obs
        others = [o for o in self.agent_ids_sorted if o != agent.agent_id]
        return {"curr_obs": np.array(
            [self.last_action_signal(o) for o in others], dtype=np.float32
        )}

    def get_obs(self, agent):
        """What the agent sees: its opinions of others, and its own standing.

        "own_rep" is the public reputation the reward is computed from -- the
        state an agent needs to attribute the reputation term to its own
        behaviour. Without it the bonus is an unexplained drift in the reward.

        With use_reputation=False there is no reputation to show, so the
        agent sees its coming opponents' last actions (see action_obs).
        """
        if self.observe_actions:
            obs = self.action_obs(agent)
        elif self.obs_scope == "next_table":
            # Just the opponents, sorted worst-first. Sorting makes the vector
            # invariant to WHICH agents are at the table, which is the right
            # inductive bias here: tables are redrawn at random, agents are
            # interchangeable, and an agent's reputation already summarises
            # its history, so slot identity carries no signal -- only the
            # distribution of standings does. Slot 0 is then always "my worst
            # opponent", a stable feature instead of a rotating one.
            coming = self.next_opponents_of(agent.agent_id)
            width = self.pair_size - 1
            vals = sorted(agent.other_agent_rep[o] for o in coming)[:width]
            pad = width - len(vals)
            obs = {"curr_obs": np.array(vals + [0.0] * pad, dtype=np.float32)}
            if self.table_mask:
                obs["table_mask"] = np.array(
                    [1.0] * len(vals) + [0.0] * pad, dtype=np.float32
                )
        else:
            others = [o for o in self.agent_ids_sorted if o != agent.agent_id]
            obs = {"curr_obs": np.array(
                [agent.other_agent_rep[o] for o in others], dtype=np.float32
            )}
        if self.observe_own_rep and not self.observe_actions:
            obs["own_rep"] = np.array(
                [self.reward_reputation(agent.agent_id)], dtype=np.float32
            )
        if self.partial_participation:
            obs["is_playing"] = np.array(
                [1.0 if agent.agent_id in self.playing else 0.0], dtype=np.float32
            )
        return obs

    def step(self, actions):
        """Play one round of the matrix game.

        Parameters
        ----------
        actions: dict {agent-id: 0 (cooperate) or 1 (defect)}
            floats are rounded, so a 0.0/1.0 policy head works unchanged.

        Returns
        -------
        observations, rewards, dones, truncated, infos
        """
        self.count += 1
        self.players_this_round = set(self.playing)
        # redraw the communication structure: it governs who sees this
        # round's actions and whose opinion tables are averaged afterwards.
        # In pairwise mode it also cuts the tables, which can strand a lone
        # agent with no opponent -- that agent is demoted to a sit-out here,
        # before anything downstream reads players_this_round.
        self.redraw_network(use_next_tables=self.lookahead)
        if self.play_mode == "pairwise" and self.pair_sitouts:
            self.players_this_round -= self.pair_sitouts
        num_players = len(self.players_this_round)

        # 1. the drawn agents act simultaneously; the rest sit out -- their
        #    action is ignored, not recorded as a cooperation or a defection
        for agent_id, agent in self.agents.items():
            if agent_id in self.players_this_round:
                action = actions[agent_id] if agent_id in actions else COOPERATE
                self.actions[agent_id] = agent.act(action)
            else:
                self.actions[agent_id] = agent.sit_out()
        # what the next observation reports; an agent that sat out has no
        # action this round
        self.last_actions = {
            agent_id: (self.actions[agent_id]
                       if agent_id in self.players_this_round else None)
            for agent_id in self.agents
        }

        # 2. hand out the matrix-game payoff to the players only
        num_cooperators = sum(
            1 for a in self.players_this_round if self.actions[a] == COOPERATE
        )
        for agent_id, agent in self.agents.items():
            if agent_id in self.players_this_round:
                if self.play_mode == "pairwise":
                    agent.add_reward(self.pairwise_payoff(agent_id))
                else:
                    agent.add_reward(self.payoff(
                        self.actions[agent_id], num_cooperators, num_players
                    ))
            # an agent sitting out receives exactly 0

        self.coop_hist.append(num_cooperators / max(1, num_players))

        # 3. update reputation (runs in the baseline too, it just does not
        #    reach the reward unless use_reputation is on)
        self.run_rep()

        # 4. collect rewards
        rewards = {}
        reward_so_far = {}
        for agent_id, agent in self.agents.items():
            reward, cumulative = agent.compute_reward()
            rewards[agent_id] = reward
            reward_so_far[agent_id] = cumulative

        if self.use_reputation:
            for agent_id, agent in self.agents.items():
                # an agent that sat out earns nothing this round, reputation
                # bonus included -- it was not judged, so it is not paid either
                if agent_id in self.players_this_round:
                    rewards[agent_id] = (rewards[agent_id] + self.rep_weight
                                         * self.reward_reputation(agent_id))

        for agent_id in rewards:
            if agent_id not in self.players_this_round:
                rewards[agent_id] = 0.0

        for agent_id, agent in self.agents.items():
            agent.last_reward = rewards[agent_id]

        # draw the NEXT round's players before building observations
        self.playing = self.sample_players()
        if self.lookahead:
            # cut the coming round's tables now, so get_obs can name the
            # opponents. This does NOT disturb pair_groups, which own_rep
            # still needs: an agent is paid by the audience that just watched
            # it, and told about the opponents it is about to face.
            self.draw_next_pair_groups()

        # 5. observations, dones and infos
        observations, dones, infos = {}, {}, {}
        for agent_id, agent in self.agents.items():
            observations[agent_id] = self.get_obs(agent)
            dones[agent_id] = agent.get_done()
            played = agent_id in self.players_this_round
            infos[agent_id] = {
                "action": self.actions[agent_id],
                "cooperated": int(played and self.actions[agent_id] == COOPERATE),
                # NB: raw matrix-game payoff, before any shaping
                "reward_this_episode": reward_so_far[agent_id],
                "coop_rate": agent.coop_rate(),
                "num_cooperators": num_cooperators,
                "mutual_cooperation": int(num_cooperators == num_players),
                "played": int(played),
                "rounds_played": agent.coop_steps + agent.defect_steps,
                "num_players": num_players,
                "group": self.group_index(agent_id),
                "group_members": len(self.group_of(agent_id)),
            }
            if self.play_mode == "pairwise":
                # how many two-player games this agent actually played, and
                # how many of those opponents cooperated
                opponents = self.opponents_of(agent_id)
                infos[agent_id]["table_size"] = len(self.pair_group_of(agent_id))
                infos[agent_id]["encounters"] = len(opponents)
                infos[agent_id]["opponent_coops"] = sum(
                    1 for o in opponents if self.actions[o] == COOPERATE
                )
            if self.use_reputation:
                # the public scalar the reward is computed from
                infos[agent_id]["reputation"] = self.reward_reputation(agent_id)
                # how many agents actually saw this one act, for diagnostics;
                # it no longer enters the reward
                infos[agent_id]["audience"] = len(self.observers_of(agent_id))

        if self.debug:
            print("===== step {} =====".format(self.count))
            print("actions:", self.actions,
                  "| cooperators: {}/{}".format(num_cooperators, num_players))
            print("groups:", [sorted(g) for g in self.groups])
            print("rewards:", rewards)

        dones["__all__"] = np.any(list(dones.values()))
        truncated = self.episode_length is not None and self.count >= self.episode_length
        trunc = {agent_id: truncated for agent_id in self.agents}
        trunc["__all__"] = truncated

        return observations, rewards, dones, trunc, infos

    def reset(self, *args, **kwargs):
        """Reset between rollouts.

        The per-agent info is laid out in the same six slots the public goods
        logs use, so the existing parsers and plotting scripts work unchanged:

            0 episode return (raw, unshaped)
            1 total cooperations          (<- total_investment)
            2 cooperation rate            (<- mean_investment)
            3 defect steps                (<- free_ride_steps)
            4 mutual-cooperation steps    (<- full_contribution_steps)
            5 reputation
        """
        info = {}
        mutual = int(sum(1 for c in self.coop_hist if c == 1.0))
        for agent in self.agents.values():
            reward, reward_so_far = agent.compute_reward()
            info[agent.agent_id] = [
                reward_so_far,
                agent.coop_steps,
                agent.coop_rate(),
                agent.defect_steps,
                mutual,
                agent.return_reputation(),
            ]
        info["coop_hist"] = self.return_coop_hist()

        self.ep_count += 1
        self.count = 0
        self.agents = {}
        self.setup_agents()
        self.actions = {agent_id: COOPERATE for agent_id in self.agents}
        self.last_actions = {agent_id: None for agent_id in self.agents}
        self.coop_hist = []
        # setup_agents() built fresh agents, so the delta baseline has to be
        # rebuilt against them or the first broadcast of the episode would
        # difference this episode's tables against the last one's
        self.prev_other_agent_rep = {
            a: dict(self.agents[a].other_agent_rep) for a in self.agents
        }
        self.playing = self.sample_players()
        self.players_this_round = set(self.playing)
        self.redraw_network()
        if self.lookahead:
            self.draw_next_pair_groups()

        observations = {}
        for agent in self.agents.values():
            observations[agent.agent_id] = self.get_obs(agent)
        return observations, info

    def seed(self, seed=None):
        np.random.seed(seed)
        # the group and participation draws have their own generator, so a
        # seeded env replays exactly the same sequence of partitions
        self.rng = np.random.default_rng(seed)

    def close(self):
        return

    def render(self, mode="human"):
        """No map to draw, so print the round."""
        line = "round {}: ".format(self.count) + ", ".join(
            "{}={}".format(a, "C" if self.actions[a] == COOPERATE else "D")
            if a in self.players_this_round else "{}=--".format(a)
            for a in self.agent_ids_sorted
        )
        n_coop = sum(1 for a in self.players_this_round
                     if self.actions[a] == COOPERATE)
        line += " | cooperators {}/{}".format(n_coop, len(self.players_this_round))
        if self.network_mode in ("groups", "table"):
            line += " | {} ".format(
                "tables" if self.network_mode == "table" else "groups"
            ) + " ".join(
                "{" + ",".join(sorted(g)) + "}" for g in self.groups
            )
        elif self.network_mode == "sample":
            line += " | hear {}".format(self.comm_j)
        if mode == "human":
            print(line)
            return None
        return line

    def return_coop_hist(self):
        return self.coop_hist

    @staticmethod
    def get_environment_callbacks():
        return DefaultCallbacks