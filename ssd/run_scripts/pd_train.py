"""Entry point for training independent DQN agents on the n-agent Prisoner's Dilemma.

Example
-------
    python run_pd_dqn.py --num-agents 5 --network ring --use-reputation \
        --rep-weight 2.0 --num-episodes 2000 --episode-len 200 --log-file pd_dqn.csv

Notes on the defaults
---------------------
* No action wrapper: PDEnv is already Discrete(2), so the Q head has exactly two
  outputs -- Q(cooperate) and Q(defect).
* Epsilon decays over half of the total environment steps. Defection is the
  dominant action from the first update, so a fast decay collapses every run to
  all-D before reputation can matter.
* --payoff-mode fraction replaces the paper's all-or-nothing scheme with a
  payoff affine in the group's cooperation rate. Every corner then pays the same
  at any group size (all-C 2, all-D 1, lone defector 3, lone cooperator 0 by
  default), defection still dominates by a constant margin, and a cooperator
  earns partial credit instead of 0 whenever one player defects. Use it when the
  comparison across --num-players should not also change the payoff matrix.
* --reward-scale defaults to 1/n under --payoff-mode threshold, because those
  payoffs grow with the agent count (a mixed-round defector earns n+1) and
  undiscounted returns would otherwise reach the thousands. Under
  --payoff-mode fraction it defaults to 1, since the payoffs are already O(1).
* Communication is network-only: agents see their neighbours' actions and
  reputation travels along edges. --network picks the topology. --network
  players is the special case where there is no fixed topology at all: the
  graph is redrawn every round as the complete graph over the k agents that
  were drawn to play, so agents talk to exactly the agents they just played.
* --num-players k draws k of the n agents each round. An agent sitting out
  neither acts nor earns nor is judged; its rounds are dropped from replay.
"""

import os
import sys
import argparse
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch

from social_dilemmas.envs.pd import PDEnv
from social_dilemmas.envs.pgg_utils import build_network
from DQN_model.pd_multi_agent import pd_mult_agent_dqn


def parse_args():
    p = argparse.ArgumentParser(description="DQN on the n-agent Prisoner's Dilemma")
    # environment
    p.add_argument("--num-agents", type=int, default=15)
    p.add_argument("--episode-len", type=int, default=1000)
    p.add_argument("--network", type=str, default="complete",
                   choices=["complete", "ring", "path", "star", "random",
                            "players"],
                   help="topology the agents observe and broadcast over; "
                        "'players' redraws it each round over the round's "
                        "k players instead of fixing it in advance")
    p.add_argument("--network-p", type=float, default=0.5)
    p.add_argument("--observe-all", action="store_true",
                   help="make actions public; the broadcast stays on the network")
    p.add_argument("--use-reputation", action="store_true")
    p.add_argument("--rep-weight", type=float, default=3.0,
                   help="weight of the reputation term in the reward; the bonus "
                        "swings over +-rep_weight as public reputation moves "
                        "over [-1, 1]. Only paid to agents that played.")
    p.add_argument("--num-players", type=int, default=None,
                   help="k: how many of the n agents are drawn to play each "
                        "round (default: all of them)")
    p.add_argument("--payoff-over", type=str, default="players",
                   choices=["players", "all"],
                   help="set the paper's n-dependent payoff levels from k "
                         "(the size of the table) or from n")
    p.add_argument("--no-nonplayers-observe", action="store_false",
                   help="agents sitting out form no first-hand opinions "
                        "(they still relay over the network); ignored with "
                        "--network players, where they are isolated anyway")
    p.add_argument("--no-mask-sitouts", action="store_true",
                   help="keep rounds the agent sat out in the replay buffer; "
                        "on by default they are dropped")
    # payoffs; leaving these unset uses the paper's n-dependent scheme
    p.add_argument("--all-coop-payoff", type=float, default=None)
    p.add_argument("--all-defect-payoff", type=float, default=1.0)
    p.add_argument("--defect-mixed-payoff", type=float, default=None)
    p.add_argument("--coop-mixed-payoff", type=float, default=0.0)
    # training
    p.add_argument("--num-episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--replay-capacity", type=int, default=5000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--update-freq", type=int, default=1)
    p.add_argument("--target-update-freq", type=int, default=1)
    p.add_argument("--fc-filters", type=int, nargs="+", default=[64, 64])
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-frac", type=float, default=0.5)
    p.add_argument("--eps-decay-style", type=str, default="LINEAR",
                   choices=["LINEAR", "EXPONENTIAL"])
    p.add_argument("--reward-scale", type=float, default=None,
                   help="default 1/num_agents; pass 1 for raw payoffs")
    # misc
    p.add_argument("--log-file", type=str, default="pd_dqn.csv")
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parse_args()
    use_rep = True
    playmode = "pairwise"
    pairsize = 5
    gossip_mode = "average"
    print("gossip mode is:", gossip_mode)
    obs_scope = "next_table" 
    print("obs_scope:", obs_scope)
    # "players" is not a fixed topology: the graph is redrawn every round from
    # whoever was drawn to play, so no adjacency matrix is passed at all
    network_mode = "sample"#"players" #if args.network == "players" else "static"
    adjacency = (None if network_mode == "players" or network_mode == "groups" or network_mode == "sample" else
                 build_network(args.network, args.num_agents,
                               p=args.network_p, seed=args.seed))
    if network_mode == "players":
        print("communication graph: redrawn each round over the round's players")
    comm_j = 0
    all_coop_payoff = 2
    all_defect_payoff = 1
    defect_mix = 3
    coop_mix = 0
    env = PDEnv(
        play_mode = playmode,
        pair_size=pairsize,
        num_agents=args.num_agents,
        episode_length=args.episode_len,
        all_coop_payoff=all_coop_payoff,
        all_defect_payoff=all_defect_payoff,
        defect_mixed_payoff=defect_mix,
        coop_mixed_payoff=coop_mix,
        use_reputation=use_rep,
        rep_weight=args.rep_weight,
        adjacency_matrix=adjacency,
        network_mode=network_mode,
        observe_neighbors_only=True,
        nonplayers_observe=not args.no_nonplayers_observe,
        comm_j = comm_j,
        gossip_mode = gossip_mode,   # <-- add this
        pair_payoff = "mean", 
        obs_scope = obs_scope,
        opinion_rule="clean_hands"
    )
    n = args.num_agents
    lv = env.payoff_levels()

    print("payoffs (table of {}): all-C {} each | all-D {} each | "
            "mixed C {} / D {}".format(
                env.payoff_n, lv["all_coop"], lv["all_defect"],
                lv["lone_cooperator"], lv["lone_defector"]))

    # threshold payoffs grow with the table (a mixed defector takes k + 1), so
    # they are scaled down by n; fraction payoffs are already O(1) and scaling
    # them would only shrink the learning signal
    reward_scale = 0.2

    print("reward scale:", round(reward_scale, 4))

    # epsilon is annealed in steps the agent actually PLAYS, not wall-clock
    # rounds: with k of n drawn each round an agent only acts on k/n of them,
    # and decaying over all of them would exhaust exploration k/n of the way
    # through its real experience
    played_frac = env.num_players / float(n)
    total_steps = int(args.num_episodes * args.episode_len * played_frac)
    epsilon_decay = (
        args.eps_start, args.eps_end, max(1, int(args.eps_decay_frac * total_steps))
    )
    trainer = pd_mult_agent_dqn(
        device=device,
        env=env,
        epsilon_decay=epsilon_decay,
        conv_filters=None,
        fc_filters=args.fc_filters,
        lr=args.lr,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        num_episode=args.num_episodes,
        tau=args.tau,
        episode_len=args.episode_len,
        log_file=args.log_file,
        update_freq=args.update_freq,
        target_update_freq=args.target_update_freq,
        use_cnn=False,
        debug=args.debug,
        epsilon_decay_style=args.eps_decay_style,
        reward_scale=reward_scale,
        mask_sitouts=not args.no_mask_sitouts,
    )

    trainer.train()

    if args.save_path is not None:
        trainer.save(args.save_path)
        print("saved policies to", args.save_path)

    if args.eval_episodes > 0:
        coop_rate, mean_return = trainer.evaluate(num_episodes=args.eval_episodes)
        print("=== greedy evaluation over {} episodes ===".format(args.eval_episodes))
        for id in sorted(coop_rate):
            print("{}: cooperation rate {:.3f}, mean episode return {:.1f}".format(
                id, coop_rate[id], mean_return[id]))
        print("group cooperation rate: {:.3f} (0 = all defect, 1 = social optimum)".format(
            float(np.mean(list(coop_rate.values())))))
        print("Q(cooperate), Q(defect) at the final state:")
        for id, q in trainer.q_values().items():
            print("  {}: C {:+.3f}  D {:+.3f}".format(id, q[0], q[1]))


if __name__ == "__main__":
    main()