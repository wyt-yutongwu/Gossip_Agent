# [Gossip Agents: Reputation through Opinion Dynamics over Evolving Networks](https://arxiv.org/abs/2610.22674)

This repository contains the code for the paper **Gossip Agents: Reputation through Opinion Dynamics over Evolving Networks**, multi-agent reinforcement learning agents that build and use reputation through opinion dynamics over evolving networks.

## Environments

The repository includes training scripts for two environments:

| Environment | Entry point | Description |
| --- | --- | --- |
| Harvest | `ssd/run_scripts/harvest_train.py` | A sequential social dilemma in which agents share a common resource, trained with DQN. |
| Prisoner's Dilemma | `ssd/run_scripts/pd_train.py` | A multi-agent prisoner's dilemma. |

## Usage

### Harvest

```bash
python ssd/run_scripts/harvest_train.py --env harvest --num-agents 10 --use-reputation True --log-file <path to log file>
```

### Prisoner's Dilemma

```bash
python ssd/run_scripts/pd.train --num-agents 15 --use-reputation True --log-file <path to log file>
```

## Arguments

| Argument | Description | Example |
| --- | --- | --- |
| `--env` | Environment to train on (Harvest script only). | `harvest` |
| `--num-agents` | Number of agents in the environment. | `10` |
| `--use-reputation` | Whether agents use the reputation mechanism. Set to `False` to train without it. | `True` |
| `--log-file` | Path to the file where training logs are written. | `logs/harvest.log` |
| `--episode-len` | Number of steps in each episode. | `1000` |
| `--total-episodes` | Total number of episodes. | `10000` |

