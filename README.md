# Algorithm-Distillation-RLHF

The current state of this repo is a preliminary version of a replication of the Algorithmic Distillation algorithm 
described in the paper [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/abs/2210.14215).
We also aim to make the codes general enough to try ideas beyond the paper.

# Quick start
A demo script/notebook is not provided yet, but the unit test `tests/test_ad.py` provides a complete routine of applying 
the transformer to the histories of toy tasks "FrozenLake-v1", "CartPole-v1". Please [take a look](tests/test_ad.py) 
and feel free to plug in your own gym env.
