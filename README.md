# Algorithm-Distillation-RLHF

The current state of this repo is a preliminary version of a replication of the Algorithmic Distillation algorithm 
described in the paper [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/abs/2210.14215).
We also aim to try ideas beyond.

# Quick start
A test script is not provided yet, but the unit test `tests/test_ad.py` provides a complete routine of applying the
AD transformer to the histories of a single toy task "FrozenLake-v1". Please [take a look](tests/test_ad.py) and feel
free to plug in your own gym env.
