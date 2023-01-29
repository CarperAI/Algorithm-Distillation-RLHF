# Algorithm-Distillation-RLHF

A reinforcement learning algorithm is characterised by the trajectories it generates during training.

We are interested in "algorithm distillation" - whether trajectories can be modelled by transformers, as studied in the original deepmind algorithm distillation paper. 

A particular focus of this repo is to extend prior work to the case where:
1. the trajectories have been generated by the TRLx library during RLHF training of language models
2. the transformer modelling the trajectories is itself a standard language model


## On data formats

A trajectory is typically defined as a list of `(state, action, reward)` triples. For training purposes, it is sometimes useful to augment this to include `logprobs`, which is, for each triple `(s, a, r)`, the probability of taking action $a$ at state $s$ as determined the policy generating the trajectory.

We therefore define an **RL Format Trajectory** as a sequence of `(state, action, reward, logprobs)` tuples.

The typical way to learn to model these trajectories with a transformer is to seperately map the final hidden state using 3 different heads. That is, for a given triple `(s,a,r,l)` a transformer $f$ maps to $(\hat{s}, \hat{a}, \hat{r}, \hat{l})$. 

In this repo, this is done via the models in `/models/rl`. 

We are also interested in the ability of standard language models (with language modeling heads) to learn trajectories. To this end we define a **Language Format Trajectory** as a trajectory serialised into a string. There are many possible ways to do this, and the optimal one requires investigation. For example, for trajectories generated using TRLx when finetuning a language model on positive sentiment, we can format the trajectory as the string: 

```
prompt: Dan went down to the shops.
completion: He smiled as he walked - the sun was shining.
reward: 0.9975
###
```

It's less obvious how to do this when the task is not a language task, such as moonlander. Enumerating the states as coordinates might work, but requires experimentation. 

Trajectories in *Language format* are learnt by models in `/models/lm`.

## To summarise:

`/models` contains the "algorithm distillation models", transformers that are trained in a supervised fashion to learn RL trajectories. We distinguish between models that operate on *RL Format* trajectories and *Language format* trajectories.

`/tasks` contains code to produce the RL trajectories that the models learn. It can store this data however it likes, but each task should expose a `torch.utils.data.Dataset` that can return trajectory data in either *RL Format* or *Language format*. 

## Generating trajectory data
I am using my own fork of TRLx that has rollout logging.

## ToDo:

Today:
[X] Set up repo structure (just for your language stuff, @H can add in his)
[X] Add train script for models/lm/casuallm 
[ ] Clone H's stuff and merge with @H stuff (/models/rl) and (/tasks/rl) (25 mins)
[ ] Proper PR for TRLx (25 mins)
[ ] Post guide and project tasks on discord  

Future:
[ ] Add more elegant meta class switching between ...LanguageTrajectories and ...RlTrajectories
[ ] Add online evaluation script for models/lm/casuallm
[ ] Improve train script to include reward accuracy
[ ] Run some preliminary experiments  
[ ] Add __main__ file with click CLI interface for running experiments