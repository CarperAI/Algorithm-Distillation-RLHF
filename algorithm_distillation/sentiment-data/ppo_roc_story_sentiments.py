from posixpath import dirname
from datasets import load_dataset
from transformers import pipeline
import os
import yaml

import trlx
import torch
from typing import List 
from trlx.data.configs import TRLConfig

from trlx.utils.loading import get_model, get_orchestrator, get_pipeline

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

default_config = yaml.safe_load(open(os.path.join(dirname(__file__), "ppo_config.yml")))

def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    
    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1
        

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str]) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    stories = load_dataset("adamlin/roc_story")
    prompts = [d['sentence1'] for d in stories['train']]
    eval_prompts = [d for d in stories['validation'][:64]['sentence1']]
    
    model = trlx.train(
        model_path="gpt2",
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
    )
    

if __name__ == "__main__":
    main()