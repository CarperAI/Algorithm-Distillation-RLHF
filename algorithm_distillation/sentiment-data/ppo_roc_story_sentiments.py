from posixpath import dirname
from datasets import load_dataset
from transformers import pipeline
import os
import yaml
from functools import partial

import trlx
import torch
from typing import List 
from trlx.data.configs import TRLConfig

from trlx.utils.loading import get_model, get_orchestrator, get_pipeline

def get_score_for_label(label, scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    label_to_score = {d['label'] : d['score'] for d in scores}
    return label_to_score[label]

default_config = yaml.safe_load(open(os.path.join(dirname(__file__), "ppo_config.yml")))

def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    
    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1
        

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "bhadresh-savani/distilbert-base-uncased-emotion",
        truncation=True,
        batch_size=256,
        device=device,
        return_all_scores=True,
    )

    def reward_fn(samples: List[str]) -> List[float]:
        output_batch = sentiment_fn(samples)
        sentiments = list(map(partial(get_score_for_label, 'joy'), output_batch))
        return sentiments

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