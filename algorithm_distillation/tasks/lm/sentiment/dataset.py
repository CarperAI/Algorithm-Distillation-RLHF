import torch
from pathlib import Path
import json
from transformers import AutoTokenizer
from typing import Dict, Any, Union

Dataset = Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]

class SentimentTrajectories(Dataset):
    def __init__(self, format:str, *args, **kwargs):
        if format == "language":
            self = SentimentAsLanguageTrajectories(*args, **kwargs)
        elif format == "rl":
            raise NotImplementedError()
            # self = SentimentAsRlTrajectories(*stargs, **kwargs)
        else:
            raise RuntimeError(f"format must be either 'language' or 'rl', got: {format}")


class SentimentAsLanguageTrajectories(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer: AutoTokenizer, rollouts_folder_fpath: str, for_generation: bool = False, verbose: bool = True):
        self.tokenizer = tokenizer 
        self.rollouts_folder = Path(rollouts_folder_fpath)
        self.verbose = verbose
        self.for_generation = for_generation
        
    def format_rollout(self, d: Dict[Any, Any]) -> str:
        return f"Prompt: {d['query_text']}\nCompletion: {d['response_text']}\nReward: {d['rewards'][-1]}\n\n"
    
    def tokenize_for_training(self, x: str):
        inputs = self.tokenizer(x, truncation=True, return_tensors='pt')
        return {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': inputs.input_ids
        }
    
    def __iter__(self):
        
        runs = [run for run in self.rollouts_folder.iterdir() if run.name.startswith('run-e')]
        if self.verbose:    
            print(f'Iterating over {len(runs)} runs...')
        
        for run in runs:
            
            config = json.loads(open(run / 'config.json', 'r').read())
            epochs = [epoch for epoch in run.iterdir() if epoch.name != 'config.json']
            
            if self.verbose:
                print(f'...and {run.name} has {len(epochs)} epochs.')
                
            epochs = sorted(epochs)
            for epoch in epochs:
                # print(f'Yielding from epoch {epoch.name}')
                
                rollouts = json.loads(open(epoch, 'r').read())
                
                rollout_idx = 0
                prompt = ""
                while rollout_idx < len(rollouts):
                    
                    if self.for_generation:
                        d = rollouts[rollout_idx]
                        rollout_idx += 1
                        generation_prompt = f"Prompt: {d['query_text']}\nCompletion:"
                        yield self.tokenize_for_training(generation_prompt)
                        
                    else:
                        rollout = self.format_rollout(rollouts[rollout_idx])
                        rollout_idx += 1
                        
                        new_prompt = prompt + rollout
                        new_ids = self.tokenizer.tokenize(new_prompt)
                        
                        if len(new_ids) > self.tokenizer.model_max_length: # self.tokenizer.model_max_length:
                            yield self.tokenize_for_training(prompt)
                            prompt = ""
                        else:
                            prompt = new_prompt
                    
                if len(prompt) > 0:      
                    yield self.tokenize_for_training(prompt)
                
    
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = SentimentTrajectories("language", tokenizer, './decoded_rollouts', for_generation=False)
    for ex in dataset:
        print(tokenizer.decode(ex['input_ids'][0]))
        print('\n---------\n')