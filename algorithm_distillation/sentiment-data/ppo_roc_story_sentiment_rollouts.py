import torch
from pathlib import Path
import json
from transformers import AutoTokenizer
from typing import Dict, Any

class RolloutsAsLanguageModellingTask(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer: AutoTokenizer, rollouts_folder_fpath: str):
        self.tokenizer = tokenizer 
        self.rollouts_folder = Path(rollouts_folder_fpath)
        
    def format_rollout(self, d: Dict[Any, Any]) -> str:
        return f"Prompt:{d['query_text']}\nCompletion:{d['response_text']}\nReward:{d['rewards'][-1]}\n\n"
    
    def __iter__(self):
        
        runs = [run for run in self.rollouts_folder.iterdir() if run.name.startswith('run-e')]
        print(f'Found {len(runs)} runs...')
        for run in runs:
            
            config = json.loads(open(run / 'config.json', 'r').read())
            epochs = [epoch for epoch in run.iterdir() if epoch.name != 'config.json']
            print(f'Run {run.name} has {len(epochs)} epochs...')
            epochs = sorted(epochs)
            for epoch in epochs:
                # print(f'Yielding from epoch {epoch.name}')
                
                rollouts = json.loads(open(epoch, 'r').read())
                
                rollout_idx = 0
                prompt = ""
                while rollout_idx < len(rollouts):
                    rollout = self.format_rollout(rollouts[rollout_idx])
                    rollout_idx += 1
                    
                    new_prompt = prompt + rollout
                    new_ids = self.tokenizer.tokenize(new_prompt)
                    
                    if len(new_ids) > tokenizer.model_max_length: # self.tokenizer.model_max_length:
                        yield self.tokenizer(prompt, return_tensors='pt')
                        prompt = ""
                    else:
                        prompt = new_prompt
                    
                        
                yield self.tokenizer(prompt, return_tensors='pt')
                
        raise StopIteration
    
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = RolloutsAsLanguageModellingTask(tokenizer, './decoded_rollouts')
    for ex in dataset:
        print(tokenizer.decode(ex['input_ids'][0]))
        print('\n---------\n')