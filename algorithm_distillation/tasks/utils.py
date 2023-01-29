import torch
import random

class ShuffledIterableDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, original_dataset: torch.utils.data.IterableDataset, buffer_size: int = 10_000):
        self.original_dataset = original_dataset
        self.buffer_size = buffer_size
        
    def __iter__(self):
        original_iterator = iter(self.original_dataset)
        
        # fill buffer (or until runs out)
        buffer = []
        while len(buffer) < self.buffer_size:
            try:
                x = next(original_iterator)
            except StopIteration:
                break
            buffer.append(x)
            
        # shuffle, yield and replace until original runs out
        for x in original_iterator:
            random.shuffle(buffer)
            yield buffer[-1]
            buffer[-1] = x
          
        # empty the remaining  
        for x in buffer:
            yield x
    