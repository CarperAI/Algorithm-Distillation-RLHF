
from tasks.lm.sentiment import SentimentLanguageTrajectories
from tasks.utils import ShuffledIterableDataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb

accelerator = Accelerator()

# Logging inits
wandb.init(project="algorithm-distillation")
logging_table = wandb.Table(columns=['step', 'generation'])

# Data
tokenizer = AutoTokenizer.from_pretrained('gpt2')
train_dataset = SentimentLanguageTrajectories(split="train", tokenizer=tokenizer)
train_dataset = ShuffledIterableDataset(train_dataset, buffer_size=10_000)
# eval_dataset = ...
# generate_dataset = ...
train_dataloader = DataLoader(train_dataset, shuffle=False)

# Setup parameters for training with accelerate
model = AutoModelForCausalLM.from_pretrained('gpt2')
optimizer = Adam(model.parameters(), lr=5e-5)
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

# Train
model.train()
total_steps = 0
for epoch in range(10):
    for batch in tqdm(train_dataloader, desc=f'Training epoch {epoch}'):
        
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        wandb.log({'loss': loss.item(), 'step': total_steps})
        accelerator.backward(loss)
        optimizer.step()
        total_steps +=1
        