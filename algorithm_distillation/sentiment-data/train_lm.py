import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from rollouts_as_lm_task import RolloutsAsLanguageModellingTask
from tqdm.auto import tqdm

accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained('gpt2')
optimizer = torch.optim.Adam(model.parameters())

tokenizer = AutoTokenizer.from_pretrained('gpt2')
dataset = RolloutsAsLanguageModellingTask(tokenizer, './decoded_rollouts')
data = torch.utils.data.DataLoader(dataset, shuffle=False)

model, optimizer, data = accelerator.prepare(model, optimizer, data)

model.train()
for epoch in range(10):
    for batch in tqdm(data, desc=f'Epoch {epoch}'):

        optimizer.zero_grad()

        output = model(**batch)
        
        loss = output.loss

        accelerator.backward(loss)

        optimizer.step()