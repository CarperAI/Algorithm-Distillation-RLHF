import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from rollouts_as_lm_task import RolloutsAsLanguageModellingTask
from tqdm.auto import tqdm

accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained('gpt2')
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
train_dataset = RolloutsAsLanguageModellingTask(tokenizer, './decoded_rollouts/train')
eval_dataset = RolloutsAsLanguageModellingTask(tokenizer, './decoded_rollouts/eval', verbose=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

model.train()
total_steps = 0
for epoch in range(10):
    for batch in tqdm(train_dataloader, desc=f'Training epoch {epoch}'):
        # train
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        accelerator.backward(loss)
        optimizer.step()
        total_steps += 1
        
        # eval
        eval_steps = 25
        eval_size = 20
        if total_steps % eval_steps == 0:
            model.eval()
            eval_loss = 0
            eval_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False)
            eval_dataloader = accelerator.prepare(eval_dataloader)
            for idx, batch in tqdm(enumerate(eval_dataloader), desc=f'Eval after {total_steps} steps.'):
                if idx >= eval_size:
                    break
                output = model(**batch)
                loss = output.loss
                eval_loss += loss.item()
            eval_loss /= idx
            print(f'Avg loss after {total_steps}: {eval_loss}')
            model.train()
                