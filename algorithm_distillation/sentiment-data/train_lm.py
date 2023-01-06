import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from rollouts_as_lm_task import RolloutsAsLanguageModellingTask
from utils import ShuffledIterableDataset
from tqdm.auto import tqdm

wandb.init(project="algorithm-distillation")
logging_table = wandb.Table(columns=["Step", "Generation"])

accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained('gpt2')
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
train_dataset = RolloutsAsLanguageModellingTask(tokenizer, './decoded_rollouts/train')
eval_dataset = RolloutsAsLanguageModellingTask(tokenizer, './decoded_rollouts/eval', verbose=False)
generate_dataset = RolloutsAsLanguageModellingTask(tokenizer, './decoded_rollouts/eval', for_generation=True, verbose=False)

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
        wandb.log({'loss': loss.item(), 'step': total_steps})
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
            wandb.log({'eval_loss': eval_loss, 'step': total_steps})
            print(f'Avg loss after {total_steps}: {eval_loss}')
            model.train()
            
        # generate examples
        generate_steps = 10
        generate_size = 3
        if total_steps % generate_steps == 0:
            model.eval()
            
            generate_dataloader = torch.utils.data.DataLoader(generate_dataset, shuffle=False)
            generate_dataloader = accelerator.prepare(generate_dataloader)
            
            for idx, batch in enumerate(generate_dataloader):
                if idx >= generate_size:
                    break
                batch = {k:v.squeeze(0) for k,v in batch.items()}
                outputs = model.generate(**batch, max_length=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(outputs[0])
                
                logging_table.add_data(total_steps, text)
            
            logging_table = wandb.Table(columns=logging_table.columns, data=logging_table.data)
            wandb.log({'Generations Table': logging_table})
            
            
            model.train()
                