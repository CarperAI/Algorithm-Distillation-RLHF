from pathlib import Path
import json
from transformers import AutoTokenizer
import click
from typing import Union
from shutil import copy2

@click.group()
def main():
    """
    CLI for formatting the rollouts into training data.
    
    \b
    1. decode-epoch: Decode a single epoch rollout .json file
    2. decode-run: Decode an entire PPO run (multiple epoch files)
    3. decode-rollouts: Deocde an entire directory (multiple runs)
    """

def get_tokenizer(tokenizer: Union[str, AutoTokenizer]) -> AutoTokenizer:
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer)
    else:
        return tokenizer


@main.command()
@click.option('--tokenizer', type=str, default='gpt2', help='tokenizer to decode with')
@click.option('--input-fpath', type=click.Path(path_type=Path), help='the input JSON file')
@click.option('--output-fpath', type=click.Path(path_type=Path), help='the path of the JSON file to be created. Will overwrite if necessary.')
def decode_epoch(tokenizer: Union[str, AutoTokenizer], input_fpath: Path, output_fpath: Path):
    _decode_epoch(tokenizer, input_fpath, output_fpath)
    
def _decode_epoch(tokenizer: Union[str, AutoTokenizer], input_fpath: Path, output_fpath: Path):
    
    assert input_fpath.exists()
    assert input_fpath.is_file()
    assert input_fpath.name.endswith('.json')
    assert output_fpath.name.endswith('.json')
    if not output_fpath.parent.exists():
        output_fpath.parent.mkdir()
    
    tokenizer = get_tokenizer(tokenizer)
    
    with open(input_fpath, 'r') as f:
        rollouts = json.loads(f.read())
    
    for rollout in rollouts:
        rollout['query_text'] = tokenizer.decode(rollout['query_tensor'], skip_special_tokens=True)
        rollout['response_text'] = tokenizer.decode(rollout['response_tensor'], skip_special_tokens=True)
    
    with open(output_fpath, 'w') as f:
        f.write(json.dumps(rollouts, indent=2))

        
@main.command()
@click.option('--tokenizer', type=str, default='gpt2', help='tokenizer to decode with')
@click.option('--input-fpath', type=click.Path(path_type=Path), help='the directory containing the JSON epoch files')
@click.option('--output-fpath', type=click.Path(path_type=Path), help='the path of the folder to be created.')       
def decode_run(tokenizer: Union[str, AutoTokenizer], input_fpath: Path, output_fpath: Path):
    _decode_run(tokenizer, input_fpath, output_fpath)

def _decode_run(tokenizer: Union[str, AutoTokenizer], input_fpath: Path, output_fpath: Path):
    
    assert input_fpath.exists()
    assert input_fpath.is_dir()
    if not output_fpath.exists():
        output_fpath.mkdir()
    
    # Copy over the config file
    assert (input_fpath / 'config.json').exists()
    copy2(input_fpath / 'config.json', output_fpath / 'config.json')
    
    # Decode the rest of the files 
    tokenizer = get_tokenizer(tokenizer)
    
    epochs = [fpath for fpath in input_fpath.iterdir() if fpath.name != 'config.json']
    for epoch in epochs:
        _decode_epoch(tokenizer, epoch, output_fpath / epoch.name)

       
@main.command()
@click.option('--tokenizer', type=str, default='gpt2', help='tokenizer to decode with')
@click.option('--input-fpath', type=click.Path(path_type=Path), help='the input directory')
@click.option('--output-fpath', type=click.Path(path_type=Path), help='the output directory')       
def decode_rollouts(tokenizer: str, input_fpath: Path, output_fpath: Path):
    _decode_rollouts(tokenizer, input_fpath, output_fpath)

def _decode_rollouts(tokenizer: str, input_fpath: Path, output_fpath: Path):
    
    assert input_fpath.exists()
    assert input_fpath.is_dir()
    if not output_fpath.exists():
        output_fpath.mkdir()
    
    tokenizer = get_tokenizer(tokenizer)
    runs = [fpath for fpath in input_fpath.iterdir() if fpath.name.startswith('run-')]
    for run in runs:
        _decode_run(tokenizer, run, output_fpath / run.name)

if __name__ == '__main__':
    main()