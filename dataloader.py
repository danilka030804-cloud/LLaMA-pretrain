from transformers import AutoTokenizer
from huggingface_hub import login
import yaml
from torch.utils.data import DataLoader
import torch


with open("configurate.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def tokenizer(config = cfg['tokenizer']):
    login(token=config['login'])

    tokenizer = AutoTokenizer.from_pretrained(config['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


tok = tokenizer()

def fn(text, length: int = 256, max_bs: int = 20, tokenizer = tok):
    text = " ".join(text)
    tokens = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False,
            padding=False
        )['input_ids']
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    bs = min(tokens.shape[0] // length, max_bs)
    if bs == 0:
        return torch.zeros((1, length), dtype=torch.long,  pin_memory=True), length
    
    result = torch.empty((bs, length), dtype=torch.long, pin_memory=True)
    
    tokens = tokens[:bs * (length-2)].view(bs, (length-2))

    result[:, 0] = 1
    result[:, 1:-1] = tokens
    result[:, -1] = 2

    return result, bs * (length-2)





def create_dataloader(dataset):
    return DataLoader(
        dataset['train']['text'],
        batch_size=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=fn,)