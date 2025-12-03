import torch
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import neptune
from dataclasses import dataclass, field
import importlib
import yaml
import os
import glob
from google.colab import drive
drive.mount('/content/drive')


def load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

with open("configurate.yaml", "r") as f:
    cfg = yaml.safe_load(f)

@dataclass
class ModelConfig:
    name: str = "LLaMA-pretrained"
    dtype: str = "float16"
    device: str = "cuda"
    embed_size: int = 960
    vocab_size: int = 32000
    num_heads: int = 15
    loss_func: str = "torch.nn.CrossEntropyLoss"
    scheduler: str = "torch.optim.lr_scheduler.CosineAnnealingLR"

@dataclass
class OptimizerConfig:
    name: str = "torch.optim.AdamW"
    args: dict = field(default_factory=lambda: {
        "lr": 3e-4,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1
    })

@dataclass
class TrainingConfig:
    total_token: int = 250e6
    token_per_step: int = 250e3
    tolerance: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


config = Config()

run = neptune.init_run(
    project = cfg['logger']['project'],
    api_token = cfg['logger']['token'],
    with_id='LLAMA-70'
    
)

run["parameters"] = {
    "target_tokens": config.training.total_token,
    "tolerance": config.training.tolerance,
    "max_grad_norm": config.training.max_grad_norm,
    "learning_rate": config.optimizer.args['lr']
}

class train_loop:
    def __init__(self, model, config: Config = config, run: neptune = run):
        self.model = model
        self.run = run
        self.loss_fn = load_class(config.model.loss_func)()
        lr = config.optimizer.args['lr']
        T_max = config.training.total_token // config.training.token_per_step
        opt_conf = config.optimizer
        opt_class = load_class(opt_conf.name)
        self.optimizer = opt_class(self.model.parameters(), **opt_conf.args)
        scheduler_class = load_class(config.model.scheduler)
        self.scheduler = scheduler_class(self.optimizer, T_max=T_max, eta_min=0.1*lr)
        self.scaler = torch.cuda.amp.GradScaler()
        self.device = "cuda"
        
        self.ckpt_dir = "/content/drive/MyDrive/checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
    
    
    def save_checkpoint(self, step, tokens):
        ckpt_path = f"{self.ckpt_dir}/model_step{step}_tokens{tokens}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": step,
            "tokens": tokens
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    
    def load_check(self, checkpoint):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    
    def cleanup_old(self, keep_last=3):
        ckpts = sorted(glob.glob(f"{self.ckpt_dir}/*.pt"))
        while len(ckpts) > keep_last:
            os.remove(ckpts[0])
            ckpts = ckpts[1:]
    
    def train(self, data_loader: DataLoader, checkpoint: dict = None, token_target: int = config.training.token_per_step, tolerance: float = config.training.tolerance):
        if checkpoint is not None:
            self.load_check(checkpoint)
            step = checkpoint["step"]
            tokens = checkpoint["tokens"]
        else:
            step = 0
            tokens = 0
        self.model.train()
        Loss = 0.0 
        tok_accum = 0  
        pbar = tqdm(initial=tokens, total=config.training.total_token)

        for x, length in data_loader:
            x = x.to(self.device, non_blocking=True)
            cur_tokens = length

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                logits = self.model(x[:, :-1])
                
                loss = self.loss_fn(
                    logits.reshape(-1, config.model.vocab_size), 
                    x[:, 1:].reshape(-1)
                )
                loss = loss * cur_tokens / token_target

            self.scaler.scale(loss).backward()

            tok_accum += cur_tokens
            Loss += loss.item()

            pbar.update(cur_tokens)

            if tok_accum >= token_target * (1 - tolerance):

                self.scaler.unscale_(self.optimizer)

                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=config.training.max_grad_norm
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                self.scheduler.step()

                self.run["train/loss"].append(Loss)
                self.run["train/tokens_per_step"].append(tok_accum)
                self.run["train/learning_rate"].append(self.optimizer.param_groups[0]['lr'])
                self.run["train/grad_norm"].append(total_norm)

                tok_accum = 0
                Loss = 0
                
                if pbar.n // 2.5e7 >= step + 1:
                    step += 1
                    self.save_checkpoint(step, pbar.n)
                    self.cleanup_old()

                if pbar.n >= config.training.total_token:
                    break
