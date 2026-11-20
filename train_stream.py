import argparse
from dataclasses import dataclass
import json
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset.dataset import CustomBatchSampler, ChatJsonlDataset, CustomDistributedBatchSampler
from transformers import AutoTokenizer
from model.hub import get_backbone_id
from model.model import stream_tn_model
from trainer_stream import Trainer

from datasets import load_dataset, Features, Value
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs
import random
import numpy as np



@dataclass
class TrainingConfig:
    data_path: str 
    eval_data_path: str
    scheduler_steps: int = 150_000
    num_warmup_steps: int = 2_000
    learning_rate: float = 1e-4
    max_tokens: int = 4_000
    max_batch_size: int = 64
    device: str = "cuda"
    precision: str = "bf16"
    freeze_backbone: bool = False
    resume_from_checkpoint: str | None = None
    finetune: bool = False



def set_all_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config: TrainingConfig, project_name: str, exp_dir: str):
    
    tokenizer = AutoTokenizer.from_pretrained(
    get_backbone_id(),
    padding_side="right",    # 右侧补零，LLM训练标准配置
    truncation_side="right", # 右侧截断
)
    # Start from the raw backbone, or from a prepared HF model dir/repo id.
    model = stream_tn_model(model_id=config.resume_from_checkpoint or get_backbone_id())
    use_wandb = False
    try:
        import wandb
        use_wandb = True
    except ImportError:
        pass

    accelerate_kwargs={
        "log_with": "wandb" if use_wandb else None,
        "mixed_precision": "bf16" if config.precision == "bf16" else "no",
        
    }
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
    )
    accelerator = Accelerator(
                # split_batches=False, 
                # dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
                kwargs_handlers=[ddp_kwargs],
                **accelerate_kwargs,
            )
    accelerator.even_batches = False

    # dataset_path = Path(config.data_path)
    dataset_path = config.data_path
    # jsonl_dataset = ChatJsonlDataset(dataset_path, tokenizer=tokenizer)
    jsonl_dataset = ChatJsonlDataset(
        dataset_path, 
        tokenizer=tokenizer,
        shuffle_on_load=True,  # 加载时打乱
        seed=args.seed  # 随机种子
    )
    batch_sampler = CustomDistributedBatchSampler(
        dataset=jsonl_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        max_length=500,        # 单条样本最大帧数
        num_buckets=10,
        max_batch_length=config.max_tokens, # 一个 batch 总帧数上限
        drop_last=True,
    )
    # batch_sampler = CustomBatchSampler(
    #     dataset=jsonl_dataset,
    #     shuffle=True,
    #     drop_last=False,
    #     max_batch_length=config.max_tokens,
    #     max_length=400,
    #     num_buckets=10
    # )
    dataloader = DataLoader(
        dataset=jsonl_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,     # 用于加载数据的子进程数
        pin_memory=True,   # 加速GPU训练
        persistent_workers=False,
        collate_fn=jsonl_dataset.chat_custom_collate_fn
    )
    eval_dataloader = None
    if config.eval_data_path:
        eval_dataset_path = config.eval_data_path
        eval_jsonl_dataset = ChatJsonlDataset(eval_dataset_path, tokenizer=tokenizer)
        eval_batch_sampler = CustomDistributedBatchSampler(
            dataset=eval_jsonl_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            max_length=600,        # 单条样本最大帧数
            num_buckets=4,
            max_batch_length=config.max_tokens, # 一个 batch 总帧数上限
            drop_last=False,
        )
        # eval_batch_sampler = CustomBatchSampler(
        #     dataset=eval_jsonl_dataset,
        #     shuffle=True,
        #     drop_last=False,
        #     max_batch_length=config.max_tokens,
        #     max_length=600,
        #     num_buckets=10
        # )
        eval_dataloader = DataLoader(
            dataset=eval_jsonl_dataset,
            batch_sampler=eval_batch_sampler,
            num_workers=4,     # 用于加载数据的子进程数
            pin_memory=True,   # 加速GPU训练
            persistent_workers=False,
            collate_fn=jsonl_dataset.chat_custom_collate_fn
        )
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [{"params": decay_params, "weight_decay": 1e-2}, {"params": nodecay_params, "weight_decay": 0.0}]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Total decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"Total non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optimizer = AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)


    trainer = Trainer(
        model=model,
        accelerator=accelerator,
        optimizer=optimizer,
        train_loader=dataloader,
        eval_loader=eval_dataloader,
        num_warmup_steps=config.num_warmup_steps, # fintune改为几百
        scheduler_steps=config.scheduler_steps,
        checkpoint_dir=exp_dir,
    )

    trainer.train(total_steps=1500_000, save_every=17_000, validate_every=1000000, project_name=project_name, use_wandb=use_wandb)


# Usage: `accelerate launch train.py <config>`

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = json.load(f)
    config = TrainingConfig(**config_data)
    set_all_seed(args.seed)

    train(config,args.project_name,args.exp_dir)
