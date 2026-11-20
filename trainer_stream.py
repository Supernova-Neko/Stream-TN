from pathlib import Path
import time

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR, ConstantLR, LambdaLR
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs

from tqdm import tqdm
import random
from model.model import stream_tn_model

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None

class Trainer:
    def __init__(
        self,
        model: stream_tn_model,
        accelerator,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_warmup_steps=10_000,
        scheduler_steps=150_000,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.accelerator = accelerator
        self.print = self.accelerator.print

        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.num_warmup_steps = num_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        # self.create_scheduler(scheduler_steps, True)
        self.create_scheduler(
            scheduler_steps=scheduler_steps,
            warmup_steps=num_warmup_steps,
            cosine_steps=scheduler_steps - num_warmup_steps,
        )




        self.model,self.optimizer,self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def num_params(model: nn.Module) -> int:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.print(f"Backbone parameter count: {num_params(model.model):,}")
        self.print(f"*Total parameter count*: {num_params(model):,}")

    def create_scheduler(self,
                        scheduler_steps: int = 1_050_000,
                        warmup_steps: int = 5_000,
                        cosine_steps: int = 1_000_000,
                        min_lr: float = 1e-6):
        """
        三段式：
        0 ~ warmup_steps                 : 线性 warmup
        warmup_steps ~ warmup+cosine     : 余弦退火到 min_lr
        warmup+cosine ~ scheduler_steps  : 恒定为 min_lr
        """
        constant_ratio = min_lr / self.optimizer.param_groups[0]['lr']

        # if cosine_steps < 0:
        cosine_steps = (scheduler_steps - warmup_steps) * self.accelerator.num_processes
        warmup_steps_actual = warmup_steps * self.accelerator.num_processes 
        # 1. warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_steps_actual,
        )
        # print(scheduler_steps - warmup_steps, self.accelerator.num_processes, cosine_steps)
        # 2. 余弦退火段（降到 min_lr）
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=min_lr,
        )

        # # 3. 恒定段（保持 min_lr）
        # constant_steps = max(scheduler_steps - warmup_steps - cosine_steps, 0)
        # constant_scheduler = ConstantLR(
        #     self.optimizer,
        #     factor=1.0,               # 保持 cosine 结束时的 lr（即 min_lr）
        #     total_iters=constant_steps,
        # )


        constant_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda _: constant_ratio  # 始终返回固定比例
        )
        constant_scheduler.total_iters = 100_000_000_000 
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
            milestones=[warmup_steps_actual, warmup_steps_actual + cosine_steps],
        )



    def safe_backward_and_step(self, loss, global_step):
        # 周期震荡跟这里无关。
        try:
            self.accelerator.backward(loss)

            if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            return True

        except RuntimeError as e:
            if "timeout" in str(e).lower() or "nccl" in str(e).lower():
                self.print(f"NCCL timeout detected at step {global_step}: {e}")
                # try to recover by skipping this step
                self.optimizer.zero_grad()
                return False
            else:
                raise e
            
    @torch.no_grad()  # 新增：验证阶段禁用梯度
    def validate_epoch(self, global_step):
        """
        执行一轮验证，返回平均loss和准确率
        """
        if self.eval_loader is None:
            self.print("无eval_loader，跳过验证")
            return None
        
        self.print(f"\n开始验证（global_step: {global_step}）...")
        # 切换模型到eval模式
        self.model.eval()
        
        total_eval_loss = 0.0
        total_eval_acc = 0.0
        num_eval_batches = 0
        
        # 验证进度条
        eval_pbar = tqdm(
            self.eval_loader,
            desc="Validating",
            disable=not self.accelerator.is_local_main_process,
        )
        
        # 设备和精度配置（和训练一致）
        device_type = self.accelerator.device.type
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        
        for batch in eval_pbar:
            # 数据移到设备
            tokens = batch["tokens"].to(torch.long).to(device_type)
            target_ids = batch["target_ids"].to(torch.long).to(device_type)
            # sys_tokens_len = batch["sys_tokens_len"]
            tokens_mask = batch["attention_mask"].to(torch.long).to(device_type)
            
            # 前向计算（禁用autocast？或和训练一致，根据需求调整）
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
                # outputs = self.model(tokens, target_ids, sys_tokens_len, tokens_mask)
                outputs = self.model(tokens, target_ids, tokens_mask)
            # 累加指标
            total_eval_loss += outputs["loss"].item()
            total_eval_acc += outputs["accuracy"].item()
            num_eval_batches += 1
            
            # 更新进度条
            eval_pbar.set_postfix(
                eval_loss=f"{outputs['loss'].item():.3f}",
                eval_acc=f"{outputs['accuracy'].item():.3f}"
            )
        
        # 计算平均指标
        avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else 0.0
        avg_eval_acc = total_eval_acc / num_eval_batches if num_eval_batches > 0 else 0.0
        
        # 记录验证指标
        self.accelerator.log(
            {
                "eval_loss": avg_eval_loss,
                "eval_accuracy": avg_eval_acc,
            },
            step=global_step,
        )
        
        self.print(f"验证完成 | eval_loss: {avg_eval_loss:.4f} | eval_acc: {avg_eval_acc:.4f}")
        
        # 切回train模式
        self.model.train()
        
        return {"eval_loss": avg_eval_loss, "eval_acc": avg_eval_acc}
    
    def train(
        self,
        total_steps: int = 1_000_000_000_000,
        save_every: int = 20_000,
        validate_every: int = 400,
        project_name="default",
        use_wandb=False,
        epoch=None,

    ):
        if self.accelerator.is_main_process:
            if use_wandb:
                wandb_run = wandb.init(
                    name=project_name,
                )

        hps = {
            "total_steps": total_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "save_every": save_every,
        }
        self.accelerator.init_trackers("StreamTn", config=hps)

        # global_step = start_step if start_step else 0
        global_step = 0
        total_tokens = 0
        consecutive_timeouts = 0
        epoch = epoch if epoch else 1
        max_consecutive_timeouts = 3
        print(f'start train')
        cnt = 0

        while global_step < total_steps:
        # while True:
            self.optimizer.train()
            self.model.train()

            # set epoch for distributed sampler if needed
            if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(epoch)

            running_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                unit="",
                disable=not self.accelerator.is_local_main_process,
            )

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            batch_record_lst = []

            for batch_idx, batch in enumerate(progress_bar):
                # import pdb;pdb.set_trace()
                tokens = batch["tokens"].to(torch.long).to(device_type)
                target_ids = batch["target_ids"].to(torch.long).to(device_type)
                # sys_tokens_len = batch["sys_tokens_len"]
                tokens_mask = batch["attention_mask"].to(torch.long).to(device_type)

                if tokens.shape[0] == 0:
                    print('跳过空batch')
                    continue

                total_tokens += tokens.shape[0] * tokens.shape[1]

                batch_record_lst.append(tokens.shape[0])
                with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
                    losses = self.model(tokens, target_ids, tokens_mask)
                    # losses = self.model(tokens, target_ids, sys_tokens_len, tokens_mask)
                loss = losses["loss"]

                success = self.safe_backward_and_step(loss, global_step)
                if not success:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        self.print("Max consecutive timeouts reached. Stopping training.")
                        break
                    continue
                else:
                    consecutive_timeouts = 0

                running_loss += loss.item()
                num_batches += 1
                global_step += 1

                if self.accelerator.is_local_main_process and global_step % 10 == 0:
                    loss = losses["loss"]
                    c0_accuracy = losses["accuracy"]

                    self.accelerator.log(
                        {
                            "loss": loss.item(),
                            "c0_accuracy": c0_accuracy.item(),
                            "total_tokens": total_tokens,
                            "batch_size": sum(batch_record_lst) / len(batch_record_lst),
                            "lr": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                    
                    progress_bar.set_postfix(
                        step=f"{global_step}",
                        loss=f"{loss.item():.3f}",
                        c0_acc=f"{c0_accuracy.item():.3f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.6f}",
                        tokens=f"{total_tokens / 1_000_000:.2f}M",
                        batch_size=f"{sum(batch_record_lst) / len(batch_record_lst):.2f}"
                    )
                    batch_record_lst = []

                # if global_step == 30:
                #     ckpt_name = f"model_{epoch}_{global_step}_test"
                #     save_path = self.checkpoint_dir / ckpt_name

                #     self.accelerator.save_state(save_path)

                #     self.print(f"rank-0 保存测试检查点：{ckpt_name}")
                
                if global_step % save_every == 0:
                    print("保存ckpt")
                    
                    self.accelerator.save_state(self.checkpoint_dir / f"model_{epoch}_{global_step}")

                # 每隔指定步数执行验证
                if global_step % validate_every == 0 and self.accelerator.is_main_process:
                    self.validate_epoch(global_step)
                    
            if self.accelerator.is_main_process and self.eval_loader is not None:
                self.validate_epoch(global_step)

            if num_batches == 0:
                self.print("No batches were produced. Check the training data size and max_tokens.")
                break
            avg_loss = running_loss / num_batches
            self.print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
            self.accelerator.log({"avg_epoch_loss": avg_loss}, step=global_step)
            if self.accelerator.is_main_process:
                self.accelerator.save_state(self.checkpoint_dir / f"model_{epoch}_{global_step}_final")
            # if self.val_loader and global_step % validate_every == 0:
            #     val_loss = self.validate_epoch(global_step)
            #     if val_loss is not None:
 
            epoch += 1
            if consecutive_timeouts >= max_consecutive_timeouts:
                break
        
        # self.save_checkpoint("final", epoch, total_tokens)
        self.accelerator.end_training()
        self.print("Training complete.")
