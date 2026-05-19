import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import time
import math

try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False
    wandb = None

from .mh_sampling import mh_power_sampling
from .verifier import compute_rewards
from .grpo import compute_grpo_loss
from .diagnostics import log_mh_diagnostics, log_group_quality, log_training_metrics


class BranchGRPOTrainer:
    def __init__(self, config, model, tokenizer, train_data, val_data):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.device = next(model.parameters()).device
        self.global_step = 0

        self._setup_optimizer()
        self._setup_lr_scheduler()

    def _setup_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def _setup_lr_scheduler(self):
        num_training_steps = (
            len(self.train_data)
            // self.config.training.per_device_batch_size
            // self.config.training.gradient_accumulation_steps
            * self.config.training.num_epochs
        )
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(self):
        batch_size = self.config.training.per_device_batch_size
        accumulation_steps = self.config.training.gradient_accumulation_steps

        for epoch in range(self.config.training.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_clip_frac = 0.0
            num_batches = 0
            generated_tokens = 0
            verifier_calls = 0
            start_time = time.time()

            indices = list(range(len(self.train_data)))
            import random

            random.shuffle(indices)

            progress = tqdm(
                range(0, len(indices), batch_size),
                desc=f"Epoch {epoch + 1}/{self.config.training.num_epochs}",
            )

            batch_loss = 0.0
            for batch_start in progress:
                batch_indices = indices[batch_start : batch_start + batch_size]
                batch_data = [self.train_data[i] for i in batch_indices]
                prompts = [d["prompt"] for d in batch_data]
                answers = [d["answer"] for d in batch_data]

                candidate_pools = []
                rewards_list = []

                for prompt, answer in zip(prompts, answers):
                    pool = mh_power_sampling(
                        self.model, self.tokenizer, prompt, self.config
                    )
                    rewards = compute_rewards(pool, answer)

                    for c, r in zip(pool, rewards):
                        c["reward"] = r

                    candidate_pools.append(pool)
                    rewards_list.append(rewards)

                    for c in pool:
                        generated_tokens += len(c["response_ids"])
                    verifier_calls += len(pool)

                loss, metrics = compute_grpo_loss(
                    self.model,
                    self.tokenizer,
                    prompts,
                    candidate_pools,
                    rewards_list,
                    clip_epsilon=self.config.training.clip_epsilon,
                    loss_mask=self.config.loss_mask,
                )

                loss = loss / accumulation_steps
                loss.backward()
                batch_loss += loss.item()

                if (num_batches + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += metrics.get("loss", loss.item())
                epoch_kl += metrics.get("approx_kl", 0)
                epoch_clip_frac += metrics.get("clip_fraction", 0)
                num_batches += 1

                if num_batches % self.config.logging.log_every_n_steps == 0:
                    log_mh_diagnostics(candidate_pools, self.global_step)
                    log_group_quality(candidate_pools, self.global_step)
                    log_training_metrics(
                        {
                            "loss": epoch_loss / max(num_batches, 1),
                            "approx_kl": epoch_kl / max(num_batches, 1),
                            "clip_fraction": epoch_clip_frac / max(num_batches, 1),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "generated_tokens": generated_tokens,
                            "verifier_calls": verifier_calls,
                        },
                        self.global_step,
                    )

                progress.set_postfix(
                    {
                        "loss": f"{epoch_loss / max(num_batches, 1):.4f}",
                        "kl": f"{epoch_kl / max(num_batches, 1):.4f}",
                        "clip": f"{epoch_clip_frac / max(num_batches, 1):.4f}",
                    }
                )

            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_kl = epoch_kl / max(num_batches, 1)
            avg_clip = epoch_clip_frac / max(num_batches, 1)

            print(
                f"Epoch {epoch + 1} | "
                f"Loss: {avg_loss:.4f} | KL: {avg_kl:.4f} | "
                f"Clip: {avg_clip:.4f} | Time: {epoch_time:.1f}s | "
                f"Tokens: {generated_tokens} | Verifier: {verifier_calls}"
            )

            if (epoch + 1) % self.config.logging.eval_every_n_epochs == 0:
                self.evaluate(epoch)

            if (epoch + 1) % self.config.logging.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

    def evaluate(self, epoch: int):
        from .eval import evaluate_on_math500

        metrics = evaluate_on_math500(
            self.model, self.tokenizer, self.config, self.val_data
        )
        from .diagnostics import log_eval_metrics

        log_eval_metrics(metrics, self.global_step)
        print(f"Eval Epoch {epoch + 1}: Accuracy={metrics['accuracy']:.4f}")

    def save_checkpoint(self, epoch: int):
        import os

        save_dir = os.path.join(
            self.config.logging.output_dir,
            self.config.get_method_name(),
            f"epoch_{epoch + 1}",
        )
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Checkpoint saved to {save_dir}")
