from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class MHConfig:
    alpha: float = 1.5
    mh_steps: int = 4
    span_len: int = 16
    proposal: Literal["entropy_branch_span_then_continue", "random_position"] = (
        "entropy_branch_span_then_continue"
    )
    branch_selection: Literal[
        "entropy_top1", "entropy_top4_random_one", "random_position"
    ] = "entropy_top1"


@dataclass
class MethodConfig:
    method: Literal[
        "standard_grpo",
        "low_temp_grpo",
        "mh_final_only",
        "mh_chain_only",
        "mh_all_proposals",
        "mh_all_proposals_dedup",
    ] = "mh_all_proposals"
    num_rollouts_per_prompt: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    mh: Optional[MHConfig] = None
    exact_dedup: bool = False
    include_chain_states: bool = False
    include_rejected: bool = True


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-Math-7B"
    train_prompts: int = 500
    max_response_length: int = 2048
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    clip_epsilon: float = 0.2
    kl_penalty_coef: float = 0.0


@dataclass
class DataConfig:
    train_dataset: str = "hendrycks/competition_math"
    train_subset: str = "all"
    train_split: str = "train"
    val_dataset: str = "hendrycks/competition_math"
    val_subset: str = "all"
    val_split: str = "test"
    prompt_template: str = (
        "Solve the following math problem step by step. "
        "Put your final answer within \\boxed{{}}.\n\n{problem}\n\n"
    )


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    wandb_project: str = "branch-grpo"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    output_dir: str = "./outputs"
    run_name: Optional[str] = None


@dataclass
class ExperimentConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    bf16: bool = True
    loss_mask: Literal["full_response", "branch_after_only"] = "branch_after_only"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml

        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> "ExperimentConfig":
        if "mh" in d.get("method", {}):
            mh = MHConfig(**d["method"]["mh"])
            mc = MethodConfig(**{**d["method"], "mh": mh})
        else:
            mc = MethodConfig(**d.get("method", {}))
        tc = TrainingConfig(**d.get("training", {}))
        dc = DataConfig(**d.get("data", {}))
        lc = LoggingConfig(**d.get("logging", {}))
        return cls(
            training=tc,
            method=mc,
            data=dc,
            logging=lc,
            seed=d.get("seed", 42),
            bf16=d.get("bf16", True),
            loss_mask=d.get("loss_mask", "branch_after_only"),
        )

    def get_method_name(self) -> str:
        m = self.method
        if m.method == "standard_grpo":
            return f"standard_grpo_g{m.num_rollouts_per_prompt}"
        elif m.method == "low_temp_grpo":
            return f"low_temp_grpo_g{m.num_rollouts_per_prompt}_t{m.temperature}"
        elif m.method == "mh_final_only":
            return f"mh_final_only_k{m.mh.mh_steps}_a{m.mh.alpha}"
        elif m.method == "mh_chain_only":
            return f"mh_chain_only_k{m.mh.mh_steps}_a{m.mh.alpha}"
        elif m.method == "mh_all_proposals":
            if m.exact_dedup:
                return f"mh_all_proposals_k{m.mh.mh_steps}_a{m.mh.alpha}_dedup"
            return f"mh_all_proposals_k{m.mh.mh_steps}_a{m.mh.alpha}"
        return m.method
