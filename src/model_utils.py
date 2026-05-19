import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.training.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.training.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.training.lora_r,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            target_modules=config.training.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def get_vllm_llm(model_name: str, tensor_parallel_size: int = 1):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    return llm


def sequence_logprob(model, tokenizer, prompt: str, response_ids: torch.Tensor) -> float:
    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)

    full_ids = torch.cat([prompt_enc.input_ids[0], response_ids.to(device)])
    full_attn = torch.ones_like(full_ids).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=full_ids.unsqueeze(0), attention_mask=full_attn)
        logits = outputs.logits[0]

    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    prompt_len = prompt_enc.input_ids.shape[1]
    for i in range(len(response_ids)):
        token_id = response_ids[i].item()
        total_logprob += logprobs[prompt_len + i - 1, token_id].item()

    return total_logprob


def compute_token_entropies(model, tokenizer, prompt: str, response_ids: torch.Tensor):
    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)

    full_ids = torch.cat([prompt_enc.input_ids[0], response_ids.to(device)])
    full_attn = torch.ones_like(full_ids).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=full_ids.unsqueeze(0), attention_mask=full_attn)
        logits = outputs.logits[0]

    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(logprobs)
    entropies = -(probs * logprobs).sum(dim=-1)

    prompt_len = prompt_enc.input_ids.shape[1]
    token_entropies = entropies[prompt_len - 1 : prompt_len - 1 + len(response_ids)]

    return token_entropies.cpu()


def recompute_logprobs_for_sequences(
    model, tokenizer, prompts: list, response_ids_list: list
) -> list:
    device = next(model.parameters()).device
    all_logprobs = []

    batch_encodings = tokenizer(
        [p + tokenizer.decode(r, skip_special_tokens=True) for p, r in zip(prompts, response_ids_list)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)

    prompt_encodings = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096
    ).to(device)

    with torch.no_grad():
        outputs = model(**batch_encodings)
        logits = outputs.logits

    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    for b in range(len(prompts)):
        prompt_len = prompt_encodings.attention_mask[b].sum().item()
        resp_len = (
            batch_encodings.attention_mask[b].sum().item() - prompt_len
        )
        seq_logprob = 0.0
        token_logprobs = []
        for t in range(resp_len):
            pos = prompt_len + t - 1
            if pos >= 0 and pos < logprobs.shape[1]:
                tok_logp = logprobs[b, pos, response_ids_list[b][t].item()].item()
                seq_logprob += tok_logp
                token_logprobs.append(tok_logp)
        all_logprobs.append((seq_logprob, token_logprobs))

    return all_logprobs
