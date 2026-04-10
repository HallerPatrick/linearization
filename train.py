from itertools import chain
import csv
import json
import numpy as np
import math
import itertools
import datetime
import gc
import multiprocessing
import os
import shutil
import sys
import time

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_constant_schedule_with_warmup,
    get_scheduler,
)

try:
    # from models.qwen import TeacherQwen3ForCausalLM
    # from models.opt import TeacherOPTForCausalLM
    from models.llama import TeacherLlamaForCausalLM
except ModuleNotFoundError:
    TeacherQwen3ForCausalLM = TeacherOPTForCausalLM = TeacherLlamaForCausalLM = None
    print(
        "Warning: Could not import teacher model wrappers from models/. "
        "Stage-1 (token-mixer) training requires these classes."
    )

try:
    from models.modular_xlstm import xQwen3Config, xOPTConfig, xGemma3Config
    from models.modular_xlstm import xQwen3ForCausalLM, xOPTForCausalLM, xGemma3ForCausalLM
except ModuleNotFoundError:
    xQwen3Config = xOPTConfig = xGemma3Config = None
    xQwen3ForCausalLM = xOPTForCausalLM = xGemma3ForCausalLM = None
except ImportError:
    xQwen3Config = xOPTConfig = xGemma3Config = None
    xQwen3ForCausalLM = xOPTForCausalLM = xGemma3ForCausalLM = None


from optimizer import get_optimizer

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def human_readable_number(num: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"


def count_frozen_parameters(model):
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return frozen, total


def get_model_layers(model):
    """Return the transformer layer list, handling DDP wrapper and architecture differences."""
    unwrapped = model.module if hasattr(model, "module") else model
    if hasattr(unwrapped, "backbone"):
        return unwrapped.backbone.blocks
    if hasattr(unwrapped, "model"):
        return unwrapped.model.layers
    return unwrapped.backbone.layers


def save_model(model, model_config, tokenizer, output_dir):
    model_config.save_pretrained(output_dir)
    try:
        model.save_pretrained(output_dir)
    except Exception:
        model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)


def load_config(config_path: str) -> OmegaConf:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = OmegaConf.create(f.read())
    OmegaConf.resolve(config)
    return config


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def _resolve_teacher_cls(config, teacher_hf_config):
    """Pick the appropriate teacher model class based on config and teacher architecture."""
    model_name = config.training.model_name
    arch = teacher_hf_config.model_type

    if model_name in ("xqwen3", "xqwen") and TeacherQwen3ForCausalLM is not None:
        return TeacherQwen3ForCausalLM
    if model_name == "xopt" and TeacherOPTForCausalLM is not None:
        return TeacherOPTForCausalLM
    if model_name == "xllama" and TeacherLlamaForCausalLM is not None:
        return TeacherLlamaForCausalLM
    if arch in ("gemma4", "gemma4_text"):
        return AutoModelForCausalLM
    if arch.startswith("llama") and TeacherLlamaForCausalLM is not None:
        return TeacherLlamaForCausalLM
    if TeacherQwen3ForCausalLM is not None:
        return TeacherQwen3ForCausalLM
    return AutoModelForCausalLM


def _load_student_from_checkpoint(model_name, checkpoint, accelerator):
    """Load a student model and its config from a checkpoint directory."""
    if model_name == "xqwen3":
        hf_cfg = xQwen3Config.from_pretrained(checkpoint, mode="train")
        model = xQwen3ForCausalLM.from_pretrained(checkpoint, config=hf_cfg)
        return model, hf_cfg

    if model_name == "xopt":
        hf_cfg = xOPTConfig.from_pretrained(checkpoint, mode="train")
        model = xOPTForCausalLM.from_pretrained(checkpoint, config=hf_cfg)
        return model, hf_cfg

    if model_name == "xgemma":
        hf_cfg = xGemma3Config.from_pretrained(checkpoint, mode="train")
        model = xGemma3ForCausalLM.from_pretrained(checkpoint, config=hf_cfg)
        return model, hf_cfg

    # Generic student path
    import importlib
    model_module = importlib.import_module(f"students.{model_name}")
    hf_cfg = model_module.StudentConfig.from_pretrained(checkpoint)
    model = model_module.StudentModel(hf_cfg)

    safe_path = os.path.join(checkpoint, "model.safetensors")
    bin_path = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No weights found in {checkpoint} "
            "(expected model.safetensors or pytorch_model.bin)."
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        accelerator.print(
            f"Checkpoint loaded with {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys."
        )
    return model, hf_cfg


def _build_student_from_scratch(model_name, config, tokenizer):
    """Instantiate a fresh student model (no checkpoint)."""
    if model_name == "xqwen3":
        hf_cfg = xQwen3Config.from_pretrained(config.teacher_model.model_name, mode="train")
        return xQwen3ForCausalLM(hf_cfg), hf_cfg

    if model_name == "xopt":
        hf_cfg = xOPTConfig.from_pretrained(config.teacher_model.model_name, mode="train")
        return xOPTForCausalLM(hf_cfg), hf_cfg

    if model_name == "xgemma":
        hf_cfg = xGemma3Config.from_pretrained(config.teacher_model.model_name, mode="train")
        return xGemma3ForCausalLM(hf_cfg), hf_cfg

    # Generic student path
    import importlib
    try:
        model_module = importlib.import_module(f"students.{model_name}")
    except ImportError as e:
        raise ImportError(
            f"Could not import 'students.{model_name}'. Ensure the module exists."
        ) from e

    extra_configs = dict(config.model.get("extra_configs", {}))
    if "ssm_cfg" in extra_configs:
        extra_configs["ssm_cfg"] = dict(extra_configs["ssm_cfg"])

    if getattr(config.model, "from_scratch", False):
        hf_cfg = model_module.StudentConfig(vocab_size=tokenizer.vocab_size, **extra_configs)
    else:
        hf_cfg = model_module.StudentConfig.from_pretrained(
            config.teacher_model.model_name, **extra_configs
        )
    model = model_module.StudentModel(hf_cfg)
    return model, hf_cfg


def initialize_model_and_teacher(config, accelerator, tokenizer, output_dir):
    config.model.vocab_size = tokenizer.vocab_size
    accelerator.print("Creating model...")

    teacher_hf_config = AutoConfig.from_pretrained(config.teacher_model.model_name)
    model_name = config.training.model_name
    checkpoint = config.training.from_checkpoint

    if checkpoint is not None:
        accelerator.print(f"Loading model from checkpoint: {checkpoint}")
        model, hf_model_config = _load_student_from_checkpoint(model_name, checkpoint, accelerator)
    else:
        model, hf_model_config = _build_student_from_scratch(model_name, config, tokenizer)

    model = model.to(dtype=torch.bfloat16)
    accelerator.print(model)

    if config.training.get("gradient_checkpointing", False):
        accelerator.print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    num_params = sum(p.numel() for p in model.parameters())
    num_params_human = human_readable_number(num_params)
    accelerator.print(f"Number of parameters: {num_params:_} ({num_params_human})")

    teacher_model = None
    if config.teacher_model.use or config.model.copy_teacher_params:
        accelerator.print("Loading teacher model...")
        teacher_cls = _resolve_teacher_cls(config, teacher_hf_config)
        teacher_model = teacher_cls.from_pretrained(
            config.teacher_model.model_name,
            attn_implementation="eager",
            dtype=torch.bfloat16,
        )
        if accelerator.is_main_process:
            print(teacher_model)

    if config.model.copy_teacher_params and checkpoint is None:
        accelerator.print("Copying parameters from teacher model...")
        model.copy_from_teacher(teacher_model)

    if config.model.freeze:
        for name, param in model.named_parameters():
            if any(k in name for k in ("lm_head", "embeddings", "mlp")):
                param.requires_grad = False

    if not config.teacher_model.use:
        teacher_model = None
        torch.cuda.empty_cache()
        gc.collect()

    return model, teacher_model, hf_model_config, num_params, num_params_human


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------

def train_end_to_end(
    step, accelerator, train_dataloader, student_model, teacher_model,
    tokenizer, hf_model_config, config, output_dir,
):
    num_training_steps = math.ceil(
        len(train_dataloader) / config.training.gradient_accumulation_steps
    )
    accelerator.print(f"[End-to-End] Total training steps: {num_training_steps}")

    progress_bar = tqdm(
        total=num_training_steps // accelerator.num_processes,
        desc="[End-to-End Train]",
        unit="step",
        colour="GREEN",
        disable=not accelerator.is_local_main_process,
    )

    optimizer = get_optimizer(
        config.training.get("optimizer", "adamw"),
        student_model,
        lr=config.training.stage_3.lr,
        lr_1d=config.training.get("lr_1d", None),
    )

    lr_scheduler = _build_lr_scheduler(
        config.training.stage_3.lr_scheduler, optimizer, num_training_steps
    )

    student_model, teacher_model, optimizer, lr_scheduler, train_dataloader = (
        accelerator.prepare(student_model, teacher_model, optimizer, lr_scheduler, train_dataloader)
    )

    checkpoint_every = config.training.get("checkpoint_every", 32_500)

    for batch in train_dataloader:
        inputs = batch["input_ids"].to(accelerator.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(accelerator.device)

        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(inputs, attention_mask=attention_mask).logits

        with accelerator.accumulate(student_model):
            with accelerator.autocast():
                output = student_model(inputs, labels=inputs, attention_mask=attention_mask)
                loss = output.loss

            if teacher_model is not None:
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(output.logits, dim=-1),
                    torch.nn.functional.softmax(teacher_logits, dim=-1),
                    reduction="batchmean",
                )
                loss = loss + kl_loss

            if torch.isnan(loss):
                accelerator.print("NaN loss encountered. Stopping training.")
                optimizer.zero_grad()
                break

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        step += 1

        if (
            step % config.training.log_every_step == 0
            and config.training.log_every_step > 0
            and accelerator.sync_gradients
        ):
            last_lr = lr_scheduler.get_last_lr()[0]
            if config.training.wandb_enabled:
                accelerator.log({"loss": loss.item(), "lr": last_lr}, step=step)
            progress_bar.set_postfix({"loss": loss.item(), "lr": last_lr})
            progress_bar.update(config.training.log_every_step)

        if step % checkpoint_every == 0 and accelerator.is_main_process:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
            save_model(accelerator.unwrap_model(student_model), hf_model_config, tokenizer, checkpoint_dir)
            accelerator.print(f"Checkpoint saved to: {checkpoint_dir}")

    accelerator.print(f"End-to-end training done (process {accelerator.process_index})")
    return step, student_model, teacher_model, optimizer


def train_hidden_alignment(
    step, accelerator, train_dataloader, student_model, teacher_model,
    tokenizer, hf_model_config, config, output_dir,
):
    num_training_steps = len(train_dataloader)
    accelerator.print(f"[Hidden Alignment] Total training steps: {num_training_steps}")

    progress_bar = tqdm(
        total=num_training_steps // accelerator.num_processes,
        desc="[Hidden States Training]",
        unit="step",
        colour="YELLOW",
        disable=not accelerator.is_local_main_process,
    )

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.training.stage_2.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    lr_scheduler = _build_lr_scheduler(
        config.training.stage_2.lr_scheduler, optimizer, num_training_steps
    )

    student_model, teacher_model, optimizer, lr_scheduler, train_dataloader = (
        accelerator.prepare(student_model, teacher_model, optimizer, lr_scheduler, train_dataloader)
    )

    for batch in train_dataloader:
        inputs = batch["input_ids"].to(accelerator.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(accelerator.device)

        with torch.no_grad():
            teacher_outputs = teacher_model(
                inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        with accelerator.accumulate(student_model):
            total_loss = torch.zeros((), device=accelerator.device)
            layer_losses = []

            for layer_idx, student_layer in enumerate(get_model_layers(student_model)):
                teacher_in = teacher_outputs.hidden_states[layer_idx].to(torch.bfloat16)
                teacher_out = teacher_outputs.hidden_states[layer_idx + 1]

                if teacher_out.shape[-1] != hf_model_config.hidden_size:
                    continue

                position_embeddings = getattr(teacher_outputs, "position_embeddings", None)
                student_out = student_layer(
                    hidden_states=teacher_in,
                    position_embeddings=position_embeddings,
                )[0]

                layer_loss = torch.norm(student_out - teacher_out, p=2, dim=-1).mean()
                total_loss = total_loss + layer_loss
                layer_losses.append(layer_loss.item())

            if not layer_losses or total_loss.isnan():
                continue

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        step += 1

        if (
            step % config.training.log_every_step == 0
            and config.training.log_every_step > 0
            and accelerator.is_local_main_process
        ):
            avg_loss = sum(layer_losses) / len(layer_losses)
            last_lr = lr_scheduler.get_last_lr()[0]
            if config.training.wandb_enabled:
                accelerator.log({"loss": avg_loss, "lr": last_lr}, step=step)
            progress_bar.set_postfix({"loss": avg_loss, "lr": last_lr})
            progress_bar.update(config.training.log_every_step)

        if step % 500 == 0:
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-hidden-to-hidden")
        save_model(
            accelerator.unwrap_model(student_model), hf_model_config, tokenizer, checkpoint_dir
        )
        accelerator.print(f"Checkpoint saved to: {checkpoint_dir}")

    accelerator.wait_for_everyone()
    return step, student_model, teacher_model, optimizer


def train_token_mixer(
    step, accelerator, train_dataloader, student_model, teacher_model,
    tokenizer, hf_model_config, config, output_dir,
):
    num_training_steps = len(train_dataloader)
    accelerator.print(f"[Token Mixer] Total training steps: {num_training_steps}")

    progress_bar = tqdm(
        total=num_training_steps // accelerator.num_processes,
        desc="[Matrix Mixing]",
        unit="step",
        colour="RED",
        disable=not accelerator.is_local_main_process,
    )

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.training.stage_1.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    lr_scheduler = get_scheduler(
        config.training.stage_1.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_training_steps // 10,
        scheduler_specific_kwargs={
            "num_decay_steps": num_training_steps // 10,
            "num_stable_steps": int(num_training_steps * 0.8),
            "min_lr_ratio": 0.0001,
        },
    )

    student_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        student_model, optimizer, lr_scheduler, train_dataloader
    )

    if teacher_model.device != accelerator.device:
        teacher_model = teacher_model.to(accelerator.device)

    for batch in train_dataloader:
        inputs = batch["input_ids"].to(accelerator.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(accelerator.device)

        with torch.no_grad():
            teacher_outputs = teacher_model(
                inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        with accelerator.accumulate(student_model):
            total_loss = torch.zeros((), device=accelerator.device)
            layer_losses = []

            for layer_idx, student_layer in enumerate(get_model_layers(student_model)):
                transfer_matrix = student_layer(
                    hidden_states=teacher_outputs.hidden_states[layer_idx].to(torch.bfloat16),
                    output_attentions=True,
                )[0]
                attn_matrix = teacher_outputs.attn_hidden_states[layer_idx]
                assert transfer_matrix.size() == attn_matrix.size(), (
                    f"Shape mismatch: student {transfer_matrix.size()} vs "
                    f"teacher {attn_matrix.size()}"
                )
                layer_loss = torch.nn.functional.mse_loss(transfer_matrix, attn_matrix)
                total_loss = total_loss + layer_loss
                layer_losses.append(layer_loss.item())

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        step += 1

        if (
            step % config.training.log_every_step == 0
            and config.training.log_every_step > 0
        ):
            avg_loss = sum(layer_losses) / len(layer_losses)
            last_lr = lr_scheduler.get_last_lr()[0]
            if config.training.wandb_enabled:
                accelerator.log({"loss": avg_loss, "lr": last_lr}, step=step)
            progress_bar.set_postfix({"loss": avg_loss, "lr": last_lr})
            progress_bar.update(config.training.log_every_step)

        if step % 1000 == 0:
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-matrix-mixing")
        save_model(
            accelerator.unwrap_model(student_model), hf_model_config, tokenizer, checkpoint_dir
        )
        accelerator.print(f"Checkpoint saved to: {checkpoint_dir}")

    return step, student_model, teacher_model, optimizer


def _build_lr_scheduler(scheduler_type, optimizer, num_training_steps):
    if scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 100,
        )
    if scheduler_type == "linear":
        return get_scheduler(
            "linear",
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_training_steps // 10,
        )
    if scheduler_type == "cosine":
        return get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_training_steps // 50,
        )
    if scheduler_type == "cosine_with_min_lr":
        return get_scheduler(
            "cosine_with_min_lr",
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_training_steps // 20,
            scheduler_specific_kwargs={"min_lr": 5e-6},
        )
    if scheduler_type in ("warmup_stable_decay", "cosine_with_warmup"):
        # cosine_with_warmup is an alias kept for config compatibility
        sched_name = "cosine" if scheduler_type == "cosine_with_warmup" else scheduler_type
        return get_scheduler(
            sched_name,
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_training_steps // 5,
            scheduler_specific_kwargs={
                "num_decay_steps": num_training_steps // 10,
                "num_stable_steps": int(num_training_steps * 0.8),
                "min_lr_ratio": 0.0001,
            },
        )
    raise ValueError(f"Unknown lr_scheduler type: {scheduler_type!r}")


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def _setup_accelerator_and_output(config):
    loggers = []
    if config.training.get("wandb_project") not in (None, ""):
        loggers.append("wandb")

    gradient_accumulation_steps = config.training.get("gradient_accumulation_steps", 1)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with=loggers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    return accelerator


def run_training(config_path: str):
    config = load_config(config_path)
    config.model.apply_rope = config.model.get("apply_rope", False)
    config.training.setdefault("log_every_step", 1)

    accelerator = _setup_accelerator_and_output(config)

    # Generate run_dir on rank 0 and broadcast so all nodes use the same path.
    if accelerator.is_main_process:
        run_dir = "run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    else:
        run_dir = None
    if torch.distributed.is_initialized():
        run_dir_container = [run_dir]
        torch.distributed.broadcast_object_list(run_dir_container, src=0)
        run_dir = run_dir_container[0]
    output_dir = os.path.join(config.training.output_dir, run_dir)

    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.print(f"Output directory: {output_dir}")

    tokenized_datasets, tokenizer = preprocess(config, accelerator)
    if config.model.context_length != tokenizer.model_max_length:
        tokenizer.model_max_length = config.model.context_length

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model, teacher_model, hf_model_config, num_params, num_params_human = (
        initialize_model_and_teacher(config, accelerator, tokenizer, output_dir)
    )

    frozen, _ = count_frozen_parameters(model)
    accelerator.print(f"Frozen parameters: {frozen:_}")

    # Build one dataloader per stage in order: stage_1, stage_2, stage_3
    STAGE_KEYS = ["stage_1", "stage_2", "stage_3"]
    STAGE_FNS = [train_token_mixer, train_hidden_alignment, train_end_to_end]

    splits = [float(s) for s in config.training.splits.split("|")]
    len_dataset = len(tokenized_datasets)
    split_ends = list(itertools.accumulate(int(s * len_dataset) for s in splits))

    stage_dataloaders = []
    start = 0
    for key, stage_fn, end in zip(STAGE_KEYS, STAGE_FNS, split_ends):
        end = min(end, len_dataset)
        subset = tokenized_datasets.select(range(start, end))
        accelerator.print(f"{key}: {len(subset)} examples")
        batch_size = config.training[key].batch_size
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        stage_dataloaders.append((dl, stage_fn))
        start = end

    # Save config and tokenizer before training (main process only)
    if accelerator.is_main_process:
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)
        tokenizer.save_pretrained(output_dir)

    if config.training.wandb_enabled:
        wandb_project = config.training.get("wandb_project", "mohawk")
        accelerator.print(f"Enabling wandb logging for project: {wandb_project}")
        config_dict = OmegaConf.to_container(OmegaConf.select(config, "model"))
        config_dict["num_params"] = num_params
        config_dict["num_params_human"] = num_params_human
        accelerator.init_trackers(
            project_name=wandb_project,
            config=config_dict,
            init_kwargs={"wandb": {"name": run_dir}},
        )

    model.train()
    num_steps = 0
    for train_dataloader, training_fn in stage_dataloaders:
        if len(train_dataloader) == 0:
            continue

        accelerator.print(f"Process {accelerator.process_index}: starting {training_fn.__name__}")
        num_steps, model, teacher_model, optimizer = training_fn(
            num_steps, accelerator, train_dataloader, model, teacher_model,
            tokenizer, hf_model_config, config, output_dir,
        )
        accelerator.print(f"Process {accelerator.process_index}: finished {training_fn.__name__}")

        accelerator.wait_for_everyone()

        model = accelerator.unwrap_model(model)
        for param in model.parameters():
            param.requires_grad = True

        del optimizer
        if teacher_model is not None:
            teacher_model = accelerator.unwrap_model(teacher_model).cpu()
        gc.collect()
        torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

    accelerator.end_training()

    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-last")
        save_model(model, hf_model_config, tokenizer, checkpoint_dir)
        accelerator.print(f"Final model saved to: {checkpoint_dir}")


def run_lr_finder(config_path: str):
    config = load_config(config_path)
    config.model.apply_rope = config.model.get("apply_rope", False)
    config.training.setdefault("log_every_step", 1)

    run_dir = "lr_finder_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = os.path.join(config.training.output_dir, run_dir)

    accelerator = _setup_accelerator_and_output(config)

    # LR finder only runs on the local main process
    if not accelerator.is_local_main_process:
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    os.makedirs(output_dir, exist_ok=True)
    accelerator.print(f"Output directory: {output_dir}")

    tokenized_datasets, tokenizer = preprocess(config, accelerator)
    if config.model.context_length != tokenizer.model_max_length:
        tokenizer.model_max_length = config.model.context_length

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model, teacher_model, hf_model_config, num_params, num_params_human = (
        initialize_model_and_teacher(config, accelerator, tokenizer, output_dir)
    )

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)
    tokenizer.save_pretrained(output_dir)

    if config.training.wandb_enabled:
        wandb_project = config.training.get("wandb_project", "mohawk")
        config_dict = OmegaConf.to_container(OmegaConf.select(config, "model"))
        config_dict.update({"num_params": num_params, "num_params_human": num_params_human})
        accelerator.init_trackers(
            project_name=wandb_project,
            config=config_dict,
            init_kwargs={"wandb": {"name": run_dir}},
        )

    lr_cfg = config.training.get("lr_finder", {})
    start_lr = lr_cfg.get("start_lr", 1e-7)
    end_lr = lr_cfg.get("end_lr", 1.0)
    num_steps = lr_cfg.get("num_steps", 200)
    beta = lr_cfg.get("beta", 0.98)
    loss_increase_factor = lr_cfg.get("loss_increase_factor", 4.0)
    log_every_step = lr_cfg.get("log_every_step", 1)
    skip_start = max(lr_cfg.get("skip_start", 10), 0)
    skip_end = max(lr_cfg.get("skip_end", 5), 0)

    # Resolve batch size: prefer explicit lr_finder.batch_size, fall back to stage configs
    batch_size = lr_cfg.get("batch_size") or config.training.get("batch_size")
    if batch_size is None:
        for key in ("stage_3", "stage_1"):
            if key in config.training:
                batch_size = config.training[key].get("batch_size")
                break
    batch_size = batch_size or 1

    train_dataloader = DataLoader(
        tokenized_datasets, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )
    optimizer = get_optimizer("adamw", model, lr=start_lr)
    model = model.to(accelerator.device)
    if teacher_model is not None:
        teacher_model = teacher_model.to(accelerator.device)

    progress_bar = tqdm(total=num_steps, desc="[LR Finder]", unit="step", colour="BLUE")

    lr_multiplier = (end_lr / start_lr) ** (1 / max(num_steps - 1, 1))
    data_iter = iter(train_dataloader)
    smoothed_loss = None
    best_loss = float("inf")
    results = []

    model.train()
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        lr = start_lr * (lr_multiplier ** step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs = batch["input_ids"].to(accelerator.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(accelerator.device)

        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(inputs, attention_mask=attention_mask).logits

        with torch.autocast(device_type=accelerator.device.type, dtype=torch.bfloat16):
            output = model(inputs, labels=inputs, attention_mask=attention_mask)
            loss = output.loss

        if teacher_model is not None:
            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(output.logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = loss + kl_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        smoothed_loss = (
            loss_val if smoothed_loss is None
            else beta * smoothed_loss + (1 - beta) * loss_val
        )
        debiased = smoothed_loss / (1 - beta ** (step + 1))
        best_loss = min(best_loss, debiased)

        results.append({"step": step, "lr": lr, "loss": loss_val, "smoothed_loss": debiased})

        if step % log_every_step == 0 and step > 0:
            progress_bar.set_postfix({"loss": loss_val, "lr": lr})
            progress_bar.update(log_every_step)
            if config.training.wandb_enabled:
                accelerator.log(
                    {"lr_finder/loss": loss_val, "lr_finder/smoothed_loss": debiased, "lr_finder/lr": lr},
                    step=step,
                )

        if step > 0 and debiased > loss_increase_factor * best_loss:
            accelerator.print(f"Early stop at step {step}: loss diverged.")
            break

    # Save results
    results_path = os.path.join(output_dir, "lr_finder.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "lr", "loss", "smoothed_loss"])
        writer.writeheader()
        writer.writerows(results)

    best_entry = min(results, key=lambda r: r["smoothed_loss"])
    summary = {"best_smoothed_lr": best_entry["lr"], "best_smoothed_loss": best_entry["smoothed_loss"]}

    if len(results) > skip_start + skip_end + 2:
        lrs = np.array([r["lr"] for r in results])
        losses = np.array([r["smoothed_loss"] for r in results])
        lrs_use = lrs[skip_start:(-skip_end if skip_end else None)]
        losses_use = losses[skip_start:(-skip_end if skip_end else None)]

        gradients = np.gradient(losses_use)
        steepest_lr = float(lrs_use[int(np.argmin(gradients))])

        dx = np.diff(np.log(lrs_use))
        dy = np.diff(losses_use)
        angles = np.unwrap(np.arctan2(dy, dx))
        curvature = np.gradient(angles)
        elbow_lr = float(lrs_use[int(np.argmax(np.abs(curvature)))])

        summary.update({
            "steepest_lr": steepest_lr,
            "elbow_lr": elbow_lr,
            "suggested_lr": elbow_lr / 10.0,
        })

    with open(os.path.join(output_dir, "lr_finder_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    accelerator.print(
        f"LR finder done. Results: {results_path} "
        f"(suggested_lr={summary.get('suggested_lr', summary['best_smoothed_lr']):.2e})"
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_only(config_path: str):
    config = load_config(config_path)
    accelerator = Accelerator()
    preprocess(config, accelerator, ask_for_overwrite=True)


def preprocess(config, accelerator=None, ask_for_overwrite=False):
    """Tokenise the dataset and cache results. Only the main process does the work."""
    hugging_face_id = config.dataset.hugging_face_id
    if isinstance(hugging_face_id, str):
        hugging_face_id = (hugging_face_id,)

    model_name = config.training.model_name
    output_path = config.dataset.output_path

    tokenizer_path = os.path.join(output_path, f"preprocessed/{model_name}/tokenizer")
    tokenized_data_path = os.path.join(output_path, f"preprocessed/{model_name}/tokenized_datasets")

    # Optionally overwrite existing cache
    if os.path.exists(tokenizer_path) and os.path.exists(tokenized_data_path) and ask_for_overwrite:
        if input("Preprocessed data already exists. Overwrite? [y/n]: ").lower() == "y":
            accelerator.print("Deleting existing preprocessed data...")
            shutil.rmtree(tokenizer_path)
            shutil.rmtree(tokenized_data_path)

    # Return cached data if available
    if os.path.exists(tokenizer_path) and os.path.exists(tokenized_data_path):
        accelerator.print("Loading preprocessed data from cache...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return load_from_disk(tokenized_data_path), tokenizer

    # Only global rank 0 downloads and tokenises; all others wait for the cache to appear
    if accelerator.is_main_process:
        accelerator.print(f"Loading dataset: {hugging_face_id}")
        raw_datasets = load_dataset(
            *hugging_face_id,
            split=config.dataset.split,
            trust_remote_code=True,
            cache_dir=os.getenv("HF_HOME", os.getenv("HF_DATASETS_CACHE")),
        )

        tokenizer_id = config.tokenizer.pretrained_id
        accelerator.print(f"Loading tokenizer: {tokenizer_id}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer.pad_token is None:
            assert tokenizer.eos_token is not None, "Tokenizer has no eos_token."
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(tokenizer_path)

        block_size = config.model.context_length

        def tokenize_function(example):
            return tokenizer(example["text"])

        def group_texts(examples):
            concatenated = {k: list(chain(*examples[k])) for k in examples}
            total = len(concatenated[next(iter(concatenated))])
            total = (total // block_size) * block_size
            return {
                k: [t[i: i + block_size] for i in range(0, total, block_size)]
                for k, t in concatenated.items()
            }

        accelerator.print("Tokenizing dataset...")
        num_proc = multiprocessing.cpu_count()
        tokenized = raw_datasets.map(
            tokenize_function, batched=True,
            remove_columns=raw_datasets.column_names, num_proc=num_proc,
        )
        tokenized = tokenized.map(
            group_texts, batched=True, num_proc=num_proc,
            desc=f"Grouping into chunks of {block_size}",
        )
        tokenized.save_to_disk(tokenized_data_path)
        accelerator.print(f"Preprocessed data saved to: {tokenized_data_path}")

        # Sanity check
        sample = tokenized[0]
        assert list(sample.keys()) in (["input_ids"], ["input_ids", "attention_mask"]), \
            f"Unexpected tokenized keys: {list(sample.keys())}"

        tokenized_datasets = tokenized
    else:
        while not os.path.exists(tokenized_data_path):
            time.sleep(1)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        tokenized_datasets = load_from_disk(tokenized_data_path)

    accelerator.wait_for_everyone()
    return tokenized_datasets, tokenizer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py [preprocess|lr_finder] <config.yaml>")
        sys.exit(1)

    if sys.argv[1] == "preprocess":
        preprocess_only(sys.argv[2])
    elif sys.argv[1] == "lr_finder":
        run_lr_finder(sys.argv[2])
    else:
        run_training(sys.argv[1])
