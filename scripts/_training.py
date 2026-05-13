"""Shared SFT + LoRA training loop. Used by Stage 3 (teacher) and Stage 6 (student).

Both stages do the exact same thing: load a Gemma model with Unsloth + QLoRA,
fine-tune via TRL's SFTTrainer on a JSONL of {input, output} pairs formatted
with the chat template, save the LoRA adapter, run deterministic sample
generations for a vibe-check, and export a merged GGUF. Differences between
the two stages (base model, LoRA rank, epochs, output dir, GGUF quant) are
config; the loop is shared.
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Unsloth must be imported BEFORE transformers/trl so its monkey-patches land.
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import TrainingArguments  # noqa: E402
from trl import SFTTrainer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import SAMPLE_INPUTS, build_messages, load_jsonl  # noqa: E402


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class TrainingConfig:
    # Required
    model_name: str
    output_dir: Path
    train_file: Path
    eval_file: Path

    # Training schedule
    max_seq_length: int = 1024
    epochs: float = 3.0
    batch_size: int = 4
    grad_accum: int = 4
    eval_batch_size: int = 8  # decoupled from train batch — large eval batches OOM on fp32 logit conversion
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_steps: int = -1

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    load_in_4bit: bool = True

    # Logging / eval / save
    eval_steps: int = 50
    save_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = 42

    # Export
    gguf_quant: str = "q3_k_m"
    skip_gguf: bool = False

    # Flow control
    resume: bool = False
    force: bool = False


def _format_for_training(example: dict, tokenizer) -> dict:
    msgs = build_messages(example["input"], example["output"])
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def _consolidate_gguf_output(gguf_dir: Path) -> None:
    """Unsloth's save_pretrained_gguf(gguf_dir, ...) writes the intermediate
    merged safetensors INTO gguf_dir and drops the actual *.gguf files into
    a sibling `<gguf_dir>_gguf/` directory. Consolidate so gguf_dir ends up
    holding only the final *.gguf artifacts (which is what downstream tools
    like 04_eval.py and the viewer scan for)."""
    sibling = gguf_dir.parent / f"{gguf_dir.name}_gguf"
    if not sibling.exists() or not sibling.is_dir():
        return  # Nothing to consolidate — older Unsloth version or already cleaned.

    gguf_files = [f for f in sibling.iterdir() if f.is_file() and f.suffix.lower() == ".gguf"]
    if not gguf_files:
        return

    # Clear the intermediate safetensors/config out of gguf_dir first.
    for stale in list(gguf_dir.iterdir()):
        if stale.suffix.lower() == ".gguf":
            continue  # paranoia: never delete a .gguf that's already there
        try:
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        except OSError as e:
            logging.warning("Could not remove intermediate %s: %s", stale, e)

    # Move every *.gguf into gguf_dir.
    for f in gguf_files:
        target = gguf_dir / f.name
        if target.exists():
            target.unlink()
        shutil.move(str(f), str(target))

    # Drop the now-empty sibling directory (plus any non-gguf leftovers like Modelfile).
    try:
        shutil.rmtree(sibling)
    except OSError as e:
        logging.warning("Could not remove %s: %s", sibling, e)

    logging.info("Consolidated %d GGUF file(s) into %s", len(gguf_files), gguf_dir)


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return ckpts[-1] if ckpts else None


def run_training(cfg: TrainingConfig) -> int:
    """Run a full training pass. Returns the process exit code (0 = success)."""
    adapters_dir = cfg.output_dir / "adapters"
    gguf_dir = cfg.output_dir / "gguf"
    checkpoints_dir = cfg.output_dir / "checkpoints"

    adapter_marker = adapters_dir / "adapter_config.json"
    if adapter_marker.exists() and not cfg.force and not cfg.resume:
        logging.info("Adapter already exists at %s. Pass force=True to retrain.", adapters_dir)
        logging.info("Skipping training. Run scripts/04_eval.py to evaluate it.")
        return 0

    if not cfg.train_file.exists() or not cfg.eval_file.exists():
        logging.error("Train/eval data missing: %s, %s", cfg.train_file, cfg.eval_file)
        return 2

    logging.info("Training %s", cfg.model_name)
    logging.info("Output: %s", cfg.output_dir)
    logging.info("Train: %s   Eval: %s", cfg.train_file, cfg.eval_file)
    logging.info("Config: %s", vars(cfg))

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        logging.info("CUDA device: %s (%.1f GB)", dev.name, dev.total_memory / 1e9)
    else:
        logging.warning("CUDA not available — training will be unusably slow on CPU.")

    # ---- model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    # ---- datasets
    raw_train = load_jsonl(cfg.train_file)
    raw_eval = load_jsonl(cfg.eval_file)
    for r in raw_train + raw_eval:
        r.pop("_source", None)
    logging.info("Loaded %d train / %d eval examples", len(raw_train), len(raw_eval))

    train_ds = Dataset.from_list(raw_train).map(
        lambda ex: _format_for_training(ex, tokenizer),
        remove_columns=["input", "output"],
    )
    eval_ds = Dataset.from_list(raw_eval).map(
        lambda ex: _format_for_training(ex, tokenizer),
        remove_columns=["input", "output"],
    )

    logging.info("Sample formatted training text:\n%s", train_ds[0]["text"])

    use_bf16 = is_bfloat16_supported()
    logging.info("Precision: %s", "bf16" if use_bf16 else "fp16")

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=cfg.batch_size,
        # Eval batch is decoupled from train batch. Large eval batches blow up
        # because Trainer materializes logits in fp32 (vocab is ~256k on Gemma 3).
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_8bit",
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        # Only keep loss during eval — skip returning logits (256k vocab × batch
        # × seq would OOM in fp32 conversion, and we don't need them for eval loss).
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to="none",
        dataloader_num_workers=0,        # Windows + Unsloth: workers=0 is safest
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=False,
        args=training_args,
    )

    resume_path: str | bool = False
    if cfg.resume:
        latest = _latest_checkpoint(checkpoints_dir)
        if latest is not None:
            logging.info("Resuming from %s", latest)
            resume_path = str(latest)
        else:
            logging.warning("resume=True but no checkpoint found. Starting fresh.")

    train_result = trainer.train(resume_from_checkpoint=resume_path)
    logging.info("Final train loss: %.4f", train_result.training_loss)

    # ---- save LoRA adapter
    adapters_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapters_dir))
    tokenizer.save_pretrained(str(adapters_dir))
    logging.info("Saved LoRA adapter -> %s", adapters_dir)

    eval_metrics = trainer.evaluate()
    logging.info("Final eval metrics: %s", eval_metrics)

    # ---- sample generations (deterministic) for a vibe-check before eval
    FastLanguageModel.for_inference(model)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)  # unwrap Gemma 3/4 processor
    pad_id = text_tok.pad_token_id or text_tok.eos_token_id
    # Generation needs LEFT padding so EOS isn't on the wrong side.
    prev_padding_side = text_tok.padding_side
    text_tok.padding_side = "left"
    if text_tok.pad_token_id is None:
        text_tok.pad_token = text_tok.eos_token
    logging.info("=" * 60)
    logging.info("Sample generations (greedy, max_new_tokens=256, batched)")
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(inp, output_obj=None),
            tokenize=False, add_generation_prompt=True,
        )
        for inp in SAMPLE_INPUTS
    ]
    enc = text_tok(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=cfg.max_seq_length,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=pad_id,
        )
    input_len = enc["input_ids"].shape[1]
    for inp, row in zip(SAMPLE_INPUTS, out):
        gen = text_tok.decode(row[input_len:], skip_special_tokens=True)
        logging.info("INPUT : %s", inp)
        logging.info("OUTPUT: %s", gen.strip())
        logging.info("-" * 40)
    text_tok.padding_side = prev_padding_side

    # ---- merged GGUF export
    if cfg.skip_gguf:
        logging.info("skip_gguf=True; skipping GGUF export.")
    else:
        gguf_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Exporting merged GGUF (%s) -> %s", cfg.gguf_quant, gguf_dir)
        try:
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer,
                quantization_method=cfg.gguf_quant,
            )
            logging.info("GGUF export complete.")
            _consolidate_gguf_output(gguf_dir)
        except Exception as e:  # noqa: BLE001
            logging.error("GGUF export failed: %s", e)
            logging.error("LoRA adapter is still saved at %s. Retry GGUF later or "
                          "convert manually with llama.cpp.", adapters_dir)
            return 1

    logging.info("Training complete.")
    return 0
