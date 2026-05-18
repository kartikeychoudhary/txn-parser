// flows.js — curated pipeline flow catalogue.
// Each flow declares its script path and the args we expose in the UI.
// buildArgv translates UI form values into a Python argv array.
//
// Arg field reference:
//   name        — kebab-case flag name (becomes "--<name>")
//   type        — "int" | "float" | "string" | "flag" | "choice" | "positional" | "config"
//   default     — initial value shown in the UI
//   required    — bool; if missing+empty, buildArgv throws
//   choices     — required for type:"choice"; UI renders as <select>
//   group       — "basic" (default) or "advanced"; UI may collapse advanced
//   help        — short description shown next to the field

export const FLOWS = [
  {
    id: "stage_1_generate_raw",
    label: "Stage 1 — Generate raw dataset",
    description: "DeepSeek synthesizes raw transaction examples into data/raw/.",
    script: "scripts/01_generate_dataset.py",
    args: [
      { name: "batches",     type: "int",    default: 25,  help: "How many consecutive batches to run starting from --start-batch (1..25). Each batch ~50 examples." },
      { name: "start-batch", type: "int",    default: 1,   help: "1-indexed batch number to start from. Use to resume after a crash." },
      { name: "model",       type: "string", default: "deepseek-v4-flash", help: "DeepSeek model ID. Try deepseek-reasoner for harder mixes." },
      { name: "temperature", type: "float",  default: 1.0, help: "Sampling temperature. Higher = more variety, lower = more repetition." },
      { name: "max-tokens",  type: "int",    default: 8000, group: "advanced", help: "Completion cap per API call. Raising it lets a batch return more rows." },
      { name: "max-retries", type: "int",    default: 5,    group: "advanced", help: "Transient errors retried per call before giving up." },
      { name: "timeout",     type: "int",    default: 300,  group: "advanced", help: "Per-call HTTP timeout in seconds." },
      { name: "base-url",    type: "string", default: "https://api.deepseek.com", group: "advanced", help: "Override for self-hosted or proxied DeepSeek endpoints." },
      { name: "output-dir",  type: "string", default: "data/raw", group: "advanced", help: "Where batch_NN.jsonl files are written." },
      { name: "force",       type: "flag",   default: false, help: "Regenerate a batch even if its file already exists (default = skip existing)." },
    ],
  },
  {
    id: "stage_2_clean_dataset",
    label: "Stage 2 — Clean + dedupe + split",
    description: "Validate raw batches, dedupe, and split into train/eval.",
    script: "scripts/02_clean_dataset.py",
    args: [
      { name: "eval-frac",  type: "float",  default: 0.1, help: "Fraction of cleaned rows held out as eval set (0.0–1.0)." },
      { name: "seed",       type: "int",    default: 42,  help: "Shuffle seed — same value = same train/eval split." },
      { name: "input-dir",  type: "string", default: "data/raw",   group: "advanced", help: "Where to read raw batch_*.jsonl from." },
      { name: "output-dir", type: "string", default: "data/clean", group: "advanced", help: "Where to write train.jsonl + eval.jsonl." },
    ],
  },
  {
    id: "stage_3_train_teacher",
    label: "Stage 3 — Train teacher",
    description: "QLoRA fine-tune of Gemma 4 E2B on data/clean/.",
    script: "scripts/03_train_teacher.py",
    args: [
      { name: "model",            type: "string", default: "unsloth/gemma-4-E2B-it", help: "HF base model ID. Don't change unless you know the chat template matches." },
      { name: "epochs",           type: "float",  default: 3.0, help: "Full passes over data/clean/train.jsonl." },
      { name: "batch-size",       type: "int",    default: 4,   help: "Per-device train batch. Effective batch = batch-size × grad-accum." },
      { name: "grad-accum",       type: "int",    default: 4,   help: "Gradient accumulation steps. Raise to grow effective batch on small VRAM." },
      { name: "eval-batch-size",  type: "int",    default: 8,   help: "Keep small. fp32 logits over 256k vocab OOM at eval time." },
      { name: "max-seq-length",   type: "int",    default: 1024, help: "Token window for prompt+answer. Bump if your rows truncate." },
      { name: "max-steps",        type: "int",    default: -1, help: "-1 = full epochs. Set e.g. 20 for a smoke test." },
      { name: "lr",               type: "float",  default: 2e-4, group: "advanced", help: "Peak learning rate after warmup. 2e-4 is QLoRA default." },
      { name: "warmup-ratio",     type: "float",  default: 0.03, group: "advanced", help: "Fraction of total steps used to ramp the LR." },
      { name: "lora-r",           type: "int",    default: 16,   group: "advanced", help: "LoRA rank. Higher = more capacity + more VRAM." },
      { name: "lora-alpha",       type: "int",    default: 32,   group: "advanced", help: "LoRA scaling. Typically 2× lora-r." },
      { name: "lora-dropout",     type: "float",  default: 0.0,  group: "advanced", help: "Dropout inside LoRA layers." },
      { name: "eval-steps",       type: "int",    default: 50,   group: "advanced", help: "Run eval every N optimizer steps." },
      { name: "save-steps",       type: "int",    default: 200,  group: "advanced", help: "Checkpoint every N steps. Last 2 kept." },
      { name: "logging-steps",    type: "int",    default: 10,   group: "advanced", help: "How often loss/grad-norm prints." },
      { name: "seed",             type: "int",    default: 42,   group: "advanced", help: "RNG seed for shuffling + LoRA init." },
      { name: "gguf-quant",       type: "string", default: "q3_k_m", group: "advanced", help: "Quant level for the merged GGUF: q3_k_m, q4_k_m, q5_k_m, q8_0, bf16." },
      { name: "skip-gguf",        type: "flag",   default: false, help: "Train + save adapter, but skip the slow GGUF export." },
      { name: "resume",           type: "flag",   default: false, help: "Continue from the latest checkpoint in models/teacher/checkpoints/." },
      { name: "force",            type: "flag",   default: false, help: "Retrain even if models/teacher/adapters/ already exists." },
    ],
  },
  {
    id: "stage_4_eval",
    label: "Stage 4 — Evaluate",
    description: "Run a model against data/clean/eval.jsonl.",
    script: "scripts/04_eval.py",
    args: [
      { name: "model",         type: "string", default: "models/teacher/gguf", required: true, help: ".gguf file, directory with one, or LoRA adapter dir. Backend picked automatically." },
      { name: "limit",         type: "int",    default: 0, help: "Eval only first N examples. 0 = all." },
      { name: "name",          type: "string", default: "", help: "Override the eval_results/<name>.jsonl filename stem." },
      { name: "max-tokens",    type: "int",    default: 512, help: "Generation cap per example." },
      { name: "batch-size",    type: "int",    default: 16, help: "Examples per forward pass. Transformers backend only — GGUF is sequential." },
      { name: "n-ctx",         type: "int",    default: 2048,  group: "advanced", help: "Context window for GGUF backend." },
      { name: "n-gpu-layers",  type: "int",    default: -1,    group: "advanced", help: "-1 = all on GPU, 0 = CPU. GGUF only." },
      { name: "eval-file",     type: "string", default: "data/clean/eval.jsonl", group: "advanced", help: "Custom eval JSONL path." },
      { name: "no-grammar",    type: "flag",   default: false, help: "Disable GBNF grammar-constrained decoding. Use for baseline comparisons; otherwise on by default for GGUF." },
    ],
  },
  {
    id: "stage_5_legacy",
    label: "Stage 5 (legacy) — Teacher-labeled distillation",
    description: "Teacher fp16 labels DeepSeek-generated synthetic inputs.",
    script: "scripts/05_generate_distillation_data.py",
    args: [
      { name: "phase",           type: "choice", choices: ["all", "inputs", "label", "eval"], default: "all", help: "all = inputs→label→eval. inputs = DeepSeek synth. label = teacher fp16 labels. eval = copy clean eval over." },
      { name: "n-inputs",        type: "int",    default: 30000, help: "Target unique synthetic inputs to generate in phase 1." },
      { name: "inputs-per-call", type: "int",    default: 200,   group: "advanced", help: "Rows DeepSeek returns per API call." },
      { name: "backend",         type: "choice", choices: ["transformers", "gguf"], default: "transformers", help: "Teacher inference backend. transformers = fp16 (default, higher quality). gguf = Q3_K_M sequential (lower VRAM)." },
      { name: "batch-size",      type: "int",    default: 16, help: "Phase 2 transformers batch. A100: 32–64. 5060 Ti: 8–16." },
      { name: "max-new-tokens",  type: "int",    default: 384, help: "Generation cap per label. Drop to 256 if outputs fit — saves wall time." },
      { name: "max-seq-length",  type: "int",    default: 1024, group: "advanced", help: "Teacher prompt+answer window." },
      { name: "limit",           type: "int",    default: 0, help: "Label only the first N pending inputs. 0 = all. Phase 2 only." },
      { name: "model",           type: "string", default: "deepseek-v4-flash", group: "advanced", help: "Phase 1 DeepSeek model." },
      { name: "max-tokens",      type: "int",    default: 8000, group: "advanced", help: "Phase 1 DeepSeek completion cap." },
      { name: "temperature",     type: "float",  default: 1.0,  group: "advanced", help: "Phase 1 sampling temperature." },
      { name: "max-retries",     type: "int",    default: 5,    group: "advanced", help: "Phase 1 transient-error retries." },
      { name: "timeout",         type: "int",    default: 300,  group: "advanced", help: "Phase 1 per-call HTTP timeout (s)." },
      { name: "base-url",        type: "string", default: "https://api.deepseek.com", group: "advanced", help: "DeepSeek API base URL override." },
      { name: "gguf-path",       type: "string", default: "", group: "advanced", help: "Teacher .gguf path. GGUF backend only." },
      { name: "n-gpu-layers",    type: "int",    default: -1,   group: "advanced", help: "GGUF only. -1 = all on GPU." },
      { name: "n-ctx",           type: "int",    default: 2048, group: "advanced", help: "GGUF only. Context window." },
      { name: "n-batch",         type: "int",    default: 512,  group: "advanced", help: "GGUF only. llama.cpp logical batch." },
      { name: "no-mmap",         type: "flag",   default: false, group: "advanced", help: "GGUF only. Disable mmap (load fully into RAM)." },
      { name: "mlock",           type: "flag",   default: false, group: "advanced", help: "GGUF only. Lock model pages in RAM." },
      { name: "retry-failed",            type: "flag", default: false, help: "Re-attempt every row currently in data/distill/failed.jsonl." },
      { name: "retry-validation-failed", type: "flag", default: false, help: "Re-attempt only rows whose failure was a validator (not JSON-parse) error, with a stricter prompt." },
      { name: "force-eval-copy",         type: "flag", default: false, group: "advanced", help: "Overwrite data/distill/eval.jsonl in the eval phase." },
      { name: "force",                   type: "flag", default: false, help: "DESTRUCTIVE: regenerate from scratch. phase=inputs deletes inputs_raw.jsonl; phase=label deletes train.jsonl + failed.jsonl." },
    ],
  },
  {
    id: "stage_5_multiprovider",
    label: "Stage 5 — Multi-provider distillation",
    description: "DeepSeek + Gemini act as labelers via providers config. The chosen JSON config controls how many inputs to generate (target_inputs) and how many parallel workers — UI fields below do NOT control that.",
    script: "scripts/05_generate_distillation_data.py",
    args: [
      { name: "phase",           type: "choice", choices: ["inputs", "label"], default: "inputs", help: "inputs = synthesize new inputs via providers (writes data/distill/inputs_raw.jsonl). label = generate labels for existing inputs (writes train.jsonl)." },
      { name: "provider-config", type: "config", default: "prod_100k_providers.json", required: true, help: "Picks the JSON below. target_inputs, batch_size, threads, rate limits all live in this file — open the editor to change them." },
      { name: "multi-provider",  type: "flag",   default: true, help: "Required for this flow. Tells the script to use the provider-config orchestration path." },
      { name: "limit",           type: "int",    default: 0, help: "Label phase only: cap inputs processed in this run. 0 = all pending. IGNORED for the inputs phase — for inputs, edit target_inputs in the config." },
      { name: "dry-run-quota",   type: "flag",   default: false, help: "Validate the config and print the quota plan without making any API calls." },
      { name: "force",           type: "flag",   default: false, help: "DESTRUCTIVE: regenerate from scratch. phase=inputs deletes inputs_raw.jsonl; phase=label deletes train.jsonl + failed.jsonl." },
    ],
  },
  {
    id: "stage_6_train_student",
    label: "Stage 6 — Train student",
    description: "QLoRA fine-tune of Gemma 3 270M on data/distill/train.jsonl.",
    script: "scripts/06_train_student.py",
    args: [
      { name: "model",            type: "string", default: "unsloth/gemma-3-270m-it", help: "HF base model ID. This is the shipping student — keep at default unless deliberately swapping bases." },
      { name: "epochs",           type: "float",  default: 2.0, help: "More distill data → fewer epochs to avoid overfit." },
      { name: "batch-size",       type: "int",    default: 8,   help: "Per-device train batch. Student is tiny — push high on big GPUs." },
      { name: "grad-accum",       type: "int",    default: 2,   help: "Gradient accumulation steps." },
      { name: "eval-batch-size",  type: "int",    default: 8,   help: "Keep small — Gemma 3 has a 256k vocab and fp32 logits OOM fast." },
      { name: "max-seq-length",   type: "int",    default: 1024, help: "Token window." },
      { name: "max-steps",        type: "int",    default: -1, help: "-1 = full epochs. Set 20 for a smoke test of the whole flow." },
      { name: "lr",               type: "float",  default: 2e-4, group: "advanced", help: "Peak LR." },
      { name: "warmup-ratio",     type: "float",  default: 0.03, group: "advanced", help: "LR warmup fraction." },
      { name: "lora-r",           type: "int",    default: 32,   group: "advanced", help: "LoRA rank. Higher than teacher because base is much smaller." },
      { name: "lora-alpha",       type: "int",    default: 64,   group: "advanced", help: "LoRA scaling." },
      { name: "lora-dropout",     type: "float",  default: 0.0,  group: "advanced", help: "LoRA dropout." },
      { name: "eval-steps",       type: "int",    default: 50,   group: "advanced", help: "Eval cadence." },
      { name: "save-steps",       type: "int",    default: 500,  group: "advanced", help: "Checkpoint cadence." },
      { name: "logging-steps",    type: "int",    default: 10,   group: "advanced", help: "Loss-print cadence." },
      { name: "seed",             type: "int",    default: 42,   group: "advanced", help: "RNG seed." },
      { name: "gguf-quant",       type: "string", default: "q4_k_m", group: "advanced", help: "Quant for the shipped student GGUF. q4_k_m is the Android default." },
      { name: "skip-gguf",        type: "flag",   default: false, help: "Train + save adapter, skip GGUF export. Faster iteration." },
      { name: "skip-comparison",  type: "flag",   default: false, help: "Skip the auto teacher-vs-student eval after training." },
      { name: "resume",           type: "flag",   default: false, help: "Continue from the latest student checkpoint." },
      { name: "force",            type: "flag",   default: false, help: "Retrain even if the student adapter already exists." },
    ],
  },
  {
    id: "predict_one",
    label: "Predict one",
    description: "Run a single prediction through a GGUF or adapter.",
    script: "scripts/predict_one.py",
    args: [
      { name: "model",        type: "string", default: "models/student/gguf", required: true, help: "Path to a .gguf file or directory containing one." },
      { name: "max-tokens",   type: "int",    default: 512, help: "Generation cap." },
      { name: "n-ctx",        type: "int",    default: 2048, group: "advanced", help: "Context window." },
      { name: "n-gpu-layers", type: "int",    default: -1,   group: "advanced", help: "-1 = all on GPU, 0 = CPU." },
      { name: "no-grammar",   type: "flag",   default: false, help: "Disable GBNF — see what the raw model emits." },
      { name: "input",        type: "positional", default: "", required: true, help: "The transcribed transaction string to parse, e.g. \"500 rs on beer 50 rs on candy\"." },
    ],
  },
  {
    id: "export_gguf",
    label: "Export GGUF",
    description: "Re-export a trained adapter to GGUF (no retraining).",
    script: "scripts/export_gguf.py",
    args: [
      { name: "role",           type: "choice", choices: ["teacher", "student"], default: "student", required: true, help: "Which adapter to re-export — teacher (Gemma 4 E2B) or student (Gemma 3 270M)." },
      { name: "quant",          type: "string", default: "", help: "Export a single quant. Leave blank to use --quants instead." },
      { name: "quants",         type: "string", default: "q4_k_m", help: "Comma-separated list of quants in one model load — e.g. bf16,q8_0,q5_k_m,q4_k_m." },
      { name: "max-seq-length", type: "int",    default: 1024, group: "advanced", help: "Max sequence length when loading the adapter." },
      { name: "keep-existing",  type: "flag",   default: false, help: "Skip quants whose output file already exists." },
    ],
  },
];

export function getFlow(id) {
  return FLOWS.find(f => f.id === id) ?? null;
}

function formatNumber(n) {
  return Number.isInteger(n) ? String(n) : String(n);
}

export function buildArgv(flow, values, extraArgsString) {
  if (!flow) throw new Error("buildArgv: flow is required");
  const argv = [flow.script];
  const positionals = [];

  for (const spec of flow.args) {
    const raw = values?.[spec.name];
    const hasValue = raw !== undefined && raw !== null && raw !== "";

    if (spec.type === "flag") {
      if (raw === true) argv.push(`--${spec.name}`);
      continue;
    }
    if (spec.type === "positional") {
      if (hasValue) positionals.push(String(raw));
      else if (spec.required) throw new Error(`missing required positional: ${spec.name}`);
      continue;
    }
    if (!hasValue) {
      if (spec.required) throw new Error(`missing required arg: ${spec.name}`);
      continue;
    }
    if (spec.type === "choice") {
      if (!spec.choices.includes(raw)) {
        throw new Error(`invalid choice for ${spec.name}: ${raw}`);
      }
      argv.push(`--${spec.name}`, String(raw));
      continue;
    }
    if (spec.type === "int") {
      const n = parseInt(raw, 10);
      if (Number.isNaN(n)) throw new Error(`invalid int for ${spec.name}: ${raw}`);
      argv.push(`--${spec.name}`, String(n));
      continue;
    }
    if (spec.type === "float") {
      const n = Number(raw);
      if (Number.isNaN(n)) throw new Error(`invalid float for ${spec.name}: ${raw}`);
      argv.push(`--${spec.name}`, formatNumber(n));
      continue;
    }
    if (spec.type === "string") {
      argv.push(`--${spec.name}`, String(raw));
      continue;
    }
    if (spec.type === "config") {
      // UI sends a bare filename like "example_providers.json"; map to configs/<name>
      argv.push(`--${spec.name}`, `configs/${raw}`);
      continue;
    }
    throw new Error(`buildArgv: unknown arg type ${spec.type}`);
  }

  if (extraArgsString && extraArgsString.trim()) {
    const tokens = extraArgsString.trim().split(/\s+/);
    for (const t of tokens) argv.push(t);
  }

  for (const p of positionals) argv.push(p);

  return argv;
}
