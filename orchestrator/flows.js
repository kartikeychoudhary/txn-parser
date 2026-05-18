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
      { name: "batches",     type: "int",    default: 25,  help: "How many batches" },
      { name: "start-batch", type: "int",    default: 1,   help: "Resume from batch N (1..)" },
      { name: "model",       type: "string", default: "deepseek-v4-flash" },
      { name: "temperature", type: "float",  default: 1.0 },
      { name: "max-tokens",  type: "int",    default: 8000, group: "advanced" },
      { name: "max-retries", type: "int",    default: 5,    group: "advanced" },
      { name: "timeout",     type: "int",    default: 300,  group: "advanced", help: "Seconds" },
      { name: "base-url",    type: "string", default: "https://api.deepseek.com", group: "advanced" },
      { name: "output-dir",  type: "string", default: "data/raw", group: "advanced" },
      { name: "force",       type: "flag",   default: false, help: "Regenerate existing batches" },
    ],
  },
  {
    id: "stage_2_clean_dataset",
    label: "Stage 2 — Clean + dedupe + split",
    description: "Validate raw batches, dedupe, and split into train/eval.",
    script: "scripts/02_clean_dataset.py",
    args: [
      { name: "eval-frac",  type: "float",  default: 0.1, help: "Fraction held out for eval" },
      { name: "seed",       type: "int",    default: 42 },
      { name: "input-dir",  type: "string", default: "data/raw",   group: "advanced" },
      { name: "output-dir", type: "string", default: "data/clean", group: "advanced" },
    ],
  },
  {
    id: "stage_3_train_teacher",
    label: "Stage 3 — Train teacher",
    description: "QLoRA fine-tune of Gemma 4 E2B on data/clean/.",
    script: "scripts/03_train_teacher.py",
    args: [
      { name: "model",            type: "string", default: "unsloth/gemma-4-E2B-it" },
      { name: "epochs",           type: "float",  default: 3.0 },
      { name: "batch-size",       type: "int",    default: 4 },
      { name: "grad-accum",       type: "int",    default: 4 },
      { name: "eval-batch-size",  type: "int",    default: 8 },
      { name: "max-seq-length",   type: "int",    default: 1024 },
      { name: "max-steps",        type: "int",    default: -1, help: "-1 = full epochs" },
      { name: "lr",               type: "float",  default: 2e-4, group: "advanced" },
      { name: "warmup-ratio",     type: "float",  default: 0.03, group: "advanced" },
      { name: "lora-r",           type: "int",    default: 16,   group: "advanced" },
      { name: "lora-alpha",       type: "int",    default: 32,   group: "advanced" },
      { name: "lora-dropout",     type: "float",  default: 0.0,  group: "advanced" },
      { name: "eval-steps",       type: "int",    default: 50,   group: "advanced" },
      { name: "save-steps",       type: "int",    default: 200,  group: "advanced" },
      { name: "logging-steps",    type: "int",    default: 10,   group: "advanced" },
      { name: "seed",             type: "int",    default: 42,   group: "advanced" },
      { name: "gguf-quant",       type: "string", default: "q3_k_m", group: "advanced", help: "e.g. q3_k_m, q4_k_m, q8_0, bf16" },
      { name: "skip-gguf",        type: "flag",   default: false },
      { name: "resume",           type: "flag",   default: false },
      { name: "force",            type: "flag",   default: false },
    ],
  },
  {
    id: "stage_4_eval",
    label: "Stage 4 — Evaluate",
    description: "Run a model against data/clean/eval.jsonl.",
    script: "scripts/04_eval.py",
    args: [
      { name: "model",         type: "string", default: "models/teacher/gguf", required: true, help: ".gguf file, dir with one, or adapter dir" },
      { name: "limit",         type: "int",    default: 0, help: "0 = all" },
      { name: "name",          type: "string", default: "", help: "Override the output filename stem" },
      { name: "max-tokens",    type: "int",    default: 512 },
      { name: "batch-size",    type: "int",    default: 16, help: "Transformers backend only" },
      { name: "n-ctx",         type: "int",    default: 2048,  group: "advanced" },
      { name: "n-gpu-layers",  type: "int",    default: -1,    group: "advanced", help: "-1 = all on GPU, 0 = CPU" },
      { name: "eval-file",     type: "string", default: "data/clean/eval.jsonl", group: "advanced" },
      { name: "no-grammar",    type: "flag",   default: false, help: "Disable GBNF (debugging only)" },
    ],
  },
  {
    id: "stage_5_legacy",
    label: "Stage 5 (legacy) — Teacher-labeled distillation",
    description: "Teacher fp16 labels DeepSeek-generated synthetic inputs.",
    script: "scripts/05_generate_distillation_data.py",
    args: [
      { name: "phase",           type: "choice", choices: ["all", "inputs", "label", "eval"], default: "all" },
      { name: "n-inputs",        type: "int",    default: 30000, help: "Target unique inputs" },
      { name: "inputs-per-call", type: "int",    default: 200,   group: "advanced" },
      { name: "backend",         type: "choice", choices: ["transformers", "gguf"], default: "transformers" },
      { name: "batch-size",      type: "int",    default: 16, help: "Phase 2 transformers" },
      { name: "max-new-tokens",  type: "int",    default: 384 },
      { name: "max-seq-length",  type: "int",    default: 1024, group: "advanced" },
      { name: "limit",           type: "int",    default: 0, help: "Label first N pending" },
      { name: "model",           type: "string", default: "deepseek-v4-flash", group: "advanced", help: "Phase 1 input gen model" },
      { name: "max-tokens",      type: "int",    default: 8000, group: "advanced" },
      { name: "temperature",     type: "float",  default: 1.0,  group: "advanced" },
      { name: "max-retries",     type: "int",    default: 5,    group: "advanced" },
      { name: "timeout",         type: "int",    default: 300,  group: "advanced" },
      { name: "base-url",        type: "string", default: "https://api.deepseek.com", group: "advanced" },
      { name: "gguf-path",       type: "string", default: "", group: "advanced", help: "GGUF backend only" },
      { name: "n-gpu-layers",    type: "int",    default: -1,   group: "advanced", help: "GGUF only" },
      { name: "n-ctx",           type: "int",    default: 2048, group: "advanced", help: "GGUF only" },
      { name: "n-batch",         type: "int",    default: 512,  group: "advanced", help: "GGUF only" },
      { name: "no-mmap",         type: "flag",   default: false, group: "advanced" },
      { name: "mlock",           type: "flag",   default: false, group: "advanced" },
      { name: "retry-failed",            type: "flag", default: false },
      { name: "retry-validation-failed", type: "flag", default: false },
      { name: "force-eval-copy",         type: "flag", default: false, group: "advanced" },
    ],
  },
  {
    id: "stage_5_multiprovider",
    label: "Stage 5 — Multi-provider distillation",
    description: "DeepSeek + Gemini act as labelers via providers config.",
    script: "scripts/05_generate_distillation_data.py",
    args: [
      { name: "phase",           type: "choice", choices: ["inputs", "label"], default: "inputs" },
      { name: "provider-config", type: "config", default: "example_providers.json", required: true, help: "configs/*.json — picker below" },
      { name: "multi-provider",  type: "flag",   default: true },
      { name: "limit",           type: "int",    default: 0 },
      { name: "dry-run-quota",   type: "flag",   default: false, help: "Validate config without API calls" },
    ],
  },
  {
    id: "stage_6_train_student",
    label: "Stage 6 — Train student",
    description: "QLoRA fine-tune of Gemma 3 270M on data/distill/train.jsonl.",
    script: "scripts/06_train_student.py",
    args: [
      { name: "model",            type: "string", default: "unsloth/gemma-3-270m-it" },
      { name: "epochs",           type: "float",  default: 2.0 },
      { name: "batch-size",       type: "int",    default: 8 },
      { name: "grad-accum",       type: "int",    default: 2 },
      { name: "eval-batch-size",  type: "int",    default: 8 },
      { name: "max-seq-length",   type: "int",    default: 1024 },
      { name: "max-steps",        type: "int",    default: -1 },
      { name: "lr",               type: "float",  default: 2e-4, group: "advanced" },
      { name: "warmup-ratio",     type: "float",  default: 0.03, group: "advanced" },
      { name: "lora-r",           type: "int",    default: 32,   group: "advanced" },
      { name: "lora-alpha",       type: "int",    default: 64,   group: "advanced" },
      { name: "lora-dropout",     type: "float",  default: 0.0,  group: "advanced" },
      { name: "eval-steps",       type: "int",    default: 50,   group: "advanced" },
      { name: "save-steps",       type: "int",    default: 500,  group: "advanced" },
      { name: "logging-steps",    type: "int",    default: 10,   group: "advanced" },
      { name: "seed",             type: "int",    default: 42,   group: "advanced" },
      { name: "gguf-quant",       type: "string", default: "q4_k_m", group: "advanced" },
      { name: "skip-gguf",        type: "flag",   default: false },
      { name: "skip-comparison",  type: "flag",   default: false },
      { name: "resume",           type: "flag",   default: false },
      { name: "force",            type: "flag",   default: false },
    ],
  },
  {
    id: "predict_one",
    label: "Predict one",
    description: "Run a single prediction through a GGUF or adapter.",
    script: "scripts/predict_one.py",
    args: [
      { name: "model",        type: "string", default: "models/student/gguf", required: true, help: "GGUF file or directory" },
      { name: "max-tokens",   type: "int",    default: 512 },
      { name: "n-ctx",        type: "int",    default: 2048, group: "advanced" },
      { name: "n-gpu-layers", type: "int",    default: -1,   group: "advanced" },
      { name: "no-grammar",   type: "flag",   default: false },
      { name: "input",        type: "positional", default: "", required: true, help: "Transcribed transaction string" },
    ],
  },
  {
    id: "export_gguf",
    label: "Export GGUF",
    description: "Re-export a trained adapter to GGUF (no retraining).",
    script: "scripts/export_gguf.py",
    args: [
      { name: "role",           type: "choice", choices: ["teacher", "student"], default: "student", required: true },
      { name: "quant",          type: "string", default: "", help: "Single quant, e.g. q4_k_m" },
      { name: "quants",         type: "string", default: "q4_k_m", help: "Comma list, e.g. bf16,q8_0,q5_k_m" },
      { name: "max-seq-length", type: "int",    default: 1024, group: "advanced" },
      { name: "keep-existing",  type: "flag",   default: false },
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
