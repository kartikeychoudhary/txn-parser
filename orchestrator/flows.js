// flows.js — curated pipeline flow catalogue.
// Each flow declares its script path and the args we expose in the UI.
// buildArgv translates UI form values into a Python argv array.

export const FLOWS = [
  {
    id: "stage_1_generate_raw",
    label: "Stage 1 — Generate raw dataset",
    description: "DeepSeek synthesizes raw transaction examples into data/raw/.",
    script: "scripts/01_generate_dataset.py",
    args: [
      { name: "batches",     type: "int",    default: 25 },
      { name: "start-batch", type: "int",    default: 1 },
      { name: "model",       type: "string", default: "deepseek-chat" },
      { name: "temperature", type: "float",  default: 1.0 },
      { name: "force",       type: "flag",   default: false },
    ],
  },
  {
    id: "stage_2_clean_dataset",
    label: "Stage 2 — Clean + dedupe + split",
    description: "Validate raw batches, dedupe, and split into train/eval.",
    script: "scripts/02_clean_dataset.py",
    args: [
      { name: "eval-frac", type: "float", default: 0.1 },
      { name: "seed",      type: "int",   default: 42 },
    ],
  },
  {
    id: "stage_3_train_teacher",
    label: "Stage 3 — Train teacher",
    description: "QLoRA fine-tune of Gemma 4 E2B on data/clean/.",
    script: "scripts/03_train_teacher.py",
    args: [
      { name: "batch-size",      type: "int",   default: 4 },
      { name: "grad-accum",      type: "int",   default: 4 },
      { name: "eval-batch-size", type: "int",   default: 8 },
      { name: "epochs",          type: "float", default: 3.0 },
      { name: "max-seq-length",  type: "int",   default: 1024 },
      { name: "max-steps",       type: "int",   default: -1 },
      { name: "resume",          type: "flag",  default: false },
      { name: "skip-gguf",       type: "flag",  default: false },
      { name: "force",           type: "flag",  default: false },
    ],
  },
  {
    id: "stage_4_eval",
    label: "Stage 4 — Evaluate",
    description: "Run a model against data/clean/eval.jsonl.",
    script: "scripts/04_eval.py",
    args: [
      { name: "model",      type: "string", default: "models/teacher/gguf", required: true },
      { name: "limit",      type: "int",    default: 0 },
      { name: "batch-size", type: "int",    default: 16 },
      { name: "no-grammar", type: "flag",   default: false },
    ],
  },
  {
    id: "stage_5_legacy",
    label: "Stage 5 (legacy) — Teacher-labeled distillation",
    description: "Teacher fp16 labels DeepSeek-generated synthetic inputs.",
    script: "scripts/05_generate_distillation_data.py",
    args: [
      { name: "phase",    type: "choice", choices: ["all", "inputs", "label", "eval"], default: "all" },
      { name: "n-inputs", type: "int",    default: 30000 },
      { name: "batch-size", type: "int",  default: 16 },
      { name: "limit",    type: "int",    default: 0 },
      { name: "retry-failed", type: "flag", default: false },
      { name: "retry-validation-failed", type: "flag", default: false },
    ],
  },
  {
    id: "stage_5_multiprovider",
    label: "Stage 5 — Multi-provider distillation",
    description: "DeepSeek + Gemini act as labelers via providers config.",
    script: "scripts/05_generate_distillation_data.py",
    args: [
      { name: "phase",           type: "choice", choices: ["inputs", "label"], default: "inputs" },
      { name: "provider-config", type: "string", default: "configs/example_providers.json", required: true },
      { name: "multi-provider",  type: "flag",   default: true },
      { name: "limit",           type: "int",    default: 0 },
      { name: "dry-run-quota",   type: "flag",   default: false },
    ],
  },
  {
    id: "stage_6_train_student",
    label: "Stage 6 — Train student",
    description: "QLoRA fine-tune of Gemma 3 270M on data/distill/train.jsonl.",
    script: "scripts/06_train_student.py",
    args: [
      { name: "batch-size",      type: "int",   default: 8 },
      { name: "grad-accum",      type: "int",   default: 2 },
      { name: "eval-batch-size", type: "int",   default: 8 },
      { name: "epochs",          type: "float", default: 2.0 },
      { name: "max-steps",       type: "int",   default: -1 },
      { name: "resume",          type: "flag",  default: false },
      { name: "skip-gguf",       type: "flag",  default: false },
      { name: "skip-comparison", type: "flag",  default: false },
    ],
  },
  {
    id: "predict_one",
    label: "Predict one",
    description: "Run a single prediction through a GGUF or adapter.",
    script: "scripts/predict_one.py",
    args: [
      { name: "model",      type: "string",     default: "models/student/gguf", required: true },
      { name: "no-grammar", type: "flag",       default: false },
      { name: "input",      type: "positional", default: "", required: true },
    ],
  },
  {
    id: "export_gguf",
    label: "Export GGUF",
    description: "Re-export a trained adapter to GGUF (no retraining).",
    script: "scripts/export_gguf.py",
    args: [
      { name: "role",   type: "choice", choices: ["teacher", "student"], default: "student", required: true },
      { name: "quants", type: "string", default: "q4_k_m" },
    ],
  },
];

export function getFlow(id) {
  return FLOWS.find(f => f.id === id) ?? null;
}

function formatNumber(n) {
  // Avoid forcing decimals on integer-valued floats: 2.0 -> "2", 2.5 -> "2.5"
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
    throw new Error(`buildArgv: unknown arg type ${spec.type}`);
  }

  // Append extra args (whitespace-split, empty tokens dropped)
  if (extraArgsString && extraArgsString.trim()) {
    const tokens = extraArgsString.trim().split(/\s+/);
    for (const t of tokens) argv.push(t);
  }

  // Positionals come last
  for (const p of positionals) argv.push(p);

  return argv;
}
