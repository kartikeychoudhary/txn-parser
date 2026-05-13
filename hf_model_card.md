---
license: apache-2.0
language:
- en
- hi
base_model:
- unsloth/gemma-3-270m-it
- unsloth/gemma-4-E2B-it
library_name: gguf
pipeline_tag: text-generation
tags:
- text-generation
- gemma
- gemma-3
- gemma-4
- gguf
- finetune
- distillation
- on-device
- android
- llama-cpp
- transaction-parsing
- json-output
- structured-output
- voice-input
- expense-tracking
- code-mixed
- hinglish
model-index:
- name: txn-parser-student
  results:
  - task:
      type: text-generation
      name: Transaction parsing (JSON output)
    metrics:
    - type: json_valid_pct
      name: JSON valid (50-example eval)
      value: 94.0
    - type: schema_valid_pct
      name: Schema valid
      value: 72.0
    - type: exact_match_pct
      name: Exact match (numeric-aware)
      value: 20.0
---

# Transaction Parser — Voice → JSON (on-device)

Distilled student model that turns voice-transcribed transaction strings into
structured JSON for an Android expense-tracking app. Examples:

| Input | Output |
|---|---|
| `"500 rs on beer 50 rs on candy"` | `[{amount: 500, item: "beer", category: "Drinks", ...}, {amount: 50, item: "candy", ...}]` |
| `"do sau rupay ka chai"` | `[{amount: 200, currency: "INR", item: "chai", category: "Drinks", ...}]` |
| `"1.5k for shoes from myntra"` | `[{amount: 1500, item: "shoes", category: "Shopping", ...}]` |
| `"got my salary 50000"` | `[{amount: 50000, type: "income", category: "Income", ...}]` |

## What's in this repo

| Path | Description |
|---|---|
| `student/gguf/gemma3_text-fixed.BF16.gguf` | Lossless ref (543 MB) |
| `student/gguf/gemma3_text-fixed.Q8_0.gguf` | High quality (~290 MB) |
| **`student/gguf/gemma3_text-fixed.Q5_K_M.gguf`** | **Default for ship (260 MB)** |
| `student/gguf/gemma3_text-fixed.Q4_K_M.gguf` | Smallest but lossy on 270M (253 MB) |
| `student/adapters/` | Trained LoRA adapter (r=32, α=64) for further finetuning |
| `teacher/gguf/gemma-4-e2b-it.Q3_K_M.gguf` | Teacher (Gemma 4 E2B) used for distillation labeling |
| `teacher/adapters/` | Teacher LoRA adapter (r=16, α=32) |

## Recommended file

**`student/gguf/gemma3_text-fixed.Q5_K_M.gguf`** — 260 MB, 94% JSON valid,
runs on-device on Android via `llama.cpp` at ~150 ms per request on a modern
mid-range device.

### Evaluation (50-example smoke test)

| Build | Size | JSON valid | Schema valid | Exact match (numeric-aware) | Mean latency (A100) |
|---|---|---|---|---|---|
| fp16 adapter (ceiling) | n/a | 98% | 94% | ~48% | 1219 ms |
| BF16 GGUF (fixed) | 543 MB | 98% | 74% | 48% | 108 ms |
| Q8_0 GGUF (fixed) | ~290 MB | ~98% | ~74% | ~46% | ~120 ms |
| **Q5_K_M GGUF (fixed)** | **260 MB** | **94%** | **72%** | **20%** | **210 ms** |
| Q4_K_M GGUF (fixed) | 253 MB | 68% | 56% | 18% | 177 ms |

The "exact-match" column uses numeric-aware comparison (`100 == 100.0`).
Most "schema invalid" failures are missing-field or enum-value drift; the
category prediction is mostly diagonal in the confusion matrix.

> **Tip for Android:** always run a `JSON.parse → schema validate → fallback UI`
> pipeline. ~6% of inputs at Q5_K_M will fail to parse — handle that as
> "couldn't understand, please try again" rather than crashing.

## Usage

### `llama.cpp` / `llama-cpp-python` (Python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="gemma3_text-fixed.Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False,
)

SYSTEM_PROMPT = (
    "You convert short, possibly code-mixed (English/Hindi/Hinglish) "
    "transcribed transaction strings into a JSON object with a single "
    '"transactions" array. Each transaction has: amount (number), '
    "currency (string, default 'INR'), item (string), category (one of "
    "Food, Drinks, Groceries, Transport, Shopping, Entertainment, Bills, "
    "Health, Education, Personal, Gifts, Income, Other), type "
    "('expense' or 'income'). Output ONLY the JSON object — no prose."
)

resp = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "500 rs on beer 50 rs on candy"},
    ],
    temperature=0.0, top_p=1.0, max_tokens=512,
)
print(resp["choices"][0]["message"]["content"])
```

### Android (`llama.cpp` JNI)

1. Bundle `gemma3_text-fixed.Q5_K_M.gguf` in your app (or download on first run).
2. Use the `llama.cpp` Android example or a JNI wrapper.
3. Set the same system prompt above; user message = the voice transcript.
4. Validate output with a JSON-schema library on the parse path.

Keep the `llama_context` alive across requests — don't reload per call.

### Quick test on Linux/macOS

```bash
huggingface-cli download kartikey31/txn-parser \
    --repo-type=model --local-dir models

python -c "
from llama_cpp import Llama
llm = Llama(model_path='models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf', n_gpu_layers=-1, verbose=False)
print(llm.create_chat_completion(messages=[
    {'role':'system','content':'Output only JSON with a transactions array...'},
    {'role':'user','content':'500 rs on beer 50 rs on candy'},
], temperature=0)['choices'][0]['message']['content'])
"
```

## Training details

- **Base model**: `unsloth/gemma-3-270m-it`
- **Method**: QLoRA via Unsloth (`r=32`, `α=64`, dropout 0.0, all linear targets)
- **Train data**: 29,890 teacher-labeled examples (`data/distill/train.jsonl`)
  generated by a fine-tuned Gemma 4 E2B teacher
- **Epochs**: 2
- **Effective batch**: 128 (A100) / 16 (5060 Ti)
- **Optimizer**: AdamW 8-bit, cosine LR, peak 2e-4, warmup 3%
- **Final eval loss**: 0.099 (eval set: 300 hand-curated examples)
- **GGUF conversion**: raw `llama.cpp/convert_hf_to_gguf.py` (NOT Unsloth's wrapper),
  preserves BOS token in chat template
- **Hardware**: A100-SXM4-80GB, ~25 min total training time at batch 128

Code, dataset generation, evaluation, and conversion scripts:
https://github.com/kartikeychoudhary/txn-parser

## Categories enum

`Food, Drinks, Groceries, Transport, Shopping, Entertainment, Bills, Health, Education, Personal, Gifts, Income, Other`

## License

Apache-2.0 (matches base model). The training data is synthetic and
released under the same license.

## Citation

```
@software{txn-parser-2026,
  author  = {Kartikey Choudhary},
  title   = {Transaction Parser: Voice-to-JSON distilled model},
  year    = {2026},
  url     = {https://huggingface.co/kartikey31/txn-parser},
  note    = {Gemma 3 270M, distilled from Gemma 4 E2B teacher},
}
```
