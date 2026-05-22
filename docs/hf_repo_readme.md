---
license: apache-2.0
tags:
  - text-generation
  - lora
  - qlora
  - gguf
  - transaction-parser
  - on-device
language:
  - en
  - hi
library_name: peft
pipeline_tag: text-generation
---

# txn-parser

QLoRA fine-tunes of three small base models for extracting structured
transaction data (amount, currency, item, category, type) from free-form
Indian-English / code-switched speech and text. Trained on the same
93k-row teacher-labeled dataset, validator-gated, and quantized to 5 GGUF
tiers each.

This repo holds **all three published students in side-by-side
subfolders** so a downstream app can pick its size/quality tradeoff:

| Base model | Subfolder | Params | Best on-device pick |
|---|---|---:|---|
| `unsloth/gemma-3-270m-it` | [`gemma-3-270m/`](./tree/main/gemma-3-270m) | 270M | `Q5_K_M` (260 MB, 99.7% schema) |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | [`smollm2-360m/`](./tree/main/smollm2-360m) | 360M | `Q4_K_M` (271 MB, 100% schema) |
| `Qwen/Qwen3-0.6B` | [`qwen3-0.6b/`](./tree/main/qwen3-0.6b) | 600M | `Q4_K_M` (397 MB, 100% schema, **best exact match**) |

Each subfolder contains:

```
<short>/
├── adapters/                          # PEFT LoRA adapter (load on top of base)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── gguf/                              # 5 merged quantizations, ship-ready
│   ├── txn-parser-<short>-F16.gguf
│   ├── txn-parser-<short>-Q8_0.gguf
│   ├── txn-parser-<short>-Q6_K.gguf
│   ├── txn-parser-<short>-Q5_K_M.gguf
│   └── txn-parser-<short>-Q4_K_M.gguf
└── README.md                          # model card with the exact SYSTEM_PROMPT
```

## Eval (300-example held-out set, grammar-constrained decoding)

| Model | Quant | Size | JSON valid | Schema valid | Exact match | Amount exact | Mean ms |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma-3-270m  | Q5_K_M | 260 MB  | 99.7%  | **99.7%**  | 51.0% | 84.7% | 1788 |
| gemma-3-270m  | Q4_K_M | 253 MB  | 93.3%  |   93.3%   | 48.3% | 80.7% | 2660 |
| smollm2-360m  | Q5_K_M | 290 MB  | 100.0% | **100.0%** | 52.7% | 89.0% | 996  |
| smollm2-360m  | Q4_K_M | 271 MB  | 100.0% | **100.0%** | 53.3% | 87.3% | 978  |
| qwen3-0.6b    | Q5_K_M | 444 MB  | 100.0% | **100.0%** | 60.0% | 91.3% | 857  |
| qwen3-0.6b    | Q4_K_M | 397 MB  | 100.0% | **100.0%** | 60.0% | 90.7% | 885  |

(Showing Q5_K_M + Q4_K_M only — full 15-row table for all 5 quants is in
the source repo's `eval_results/REPORT.md`.)

**Recommendations:**

- **Best accuracy on-device:** `qwen3-0.6b-Q4_K_M` (397 MB, 60% exact, ~885 ms)
- **Smallest ship size:** `smollm2-360m-Q4_K_M` (271 MB, 53% exact, ~978 ms)
- **Lowest latency:** `qwen3-0.6b-Q8_0` (639 MB, 59% exact, ~851 ms)
- **Avoid:** `gemma-3-270m-Q4_K_M` — quality cliff vs Q5_K_M (93% → 99.7% schema)

## Quick download

```bash
# Just one quant of one model (small)
huggingface-cli download kartikey31/txn-parser \
    smollm2-360m/gguf/txn-parser-smollm2-360m-Q4_K_M.gguf --local-dir .

# Everything (3 models × 5 quants, ~5 GB)
huggingface-cli download kartikey31/txn-parser --local-dir ./txn-parser
```

Or from Python:

```python
from huggingface_hub import hf_hub_download

gguf = hf_hub_download(
    "kartikey31/txn-parser",
    "qwen3-0.6b/gguf/txn-parser-qwen3-0.6b-Q4_K_M.gguf",
)
```

## Inference (llama-cpp-python)

```python
from llama_cpp import Llama

# Same SYSTEM_PROMPT for ALL three models — they were trained with it.
SYSTEM_PROMPT = """You convert voice-transcribed transaction descriptions into structured JSON.

Output ONLY a JSON object with this schema, no other text:
{"transactions":[{"amount":<number>,"currency":"INR"|"USD","item":"<lowercase singular noun phrase>","category":"<enum>","type":"expense"|"income"}]}

Categories: Food, Drinks, Groceries, Transport, Shopping, Entertainment, Bills, Health, Education, Personal, Gifts, Income, Other.

Rules:
- Currency defaults to INR. Use USD only when the input explicitly says "dollars" or contains "$".
- Amounts: "k" = ×1000, "hazaar" = ×1000, "sau" = ×100, "lakh" = ×100000. Convert number-words ("five hundred") to digits.
- type is "expense" by default; "income" only for explicit salary, cashback, refund, gift received, payment received.
- For disfluencies and corrections ("500 wait no 600"), output the CORRECTED amount only.
- For ambiguous items ("that thing", "stuff"), use item "unspecified" and category "Other".
- Item field: lowercase singular noun phrase ("uber ride", "beer", "chai" — not "Beers" or "Uber").
- Multi-transaction inputs become multiple array entries in spoken order.
- Category heuristics: uber/ola/auto/petrol/bus/metro → Transport; beer/wine/chai/coffee/juice → Drinks; rent/electricity/wifi/recharge/gas → Bills; movie/netflix/concert → Entertainment; doctor/medicine/hospital → Health."""

llm = Llama(
    model_path="txn-parser-qwen3-0.6b-Q4_K_M.gguf",
    n_gpu_layers=-1, n_ctx=1024,
)
out = llm.create_chat_completion(messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "200 ka samosa and 50 chai"},
], temperature=0.0)
print(out["choices"][0]["message"]["content"])
# {"transactions":[{"amount":200,"currency":"INR","item":"samosa",...}, ...]}
```

For Android / on-device deployment guidance (recommended llama.cpp
params, battery checklist, Kotlin POC), see the [training pipeline
README](https://github.com/kartikeychoudhary/txn-parser#android-deployment).

## Training pipeline

Full reproduction pipeline (data generation → teacher training →
distillation → multi-model student training → multi-quant export → eval
report) lives at
[`github.com/kartikeychoudhary/txn-parser`](https://github.com/kartikeychoudhary/txn-parser).

A single command reproduces all three models from a fresh checkout:

```bash
git clone https://github.com/kartikeychoudhary/txn-parser
cd txn-parser
bash setup.sh
python scripts/05_generate_distillation_data.py --phase eval --force-eval-copy
python scripts/train_and_publish.py    # trains gemma, smollm, qwen; publishes here
python scripts/eval_all_quants.py      # regenerates the eval table
```

## License

Apache 2.0 (matches all three base model licenses).
