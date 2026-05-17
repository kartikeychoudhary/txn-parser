# Multi-Provider Generation Config

The JSON config consumed by `scripts/05_generate_distillation_data.py --provider-config <path>` controls how Stage 5 generates synthetic inputs and teacher labels across one or more LLM providers. Slice 1 only supports `--dry-run-quota`; real execution lands in Slices 2/3.

See the spec at `docs/superpowers/specs/2026-05-16-multi-provider-generation-slice1-design.md` for the full schema.

## Top-level structure

```json
{
  "version": 1,
  "input_generation":  { ... },
  "output_generation": { ... },
  "validation":        { ... },
  "rate_limits":       { ... }
}
```

`version` must be `1` if present; missing is treated as `1`.

## `input_generation`

Controls how many synthetic input strings are produced and which providers contribute.

| Field | Type | Notes |
|---|---|---|
| `enabled` | bool | If false, the phase is skipped entirely. |
| `target_inputs` | int ≥ 0 | Total unique input strings to produce. |
| `batch_size` | int ≥ 1 | Inputs requested per API call. |
| `dedupe` | bool | Global dedupe across providers (Slice 2 behavior). |
| `providers` | list | Non-empty when `enabled=true` and `target_inputs≥1`. |

## `output_generation`

Controls how teacher labels are produced for each input.

| Field | Type | Notes |
|---|---|---|
| `enabled` | bool | If false, the phase is skipped entirely. |
| `label_attempts_per_input` | int ≥ 1 | How many candidate labels to generate per input. |
| `selection_policy` | str | Currently only `"first_valid_then_score"`. |
| `providers` | list | Non-empty when `enabled=true`. |
| `provider_priority` | list[str] | Tie-breaker order; each entry must reference a declared provider. |

## `ProviderConfig`

| Field | Type | Notes |
|---|---|---|
| `name` | str | Unique within its phase. |
| `type` | one of `fake`, `deepseek`, `gemini`, `local_teacher` | Slice 1 ships only `fake`. Others become `UnimplementedProvider`. |
| `weight` | int ≥ 1 | Used for quota allocation and round-robin scheduling. |
| `threads` | int ≥ 1 | Concurrency hint for Slice 2/3. |
| `model` | str | Required for `deepseek` and `gemini`. |
| `temperature` | null or float ≥ 0 | Optional. |
| `max_tokens` | null or int ≥ 1 | Optional. |
| `max_retries` | int ≥ 0 | Default 3. |
| `structured_output` | bool | Gemini only — uses JSON-schema response API. |
| `seed` | int or null | Reserved for forward compatibility. |
| `fixture_inputs` | str | Required for `type=fake` in input phase. Path relative to config dir. |
| `fixture_labels` | str | Required for `type=fake` in output phase. Path relative to config dir. |

## `validation`

| Field | Type | Notes |
|---|---|---|
| `schema` | bool | Run JSON-schema check (Slice 3 default true). |
| `semantic_validator` | bool | Run the validator from the prior slice. |
| `reject_invalid` | bool | Drop rows that fail validation (vs. flag-and-keep). |
| `retry_invalid_with_stricter_prompt` | bool | Try a stricter repair prompt on failure. |
| `max_repair_attempts` | int ≥ 0 | Cap on repair attempts. |

## `rate_limits`

| Field | Type | Notes |
|---|---|---|
| `global_max_workers` | int ≥ 1 | Cap across all providers. |
| `write_flush_every` | int ≥ 1 | Flush to JSONL every N accepted rows. |

## Fixture path resolution

Relative paths in `fixture_inputs` and `fixture_labels` resolve **relative to the directory containing the config file**. Absolute paths are used as-is. The loader validates the resolved path exists at load time.

## What loads and runs

Config loading and `--dry-run-quota` do not construct providers and do not require
SDK/API/GPU availability. Actual execution constructs providers and requires the
relevant SDKs, keys, or local model assets.

**Slice 1 (configuration + dry-run):**
- Any `type: fake` provider with proper fixture paths loads and runs.
- Any `type: deepseek` / `gemini` / `local_teacher` provider loads cleanly for
  config parsing and `--dry-run-quota` (no provider construction at that stage).

**Slice 2 (real input generation):**
- `type: deepseek` and `type: gemini` providers run real API calls when invoked
  via `--phase inputs --provider-config <path> --multi-provider`. Provider
  construction imports the relevant SDK (`openai` / `google-genai`) and reads
  `DEEPSEEK_API_KEY` / `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) env vars.

**Slice 3 (real output labeling):**
- `--phase label --provider-config <path> --multi-provider` runs the validator-
  gated multi-provider labeling loop.
- `type: local_teacher` providers require `model` (path to the adapter directory).
  Construction is lazy: the fp16 backend is built on first `generate_label` call,
  which imports `unsloth` + `torch` and loads model weights.
- Gemini's `structured_output: true` activates response_schema + response_mime_type
  for label generation only. Input generation ignores it.
- `validation.retry_invalid_with_stricter_prompt: true` plus `max_repair_attempts: N`
  enables the repair loop: when all initial attempts fail, re-prompt the highest-
  priority provider with a stricter repair prompt up to N times.
- `--phase eval` is legacy-only — omit `--multi-provider` for eval. `--phase all`
  exits via `parser.error`.

## Example configs

- `configs/test_providers.json` — fake-only, 10-input target, used by smoke tests.
- `configs/example_providers.json` — realistic shape with deepseek/gemini/local_teacher providers. Loads cleanly; cannot run in Slice 1.

## Metrics (Slice 4)

Multi-provider runs (`--multi-provider --phase {inputs,label}`) emit
`data/distill/metrics.json` and log a compact stdout summary.

Per-model USD prices come from `configs/prices.json`. The lookup order is:

1. `<provider_type>:<model>` (e.g., `gemini:gemini-2.5-flash`)
2. `<model>` (e.g., `gemini-2.5-flash`)
3. `<provider_type>` (e.g., `gemini`)
4. `default`

Each entry has the shape:

```json
{"input_per_million": 0.30, "output_per_million": 2.50}
```

If `configs/prices.json` is missing, the run continues with $0.00 cost and a
warning log line. Costs are estimates — the script never contacts a live
pricing API.
