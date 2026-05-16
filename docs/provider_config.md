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

## What loads successfully in Slice 1

- Any valid `type: fake` provider with proper fixture paths.
- Any `type: deepseek` / `gemini` / `local_teacher` provider — these become `UnimplementedProvider` and any actual generation call raises `NotImplementedError`. Dry-run-quota still works.

## Example configs

- `configs/test_providers.json` — fake-only, 10-input target, used by smoke tests.
- `configs/example_providers.json` — realistic shape with deepseek/gemini/local_teacher providers. Loads cleanly; cannot run in Slice 1.
