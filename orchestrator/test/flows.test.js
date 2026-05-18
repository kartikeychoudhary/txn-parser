import { test } from "node:test";
import assert from "node:assert/strict";
import { FLOWS, getFlow, buildArgv } from "../flows.js";

test("catalogue exposes all expected flows", () => {
  const ids = FLOWS.map(f => f.id);
  assert.deepEqual(ids.sort(), [
    "export_gguf",
    "predict_one",
    "stage_1_generate_raw",
    "stage_2_clean_dataset",
    "stage_3_train_teacher",
    "stage_4_eval",
    "stage_5_legacy",
    "stage_5_multiprovider",
    "stage_6_train_student",
  ]);
});

test("getFlow returns null for unknown id", () => {
  assert.equal(getFlow("nope"), null);
});

test("buildArgv: int arg only included when set", () => {
  const flow = getFlow("stage_6_train_student");
  const argv = buildArgv(flow, { "batch-size": 16 }, "");
  assert.deepEqual(argv, ["scripts/06_train_student.py", "--batch-size", "16"]);
});

test("buildArgv: flag args are omitted when false, present when true", () => {
  const flow = getFlow("stage_6_train_student");
  const off = buildArgv(flow, { resume: false }, "");
  assert.deepEqual(off, ["scripts/06_train_student.py"]);
  const on = buildArgv(flow, { resume: true }, "");
  assert.deepEqual(on, ["scripts/06_train_student.py", "--resume"]);
});

test("buildArgv: string arg quoted-safe (split on spaces in shell, we pass as one argv)", () => {
  const flow = getFlow("predict_one");
  const argv = buildArgv(flow, { model: "models/student/gguf", input: "500 rs on beer" }, "");
  assert.deepEqual(argv, [
    "scripts/predict_one.py",
    "--model", "models/student/gguf",
    "500 rs on beer",
  ]);
});

test("buildArgv: extraArgs are appended (split on whitespace, empty preserved as empty)", () => {
  const flow = getFlow("stage_4_eval");
  const argv = buildArgv(flow, { model: "models/teacher/gguf" }, "--limit 50 --no-grammar");
  assert.deepEqual(argv, [
    "scripts/04_eval.py",
    "--model", "models/teacher/gguf",
    "--limit", "50", "--no-grammar",
  ]);
});

test("buildArgv: float arg formatted without forcing decimals when integer-valued", () => {
  const flow = getFlow("stage_6_train_student");
  const argv = buildArgv(flow, { epochs: 2 }, "");
  assert.deepEqual(argv, ["scripts/06_train_student.py", "--epochs", "2"]);
});

test("buildArgv: choice arg validated", () => {
  const flow = getFlow("stage_5_legacy");
  const argv = buildArgv(flow, { phase: "label" }, "");
  assert.deepEqual(argv, ["scripts/05_generate_distillation_data.py", "--phase", "label"]);
  assert.throws(() => buildArgv(flow, { phase: "bogus" }, ""), /invalid choice/);
});
